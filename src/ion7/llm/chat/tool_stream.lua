--- @module ion7.llm.chat.tool_stream
--- @author  ion7 / Ion7 Project Contributors
---
--- State machine that detects tool-call markers inside the streaming
--- assistant text and emits incremental `tool_call_delta` /
--- `tool_call_done` chunks WITHOUT waiting for the full generation.
---
--- The chat-template-specific tool-call envelopes vary across model
--- families. We support three concrete formats out of the box ; a
--- consumer that needs another format can pass `opts.markers`.
---
---   format        open marker         args delimiter      close marker
---   ─────────────────────────────────────────────────────────────────
---   "openai"     `<tool_call>`        in JSON `arguments`  `</tool_call>`
---   "mistral"    `[TOOL_CALLS]`       `[ARGS]`             newline / EOG
---   "qwen"       `<|tool_call_begin|>` `<|tool_arg_begin|>` `<|tool_call_end|>`
---
--- Detection runs on the rendered piece string AFTER the thinking
--- demux has classified it as `content`. While we are inside a tool
--- call, the bytes go through `partial_json` so we can yield typed
--- argument deltas as soon as the brace counter closes.
---
--- The state machine is tail-buffer-bounded — the largest open marker
--- is `<|tool_call_begin|>` (20 chars), so we keep the last 19 chars
--- of content seen, plus the newest piece, and scan that. Cost is
--- O(piece) per token and zero-allocation in the steady state.

local PartialJson = require "ion7.llm.util.partial_json"

local find = string.find
local sub  = string.sub

local M = {}

--- @class ion7.llm.chat.tool_stream.Markers
--- @field open      string   Tag that starts a tool call (e.g. `<tool_call>`).
--- @field args_at   string?  Optional separator between name and arguments.
--- @field close     string?  Tag that ends a tool call. Nil = until newline.
--- @field name_re   string?  Lua pattern that captures the tool name.

--- Bundled marker presets for the major chat-template tool formats.
M.formats = {
    openai = {
        open    = "<tool_call>",
        close   = "</tool_call>",
        -- OpenAI / Hermes : `<tool_call>{"name":"x","arguments":{...}}</tool_call>`
        -- → name + arguments live inside one JSON object.
        json_envelope = true,
    },

    qwen = {
        open    = "<|tool_call_begin|>",
        args_at = "<|tool_arg_begin|>",
        close   = "<|tool_call_end|>",
        json_envelope = false,
    },

    mistral = {
        open    = "[TOOL_CALLS]",
        args_at = "[ARGS]",
        close   = nil,    -- ends at next newline / next [TOOL_CALLS] / EOG
        json_envelope = false,
    },
}

--- Build a tool-stream state machine. The default format probes for
--- every known marker — that way a consumer who does not know which
--- tool format the model will emit still gets `tool_call_delta`
--- chunks when the model picks any of them.
---
--- @param  opts table?
---   `format`  (string?, default `"auto"`)  `"auto" | "openai" | "qwen" | "mistral"`.
---   `markers` (Markers?)                   Override the active format.
--- @return ion7.llm.chat.tool_stream
function M.new(opts)
    opts = opts or {}

    local active
    if opts.markers then
        active = { opts.markers }
    elseif opts.format and opts.format ~= "auto" then
        local m = M.formats[opts.format]
        assert(m, "[ion7.llm.chat.tool_stream] unknown format : " .. tostring(opts.format))
        active = { m }
    else
        -- Auto : probe every shipped format. The first to trigger wins.
        active = {
            M.formats.openai,
            M.formats.qwen,
            M.formats.mistral,
        }
    end

    -- Pick the longest tail buffer needed for any open marker, minus 1.
    local max_open = 0
    for i = 1, #active do
        local n = #active[i].open
        if n > max_open then max_open = n end
    end

    return setmetatable({
        _formats     = active,
        _state       = "idle",  -- "idle" | "in_call" | "in_args"
        _open_tail   = "",
        _open_max    = max_open - 1,
        _active_fmt  = nil,    -- the matched format struct
        _name_buf    = {},
        _args_pj     = nil,    -- PartialJson buffer for arguments
        _args_buf    = nil,    -- raw bytes of arguments (non-JSON formats)
        _call_id     = 0,
        _calls_done  = {},     -- list of finished `{ id, name, arguments }`
        _last_yielded_args = "",
    }, { __index = M })
end

-- ── Internal helpers ─────────────────────────────────────────────────────

-- Try every active format on `check`. Returns the format struct +
-- start index of the open marker on first hit, nil otherwise.
local function probe_open(formats, check)
    for i = 1, #formats do
        local f = formats[i]
        local idx = find(check, f.open, 1, true)
        if idx then return f, idx end
    end
    return nil, nil
end

-- Slice piece around an open marker that lies in `tail .. piece` at
-- 1-based position `idx`. Returns the part of `piece` that falls
-- BEFORE the marker (still content) and the part that falls AFTER
-- (now inside the tool call). Mirrors `chat.thinking.slice_around`.
local function slice_after_open(tail, piece, idx, marker_len)
    local L = #tail
    local before_end = idx - 1 - L
    if before_end < 0 then before_end = 0 end
    local before = sub(piece, 1, before_end)
    local after_start = idx + marker_len - L
    if after_start < 1 then after_start = 1 end
    local after = after_start > #piece and "" or sub(piece, after_start)
    return before, after
end

-- ── Public API ───────────────────────────────────────────────────────────

--- Feed one piece of content (NOT thinking — caller is responsible for
--- routing thinking pieces elsewhere). Returns a list of typed events :
---
---   { kind = "content",         text = "..." }
---   { kind = "tool_call_delta", call_id, name, args_partial }
---   { kind = "tool_call_done",  call_id, call = { id, name, arguments } }
---
--- An empty list means "nothing crossed a marker, just drop the piece
--- on the content channel as-is".
---
--- @param  piece string
--- @return table[]
function M:feed(piece)
    local out = {}

    -- ── State : idle (looking for an open marker) ───────────────────────
    if self._state == "idle" then
        local check = self._open_tail .. piece
        local fmt, idx = probe_open(self._formats, check)
        if fmt then
            local before, after = slice_after_open(
                self._open_tail, piece, idx, #fmt.open)
            if #before > 0 then
                out[#out + 1] = { kind = "content", text = before }
            end
            -- Transition to in_call. Allocate a fresh buffer set.
            self._state      = "in_call"
            self._active_fmt = fmt
            self._open_tail  = ""
            self._name_buf   = {}
            self._args_pj    = nil
            self._args_buf   = nil
            self._last_yielded_args = ""
            -- Recurse into the new state with the post-marker remainder.
            if after ~= "" then
                local nested = self:feed(after)
                for i = 1, #nested do out[#out + 1] = nested[i] end
            end
            return out
        end
        -- No marker — keep last (open_max) chars in the tail and emit
        -- the piece on the content channel.
        if self._open_max > 0 then
            self._open_tail = sub(check, -self._open_max)
        end
        if #piece > 0 then
            out[#out + 1] = { kind = "content", text = piece }
        end
        return out
    end

    -- ── State : in_call (collecting name) ───────────────────────────────
    if self._state == "in_call" then
        local fmt = self._active_fmt
        if fmt.json_envelope then
            -- OpenAI-style : everything is JSON inside the open / close
            -- markers. Switch straight to in_args, the args buffer
            -- absorbs the whole envelope.
            self._state   = "in_args"
            self._args_pj = PartialJson.new()
            local nested  = self:feed(piece)
            for i = 1, #nested do out[#out + 1] = nested[i] end
            return out
        end
        -- Qwen / Mistral : name comes first, then args_at delimiter.
        local args_at = fmt.args_at
        if args_at then
            local pos = find(piece, args_at, 1, true)
            if pos then
                self._name_buf[#self._name_buf + 1] = sub(piece, 1, pos - 1)
                local rest = sub(piece, pos + #args_at)
                self._state   = "in_args"
                self._args_pj = PartialJson.new()
                if rest ~= "" then
                    local nested = self:feed(rest)
                    for i = 1, #nested do out[#out + 1] = nested[i] end
                end
                return out
            end
        end
        self._name_buf[#self._name_buf + 1] = piece
        return out
    end

    -- ── State : in_args (collecting argument bytes / JSON) ──────────────
    if self._state == "in_args" then
        local fmt = self._active_fmt
        local close = fmt.close

        -- Did this piece carry the close marker ?
        if close then
            local pos = find(piece, close, 1, true)
            if pos then
                local args_chunk = sub(piece, 1, pos - 1)
                if #args_chunk > 0 then
                    self._args_pj:append(args_chunk)
                end
                self:_yield_partial_args(out)
                self:_close_call(out)
                local rest = sub(piece, pos + #close)
                if rest ~= "" then
                    local nested = self:feed(rest)
                    for i = 1, #nested do out[#out + 1] = nested[i] end
                end
                return out
            end
        elseif fmt.open then
            -- Mistral has no close marker ; a newline OR another open
            -- marker terminates the call.
            local nl_pos = find(piece, "\n", 1, true)
            local another = find(piece, fmt.open, 1, true)
            if nl_pos or another then
                local end_pos = nl_pos or another
                local args_chunk = sub(piece, 1, end_pos - 1)
                if #args_chunk > 0 then
                    self._args_pj:append(args_chunk)
                end
                self:_yield_partial_args(out)
                self:_close_call(out)
                local rest = sub(piece, end_pos)  -- include the marker
                if rest ~= "" then
                    local nested = self:feed(rest)
                    for i = 1, #nested do out[#out + 1] = nested[i] end
                end
                return out
            end
        end

        -- No close yet — accumulate and emit a delta if we have new bytes.
        if #piece > 0 then
            self._args_pj:append(piece)
            self:_yield_partial_args(out)
        end
        return out
    end

    return out
end

-- Emit a tool_call_delta chunk for any new argument bytes since the
-- last emit. We emit the running raw string, not just the diff — the
-- consumer can compute the diff from `args_partial` if needed, but
-- having the cumulative buffer matches the OpenAI streaming protocol.
function M:_yield_partial_args(out)
    if not self._args_pj then return end
    local s = self._args_pj:string()
    if s == self._last_yielded_args then return end
    self._last_yielded_args = s

    local name = (self._active_fmt and self._active_fmt.json_envelope)
        and "" -- name is buried in the JSON ; revealed at close
        or table.concat(self._name_buf)

    out[#out + 1] = {
        kind         = "tool_call_delta",
        call_id      = tostring(self._call_id),
        name         = #name > 0 and name or nil,
        args_partial = s,
    }
end

-- Finalise the active call and emit a tool_call_done chunk.
function M:_close_call(out)
    if not self._args_pj then return end
    local raw = self._args_pj:string()
    local name, args

    if self._active_fmt and self._active_fmt.json_envelope then
        -- OpenAI envelope : raw is the WHOLE JSON `{"name":..,"arguments":..}`.
        local parsed = self._args_pj:value()
        if type(parsed) == "table" then
            name = parsed.name
            args = parsed.arguments or {}
        end
    else
        name = table.concat(self._name_buf):gsub("^%s+", ""):gsub("%s+$", "")
        args = self._args_pj:value() or raw
    end

    local call = {
        id        = tostring(self._call_id),
        name      = name or "",
        arguments = args,
    }
    self._calls_done[#self._calls_done + 1] = call
    out[#out + 1] = { kind = "tool_call_done", call_id = call.id, call = call }

    -- Reset for next call.
    self._call_id    = self._call_id + 1
    self._state      = "idle"
    self._active_fmt = nil
    self._name_buf   = {}
    self._args_pj    = nil
    self._args_buf   = nil
    self._last_yielded_args = ""
    self._open_tail  = ""
end

--- True when the SM is currently inside a tool call (between open and
--- close markers).
--- @return boolean
function M:in_tool_call() return self._state ~= "idle" end

--- Snapshot of every completed tool call since the last `:reset`.
--- @return table[]
function M:done_calls() return self._calls_done end

--- Reset to the initial state.
function M:reset()
    self._state            = "idle"
    self._open_tail        = ""
    self._active_fmt       = nil
    self._name_buf         = {}
    self._args_pj          = nil
    self._args_buf         = nil
    self._call_id          = 0
    self._calls_done       = {}
    self._last_yielded_args = ""
end

return M
