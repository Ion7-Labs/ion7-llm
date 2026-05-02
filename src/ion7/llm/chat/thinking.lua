--- @module ion7.llm.chat.thinking
--- @author  ion7 / Ion7 Project Contributors
---
--- State machine that demultiplexes a streaming token sequence into
--- two channels — assistant `content` and `<think>...</think>`
--- reasoning. Used by the engine's hot loop so the UI sees plain
--- assistant text on one channel while the orchestration layer can
--- log / cap the reasoning on the other.
---
--- Two design points worth documenting :
---
--- ── Tail buffers, not running concat ─────────────────────────────
--- A token can split a `<think>` tag mid-character ("<th", "ink>").
--- We keep two tiny tail buffers — the last seven characters seen
--- outside a thinking block (`<think>` is 7 chars long) and the last
--- eight characters inside one (`</think>` is 8) — and re-scan only
--- the tail+new piece on each step. Net cost : O(piece) per token,
--- no growing concat.
---
--- ── Interleaved-thinking-aware ───────────────────────────────────
--- v2 reasoning models emit a `<think>` block then call a tool, then
--- think again about the result, then call another tool. The block
--- the engine collects on each turn becomes part of the next turn's
--- assistant message (`messages[i].thinking`) so the chat template
--- can re-render the full envelope. Stripping or summarising the
--- block on the way out would lose that context.

-- Hot-path locals : thinking:feed runs once per generated token, so
-- caching the string library entry points as upvalues lets LuaJIT
-- skip the metatable lookup on every iteration.
local string_find   = string.find
local string_sub    = string.sub
local table_concat  = table.concat

local Thinking = {}
Thinking.__index = Thinking

local OPEN_TAG  = "<think>"
local CLOSE_TAG = "</think>"
local OPEN_TAG_LEN  = #OPEN_TAG
local CLOSE_TAG_LEN = #CLOSE_TAG
local OPEN_TAIL_MAX  = OPEN_TAG_LEN  - 1
local CLOSE_TAIL_MAX = CLOSE_TAG_LEN - 1

--- @class ion7.llm.chat.Thinking
--- @field _in_think    boolean
--- @field _think_text  string[]    Accumulator for the active think block.
--- @field _open_tail   string      Last 6 chars seen outside a think block.
--- @field _close_tail  string      Last 7 chars seen inside a think block.
--- @field _tok_count   integer     Tokens emitted inside the active block.
--- @field _all_think   string[]    Concatenation of every closed block.

--- @return ion7.llm.chat.Thinking
function Thinking.new()
    return setmetatable({
        _in_think    = false,
        _think_text  = {},
        _open_tail   = "",
        _close_tail  = "",
        _tok_count   = 0,
        _all_think   = {},
    }, Thinking)
end

--- Reset to the initial state — call between generations. Returns
--- the thinking text accumulated since the last reset (useful when
--- the upstream loop wants to clear the buffer AND keep the trace).
--- @return string
function Thinking:reset()
    local out = self:thinking()
    self._in_think    = false
    self._think_text  = {}
    self._open_tail   = ""
    self._close_tail  = ""
    self._tok_count   = 0
    self._all_think   = {}
    return out
end

--- True when the buffer is currently inside a `<think>` block.
--- @return boolean
function Thinking:in_think() return self._in_think end

--- Total tokens consumed inside the active block. Resets to 0 each
--- time a block closes. Used by the reasoning-budget guard.
--- @return integer
function Thinking:active_token_count() return self._tok_count end

--- Concatenated text of every closed `<think>` block since the last
--- `:reset`, plus whatever is currently buffered in the active one.
--- @return string?
function Thinking:thinking()
    local at, tt = self._all_think, self._think_text
    local n_at, n_tt = #at, #tt
    if n_at == 0 and n_tt == 0 then return nil end
    if n_at == 0 then return table_concat(tt) end
    if n_tt == 0 then return table_concat(at) end
    return table_concat(at) .. table_concat(tt)
end

--- Feed a decoded piece. Returns one of the three transitions :
---
---   - `"content"` — emit `text` on the assistant channel.
---   - `"thinking"`— emit `text` on the reasoning channel.
---   - `"split"`   — the piece straddles the `<think>` open tag :
---                    `prefix` is content, `suffix` is thinking.
---                    Same for `</think>` in the other direction.
---
--- @param  piece string
--- @return string  kind  `"content" | "thinking" | "split"`
--- @return string         text or prefix.
--- @return string?        suffix on a `"split"` transition.
-- Slice `piece` around a tag detected at position `idx` in
-- `tail .. piece`. Returns `(before, after)` — `before` is the part
-- of the piece that lies BEFORE the tag (empty when the tag started
-- inside the rolling tail), `after` is the part that lies AFTER the
-- tag (empty when the piece ends mid-tag).
local function slice_around(tail_len, piece, piece_len, idx, tag_len)
    local before_end = idx - 1 - tail_len
    if before_end < 0 then before_end = 0 end
    local before = string_sub(piece, 1, before_end)
    local after_start = idx + tag_len - tail_len
    if after_start < 1 then after_start = 1 end
    local after
    if after_start > piece_len then
        after = ""
    else
        after = string_sub(piece, after_start)
    end
    return before, after
end

function Thinking:feed(piece)
    local piece_len = #piece
    if not self._in_think then
        local tail = self._open_tail
        local check = tail .. piece
        local idx = string_find(check, OPEN_TAG, 1, true)
        if idx then
            local prefix, body = slice_around(#tail, piece, piece_len,
                                              idx, OPEN_TAG_LEN)
            self._in_think     = true
            self._open_tail    = ""
            self._close_tail   = ""
            self._tok_count    = 1
            if body ~= "" then
                local tt = self._think_text
                tt[#tt + 1] = body
            end
            return "split", prefix, body
        end
        self._open_tail = string_sub(check, -OPEN_TAIL_MAX)
        return "content", piece
    end

    -- Inside a think block.
    self._tok_count = self._tok_count + 1
    local tail  = self._close_tail
    local check = tail .. piece
    local idx   = string_find(check, CLOSE_TAG, 1, true)
    if idx then
        local thinking_body, after_tag =
            slice_around(#tail, piece, piece_len, idx, CLOSE_TAG_LEN)
        local tt = self._think_text
        if thinking_body ~= "" then tt[#tt + 1] = thinking_body end
        local at = self._all_think
        at[#at + 1] = table_concat(tt)
        self._think_text = {}
        self._in_think   = false
        self._open_tail  = ""
        self._close_tail = ""
        self._tok_count  = 0
        if after_tag == "" then
            return "thinking", thinking_body
        end
        return "split", thinking_body, after_tag
    end

    self._close_tail = string_sub(check, -CLOSE_TAIL_MAX)
    local tt = self._think_text
    tt[#tt + 1] = piece
    return "thinking", piece
end

--- Force-close the active think block. Used by the reasoning-budget
--- guard when the model exceeds its allotment without emitting
--- `</think>`. Returns the body collected so far.
--- @return string
function Thinking:force_close()
    if not self._in_think then return "" end
    local body = table.concat(self._think_text)
    self._all_think[#self._all_think + 1] = body
    self._think_text  = {}
    self._in_think    = false
    self._open_tail   = ""
    self._close_tail  = ""
    self._tok_count   = 0
    return body
end

return Thinking
