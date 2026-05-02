--- @module ion7.llm.chat.parse
--- @author  ion7 / Ion7 Project Contributors
---
--- Post-generation parser. Takes the raw assistant text the model
--- produced for one turn and demultiplexes it into the three logical
--- channels : `content`, `thinking`, `tool_calls`.
---
--- Two-stage strategy :
---
---   1. Try the bridge's `ion7_chat_parse` (libcommon's
---      `common_chat_parse`). It auto-detects the chat-template
---      format and handles every model whose template is registered
---      upstream — typically OpenAI / Llama / Hermes / DeepSeek.
---
---   2. If the bridge returns `has_tools = 0` AND the raw text still
---      contains a recognisable tool-call marker (Mistral
---      `[TOOL_CALLS]`, Qwen `<|tool_call_begin|>`, generic
---      `<tool_call>`), fall back to a pure-Lua scan that mirrors the
---      `chat.tool_stream` marker set. Catches the cases where
---      libcommon does not recognise a non-canonical format.
---
--- The fallback is post-generation, never on the hot path. Cost is
--- O(text) once per chat turn — irrelevant next to the model forward
--- pass.

local ffi  = require "ffi"
local json = require "ion7.vendor.json"
local tool_stream = require "ion7.llm.chat.tool_stream"

local find = string.find
local sub  = string.sub

local M = {}

local CONTENT_BUF_SIZE  = 65536
local THINKING_BUF_SIZE = 16384
local TOOLS_BUF_SIZE    = 32768

-- ════════════════════════════════════════════════════════════════════════
-- Bridge-backed parse
-- ════════════════════════════════════════════════════════════════════════

local function bridge_parse(vocab, raw_text, enable_thinking)
    if vocab._tmpls == nil then return nil end

    local bridge = require "ion7.core.ffi.bridge"

    local content_buf  = ffi.new("char[?]", CONTENT_BUF_SIZE)
    local thinking_buf = ffi.new("char[?]", THINKING_BUF_SIZE)
    local tools_buf    = ffi.new("char[?]", TOOLS_BUF_SIZE)
    local has_tools    = ffi.new("int[1]", 0)

    local et
    if     enable_thinking == nil   then et = -1
    elseif enable_thinking == true  then et = 1
    else                                  et = 0
    end

    local rc = bridge.ion7_chat_parse(
        vocab._tmpls,
        raw_text, et,
        content_buf,  CONTENT_BUF_SIZE,
        thinking_buf, THINKING_BUF_SIZE,
        tools_buf,    TOOLS_BUF_SIZE,
        has_tools)

    if rc < 0 then return nil end

    local content      = ffi.string(content_buf)
    local thinking_str = ffi.string(thinking_buf)
    local tools_str    = ffi.string(tools_buf)

    local tool_calls = {}
    if has_tools[0] == 1 and tools_str ~= "" then
        local ok, decoded = pcall(json.decode, tools_str)
        if ok and type(decoded) == "table" then
            tool_calls = decoded
        end
    end

    return {
        content    = content,
        thinking   = thinking_str ~= "" and thinking_str or nil,
        tool_calls = tool_calls,
        has_tools  = has_tools[0] == 1,
    }
end

-- ════════════════════════════════════════════════════════════════════════
-- Lua fallback : marker-based scan
-- ════════════════════════════════════════════════════════════════════════
--
-- Re-uses the format presets from `chat.tool_stream` so the streaming
-- and post-gen parsers stay in lockstep. Picks the first format whose
-- open marker appears in the text. Splits content (everything before
-- the first marker) from the tool-call envelope, then runs the
-- envelope through `tool_stream` to extract `{ id, name, arguments }`.

local function fallback_scan(raw_text, format_hint)
    if format_hint == "auto" or format_hint == nil then
        format_hint = nil
    end

    -- Pick which format(s) to probe.
    local formats
    if format_hint then
        local f = tool_stream.formats[format_hint]
        if not f then return nil end
        formats = { f }
    else
        formats = {
            tool_stream.formats.openai,
            tool_stream.formats.qwen,
            tool_stream.formats.mistral,
        }
    end

    -- Earliest open marker wins.
    local best_idx, best_fmt = nil, nil
    for i = 1, #formats do
        local f = formats[i]
        local idx = find(raw_text, f.open, 1, true)
        if idx and (not best_idx or idx < best_idx) then
            best_idx, best_fmt = idx, f
        end
    end
    if not best_idx then return nil end

    local content = sub(raw_text, 1, best_idx - 1)
    local rest    = sub(raw_text, best_idx)

    -- Run `rest` through tool_stream to extract every call. The SM
    -- yields `tool_call_done` chunks ; collect them into the result.
    local sm = tool_stream.new({ format = format_hint or "auto" })
    local tool_calls = {}
    -- One feed of the entire envelope is enough — the SM is incremental
    -- but accepts a complete buffer in a single call.
    local events = sm:feed(rest)
    for i = 1, #events do
        local e = events[i]
        if e.kind == "tool_call_done" then
            tool_calls[#tool_calls + 1] = e.call
        end
    end
    -- Anything the SM is still holding (end of stream without a close
    -- marker) gets force-closed via the done_calls list.
    local done = sm:done_calls()
    if #done > #tool_calls then
        tool_calls = done
    end

    -- Trim trailing whitespace from the content half.
    content = content:gsub("%s+$", "")

    return {
        content    = content,
        thinking   = nil,
        tool_calls = tool_calls,
        has_tools  = #tool_calls > 0,
    }
end

-- ════════════════════════════════════════════════════════════════════════
-- Public entry
-- ════════════════════════════════════════════════════════════════════════

--- Demultiplex an assistant turn into the three logical channels.
---
--- @param  vocab     ion7.core.Vocab
--- @param  raw_text  string
--- @param  opts      table?
---   `enable_thinking` (boolean?)  Forwarded to the bridge.
---   `tool_format`     (string?)   `"auto" | "openai" | "qwen" | "mistral"`.
---                                  Influences the Lua fallback when the
---                                  bridge auto-detect misses the format.
--- @return table
---   `content`     (string)
---   `thinking`    (string?)
---   `tool_calls`  (table[])
---   `has_tools`   (boolean)
function M.split(vocab, raw_text, opts)
    opts = opts or {}

    local primary = bridge_parse(vocab, raw_text, opts.enable_thinking)

    -- Bridge succeeded AND found tool calls : trust it.
    if primary and primary.has_tools then return primary end

    -- Try the marker-based fallback. If it finds tool calls AND the
    -- bridge did NOT, the fallback wins. If neither finds calls, we
    -- return the bridge result (or a plain wrapper around the raw text
    -- when the bridge is unavailable).
    local fb = fallback_scan(raw_text, opts.tool_format)

    if fb and fb.has_tools then
        -- Promote thinking from the bridge result if we have it.
        if primary and primary.thinking then
            fb.thinking = primary.thinking
        end
        return fb
    end

    if primary then return primary end

    return {
        content    = raw_text,
        thinking   = nil,
        tool_calls = {},
        has_tools  = false,
    }
end

return M
