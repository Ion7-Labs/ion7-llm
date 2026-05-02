--- @module ion7.llm.chat.stream
--- @author  ion7 / Ion7 Project Contributors
---
--- Typed-chunk iterator that wraps the engine's per-token yields.
---
--- The engine's hot loop produces a sequence of `(piece, role)` pairs
--- per token, where `role` is one of `content` / `thinking`. This
--- module turns those pairs into a richer chunk shape the consumer
--- can `for ... in` over without a stop-string disambiguation step :
---
---   { kind = "content",         text = "Hello" }
---   { kind = "thinking",        text = "Let me think…" }
---   { kind = "tool_call_delta", call_id = "0", name = "search",
---     args_partial = '{"q":"…' }
---   { kind = "tool_call_done",  call_id = "0", call = { … } }
---   { kind = "stop",            reason = "stop" | "length" | "stop_string" }
---
--- The wrapping is a thin generator function — the engine creates the
--- coroutine, this module wraps it so `for chunk in llm.stream(...)`
--- emits typed values instead of raw strings.
---
--- A backwards-compatible flag (`legacy_strings = true`) is NOT
--- provided ; the v2 stream is typed by design and the engine has a
--- separate `:chat()` entry point for callers that just want the
--- final assembled string.

local M = {}

--- Build a typed-chunk constructor of the right shape.
--- @param  text string
--- @return table
function M.content(text)
    return { kind = "content", text = text }
end

--- @param  text string
--- @return table
function M.thinking(text)
    return { kind = "thinking", text = text }
end

--- @param  call_id      string
--- @param  name         string?
--- @param  args_partial string
--- @return table
function M.tool_call_delta(call_id, name, args_partial)
    return {
        kind         = "tool_call_delta",
        call_id      = call_id,
        name         = name,
        args_partial = args_partial,
    }
end

--- @param  call_id string
--- @param  call    table  `{ id, name, arguments }`
--- @return table
function M.tool_call_done(call_id, call)
    return { kind = "tool_call_done", call_id = call_id, call = call }
end

--- @param  reason string  `"stop" | "length" | "stop_string" | "tool_use" | "error"`
--- @return table
function M.stop(reason)
    return { kind = "stop", reason = reason }
end

--- Materialise an entire chunk stream into a `Response`-shaped table.
--- The caller passes the iterator returned by `engine:stream` and
--- receives a synthesised `{ content, thinking, tool_calls, ... }`
--- view, identical in shape to what `engine:chat` would return.
---
--- Useful when a consumer wired itself to the typed stream and
--- still needs the post-generation summary (logging, persistence).
---
--- @param  iter function  Coroutine iterator yielding chunks.
--- @return table          `{ content, thinking, tool_calls, stop_reason }`
function M.collect(iter)
    local content_parts  = {}
    local thinking_parts = {}
    local tool_calls     = {}
    local stop_reason    = "stop"

    for chunk in iter do
        local k = chunk.kind
        if     k == "content"        then content_parts [#content_parts  + 1] = chunk.text
        elseif k == "thinking"       then thinking_parts[#thinking_parts + 1] = chunk.text
        elseif k == "tool_call_done" then tool_calls    [#tool_calls     + 1] = chunk.call
        elseif k == "stop"           then stop_reason   = chunk.reason
        end
    end

    return {
        content     = table.concat(content_parts),
        thinking    = #thinking_parts > 0 and table.concat(thinking_parts) or nil,
        tool_calls  = tool_calls,
        stop_reason = stop_reason,
    }
end

return M
