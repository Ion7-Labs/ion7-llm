--- @module ion7.llm.response
--- @author  ion7 / Ion7 Project Contributors
---
--- Generation result. Three logical channels :
---
---   - `content`     plain assistant text, post-thinking, post-tool-call
---   - `thinking`    `<think>...</think>` body (raw), or nil
---   - `tool_calls`  array of `{ id, name, arguments }`, or empty
---
--- Plus diagnostic data : token ids, stop reason, perf snapshot.
---
--- The class is intentionally inert — no methods that talk to the
--- model, no methods that mutate the parent session. A `Response` is
--- a value object you can pass around, log, or persist.
---
---   local r = engine:chat(session)
---   if r:has_tools() then
---       for _, call in ipairs(r.tool_calls) do
---           dispatch[call.name](call.arguments)
---       end
---   else
---       print(r.content)
---   end

local Response = {}
Response.__index = Response

--- @class ion7.llm.Response
--- @field content     string                Plain assistant text (post-strip).
--- @field thinking    string?               Reasoning block, preserved on tool turns.
--- @field tool_calls  table[]               `[{ id, name, arguments }, ...]`
--- @field tokens      integer[]             Sampled token ids.
--- @field n_tokens    integer
--- @field stop_reason string                `"stop" | "length" | "stop_string" | "tool_use" | "error"`
--- @field perf        table                 `{ tok_per_s, n_eval, t_eval_ms, n_p_eval }`

--- Build a Response.
---
--- @param  fields table
---   `content`     (string, default `""`)  Assistant reply.
---   `thinking`    (string?)               Reasoning block.
---   `tool_calls`  (table[]?)              Parsed tool calls.
---   `tokens`      (integer[]?)            Sampled token ids.
---   `stop_reason` (string?, default `"stop"`)
---   `perf`        (table?)                Perf snapshot.
--- @return ion7.llm.Response
function Response.new(fields)
    fields = fields or {}
    local tokens = fields.tokens or {}
    local tools  = fields.tool_calls or {}
    return setmetatable({
        content     = fields.content     or "",
        thinking    = fields.thinking    or nil,
        tool_calls  = tools,
        tokens      = tokens,
        n_tokens    = #tokens,
        stop_reason = fields.stop_reason or "stop",
        perf        = fields.perf        or {},
    }, Response)
end

--- True when the response carries at least one tool call.
--- @return boolean
function Response:has_tools()
    return self.tool_calls and #self.tool_calls > 0 or false
end

--- One-line summary suited for logging.
---   "[42 tok | 38.2 tok/s | stop]"
--- @return string
function Response:summary()
    return string.format("[%d tok | %.1f tok/s | %s]",
        self.n_tokens, self.perf.tok_per_s or 0, self.stop_reason)
end

--- `tostring(response)` returns the assistant content.
function Response:__tostring()
    return self.content
end

return Response
