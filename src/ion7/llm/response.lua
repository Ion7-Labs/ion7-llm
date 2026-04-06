--- @module ion7.llm.response
--- SPDX-License-Identifier: MIT
--- Structured generation result.
---
--- @author Ion7-Labs
--- @version 0.1.0

local Response = {}
Response.__index = Response

--- @param  text        string
--- @param  tokens      table
--- @param  stop_reason string   "stop" | "length" | "stop_string" | "error"
--- @param  perf        table    { tok_per_s, n_eval, t_eval_ms, n_p_eval }
--- @param  think       string?  Raw <think>...</think> content, or nil.
--- @return Response
function Response.new(text, tokens, stop_reason, perf, think)
    return setmetatable({
        text        = text        or "",
        tokens      = tokens      or {},
        n_tokens    = tokens and #tokens or 0,
        stop_reason = stop_reason or "stop",
        perf        = perf        or {},
        _think      = think       or nil,
    }, Response)
end

function Response:__tostring() return self.text end

--- Raw think block content, or nil if none / think=false.
--- @return string?
function Response:think()
    return self._think
end

--- One-line summary: "[N tok | X tok/s | reason]"
--- @return string
function Response:summary()
    return string.format("[%d tok | %.1f tok/s | %s]",
        self.n_tokens, self.perf.tok_per_s or 0, self.stop_reason)
end

return Response
