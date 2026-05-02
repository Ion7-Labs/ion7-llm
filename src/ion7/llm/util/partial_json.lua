--- @module ion7.llm.util.partial_json
--- @author  ion7 / Ion7 Project Contributors
---
--- Streaming JSON accumulator for partial tool-call arguments.
---
--- During a streaming response the model emits a tool call's arguments
--- as a sequence of partial JSON fragments — `{"city`, `":`, ` "Paris"`,
--- `}` — and only the concatenation of those fragments parses cleanly.
--- This buffer accepts every fragment via `:append`, exposes the raw
--- accumulated string via `:string`, and emits a parsed Lua table once
--- the buffer is balanced (every `{` / `[` matched).
---
--- Balance detection is a one-pass scan that respects strings + escape
--- sequences, so we don't choke on a `{ "a": "}" }` where the brace
--- inside the string would otherwise close the object early.
---
--- The final parse uses `ion7.vendor.json` — the same encoder/decoder
--- the rest of ion7-llm shares with ion7-core. `:string()` and
--- `:complete()` stay pure-Lua so a caller that already has the bytes
--- can read them out without a parse.

local json = require "ion7.vendor.json"

local M = {}
local Buffer = {}
Buffer.__index = Buffer

--- New buffer.
--- @return ion7.llm.util.PartialJsonBuffer
function M.new()
    return setmetatable({
        _buf      = "",
        _depth    = 0,         -- net `{` minus `}` plus `[` minus `]`
        _in_str   = false,
        _escape   = false,
        _started  = false,     -- true after the first non-whitespace char
    }, Buffer)
end

--- Feed a fragment. The internal balance counter is updated in O(n).
--- @param  s string
--- @return ion7.llm.util.PartialJsonBuffer self (chainable)
function Buffer:append(s)
    if not s or s == "" then return self end
    self._buf = self._buf .. s
    for i = 1, #s do
        local c = s:sub(i, i)
        if self._escape then
            -- Skip the escaped character — it can't change balance.
            self._escape = false
        elseif self._in_str then
            if c == "\\" then
                self._escape = true
            elseif c == '"' then
                self._in_str = false
            end
        else
            if c == '"' then
                self._in_str = true
                self._started = true
            elseif c == "{" or c == "[" then
                self._depth = self._depth + 1
                self._started = true
            elseif c == "}" or c == "]" then
                self._depth = self._depth - 1
            elseif not self._started and not c:match("%s") then
                -- Non-whitespace primitive (number, true, false, null) ;
                -- the buffer is "started" and ends as soon as whitespace
                -- or EOF arrives. We don't treat these specially — the
                -- final parse will validate them.
                self._started = true
            end
        end
    end
    return self
end

--- Raw accumulated string. Useful when the caller wants to parse with
--- their own JSON library, or to stash the buffer for replay.
--- @return string
function Buffer:string()
    return self._buf
end

--- True when the buffer contains a balanced JSON value (object or
--- array) AND we are not currently inside a string. For primitive
--- top-level values (number / bool / null) the buffer is considered
--- complete as soon as `_started` is true and not inside a string.
--- @return boolean
function Buffer:complete()
    if self._in_str then return false end
    if not self._started then return false end
    return self._depth == 0
end

--- Clear the buffer.
function Buffer:reset()
    self._buf, self._depth = "", 0
    self._in_str, self._escape, self._started = false, false, false
end

--- Parse the buffer to a Lua value. Returns `(value, nil)` on
--- success or `(nil, err)` on parse failure. Calling `:value()` on
--- an incomplete buffer returns `(nil, "incomplete")`.
--- @return any|nil
--- @return string?
function Buffer:value()
    if not self:complete() then return nil, "incomplete" end
    local ok, decoded = pcall(json.decode, self._buf)
    if not ok then return nil, tostring(decoded) end
    return decoded
end

return M
