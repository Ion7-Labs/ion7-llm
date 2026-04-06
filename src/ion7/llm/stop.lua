--- @module ion7.llm.stop
--- SPDX-License-Identifier: MIT
--- Multi-token stop string detection via rolling buffer.
---
--- Stop strings often span multiple tokens ("<|im_end|>" -> ["<|", "im", "_end", "|>"]).
--- Each piece is appended to a bounded buffer; matches are checked after every token.
---
--- @author Ion7-Labs
--- @version 0.1.0

local Stop = {}
Stop.__index = Stop

local COMMON_STOPS = {
    "<|im_end|>",                -- ChatML
    "<|eot_id|>", "<|end_of_turn|>",  -- Llama 3
    "<end_of_turn>",             -- Gemma
    "</s>", "[INST]", "[/INST]", -- generic
    "<｜end▁of▁sentence｜>",    -- DeepSeek
}

--- @param  opts  table?
---   opts.strings  table?   Custom stop strings (replaces defaults).
---   opts.extra    table?   Additional strings merged with defaults.
---   opts.buf_size number?  Rolling buffer size in chars (default: 256).
--- @return Stop
function Stop.new(opts)
    opts = opts or {}

    local base = opts.strings or COMMON_STOPS
    local list = {}
    for _, s in ipairs(base) do list[#list + 1] = s end
    if opts.extra then
        for _, s in ipairs(opts.extra) do list[#list + 1] = s end
    end

    -- Deduplicate
    local seen, deduped = {}, {}
    for _, s in ipairs(list) do
        if not seen[s] then seen[s] = true; deduped[#deduped + 1] = s end
    end

    -- Longest first: longer stops match before their prefixes
    table.sort(deduped, function(a, b) return #a > #b end)

    return setmetatable({
        _list     = deduped,
        _buf      = "",
        _buf_size = opts.buf_size or 256,
    }, Stop)
end

--- Feed a decoded piece. Returns matched stop string + buffer offset, or nil.
--- @param  piece  string
--- @return string?, number?
function Stop:feed(piece)
    self._buf = self._buf .. piece
    if #self._buf > self._buf_size then
        self._buf = self._buf:sub(-self._buf_size)
    end
    for _, stop in ipairs(self._list) do
        local idx = self._buf:find(stop, 1, true)
        if idx then return stop, idx end
    end
    return nil
end

--- Text accumulated before the matched stop string.
--- Call after feed() returns a match.
--- @param  stop  string
--- @return string
function Stop:text_before(stop)
    local idx = self._buf:find(stop, 1, true)
    if not idx then return self._buf end
    return self._buf:sub(1, idx - 1)
end

--- Clear buffer (call at start of each generation).
function Stop:reset()
    self._buf = ""
end

--- Add a stop string at runtime.
--- @param  s  string
function Stop:add(s)
    if s and s ~= "" then
        self._list[#self._list + 1] = s
        table.sort(self._list, function(a, b) return #a > #b end)
    end
end

--- All registered stop strings (sorted longest-first).
--- @return table
function Stop:list()
    local out = {}
    for i, s in ipairs(self._list) do out[i] = s end
    return out
end

return Stop
