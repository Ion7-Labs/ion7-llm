--- @module ion7.llm.stop
--- @author  ion7 / Ion7 Project Contributors
---
--- Multi-token stop-string detector.
---
--- Stop strings rarely arrive whole in a single token — `<|im_end|>`
--- gets split into `<|`, `im`, `_end`, `|>` by most BPE tokenisers.
--- Each yielded piece appends to a bounded rolling buffer ; after
--- every append, the buffer is scanned for any of the registered
--- stop strings.
---
--- The default list covers the chat-end markers of the major model
--- families — ChatML, Llama 3, Gemma, generic INST tags, DeepSeek.
--- Pass `opts.strings` to replace the default, or `opts.extra` to
--- merge additions on top.

-- Hot-path locals : :feed runs once per generated token.
local string_find = string.find
local string_sub  = string.sub

local Stop = {}
Stop.__index = Stop

local DEFAULT_STOPS = {
    "<|im_end|>",                       -- ChatML (Qwen, OpenChat, …)
    "<|eot_id|>", "<|end_of_turn|>",    -- Llama 3
    "<end_of_turn>",                    -- Gemma
    "</s>", "[INST]", "[/INST]",        -- generic
    "<|end_of_text|>",
    "<｜end▁of▁sentence｜>",          -- DeepSeek
}

--- @param  opts table?
---   `strings` (string[])  Replaces the default list.
---   `extra`   (string[])  Merged on top of the active list.
---   `buf_size`(integer)   Rolling buffer size in chars (default 256).
--- @return ion7.llm.Stop
function Stop.new(opts)
    opts = opts or {}

    local base = opts.strings or DEFAULT_STOPS
    local list = {}
    for _, s in ipairs(base) do list[#list + 1] = s end
    if opts.extra then
        for _, s in ipairs(opts.extra) do list[#list + 1] = s end
    end

    -- Deduplicate + sort longest-first so a longer stop matches before
    -- one of its prefixes.
    local seen, deduped = {}, {}
    for _, s in ipairs(list) do
        if not seen[s] then seen[s] = true ; deduped[#deduped + 1] = s end
    end
    table.sort(deduped, function(a, b) return #a > #b end)

    return setmetatable({
        _list     = deduped,
        _buf      = "",
        _buf_size = opts.buf_size or 256,
    }, Stop)
end

--- Append a piece, then probe the buffer for any registered stop
--- string. Returns the matched stop and its position when one fires,
--- nil otherwise.
---
--- @param  piece string
--- @return string? matched_stop
--- @return integer? position
function Stop:feed(piece)
    -- Concat then bound the rolling buffer in one go.
    local buf = self._buf .. piece
    local sz  = self._buf_size
    if #buf > sz then
        buf = string_sub(buf, -sz)
    end
    self._buf = buf

    -- Linear scan over the longest-first sorted list. O(N · |buf|)
    -- per token where N is small (typically 8) and |buf| is bounded
    -- to `buf_size` (256 by default).
    local list = self._list
    for i = 1, #list do
        local stop = list[i]
        local idx  = string_find(buf, stop, 1, true)
        if idx then return stop, idx end
    end
    return nil
end

--- Text accumulated before the matched stop string. Call right
--- after `:feed` returns a hit when you need the assistant's reply
--- minus the stop marker.
--- @param  stop string
--- @return string
function Stop:text_before(stop)
    local idx = self._buf:find(stop, 1, true)
    if not idx then return self._buf end
    return self._buf:sub(1, idx - 1)
end

--- Wipe the rolling buffer. Call between generations.
function Stop:reset()
    self._buf = ""
end

--- Add an additional stop string at runtime. The list is re-sorted
--- so the new entry slots into the right longest-first position.
--- @param  s string
function Stop:add(s)
    if s and s ~= "" then
        self._list[#self._list + 1] = s
        table.sort(self._list, function(a, b) return #a > #b end)
    end
end

--- Snapshot of the active stop-string list.
--- @return string[]
function Stop:list()
    local out = {}
    for i, s in ipairs(self._list) do out[i] = s end
    return out
end

return Stop
