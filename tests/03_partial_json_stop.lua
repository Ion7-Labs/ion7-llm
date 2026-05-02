#!/usr/bin/env luajit
--- @module tests.03_partial_json_stop
--- @author  ion7 / Ion7 Project Contributors
---
--- Pure-Lua coverage for the two streaming building blocks that the
--- engine and pool wire into their hot loop :
---
---   - `util.partial_json` accumulates JSON fragments (one per chunk)
---     and reports whether the running buffer is balanced. The streaming
---     tool-call protocol depends on this : the engine ships partial
---     arguments to the consumer until the brace counter closes, then
---     reports a finished call.
---
---   - `Stop` is the multi-token stop-string detector. Two ways for it
---     to misbehave : (a) miss a stop string that gets split across
---     multiple yielded tokens, (b) match a prefix of a longer stop and
---     truncate too early. The tests below cover both, plus the runtime
---     `:add` API and the rolling-buffer overflow guard.
---
--- Both modules are pure Lua — no model needed — but a regression in
--- either silently corrupts every higher-level chat response, so a
--- thorough pure-Lua suite here pays off in stability later.

local T = require "tests.framework"
require "tests.helpers"

local PartialJson = require "ion7.llm.util.partial_json"
local Stop        = require "ion7.llm.stop"

-- ════════════════════════════════════════════════════════════════════════
-- partial_json — the streaming JSON accumulator
-- ════════════════════════════════════════════════════════════════════════

T.suite("partial_json — empty buffer + initial state")

T.test("a fresh buffer is incomplete and parses to nil/incomplete", function()
    local b = PartialJson.new()
    T.eq(b:complete(), false)
    T.eq(b:string(),   "")
    local v, err = b:value()
    T.eq(v,   nil)
    T.eq(err, "incomplete")
end)

T.test("appending an empty / nil fragment is a safe no-op", function()
    local b = PartialJson.new()
    b:append(nil)
    b:append("")
    T.eq(b:string(),   "")
    T.eq(b:complete(), false)
end)

T.suite("partial_json — incremental object assembly")

T.test("balanced object becomes complete after the closing brace", function()
    local b = PartialJson.new()
    b:append('{"city')
    T.eq(b:complete(), false, "buffer is open mid-key")
    b:append('": "Paris"')
    T.eq(b:complete(), false, "buffer still open")
    b:append("}")
    T.eq(b:complete(), true, "balanced object should be complete")
    local v = b:value()
    T.eq(v.city, "Paris")
end)

T.test("nested arrays + objects close together at the right depth", function()
    local b = PartialJson.new()
    b:append('{"a": [1, 2, ')
    T.eq(b:complete(), false)
    b:append('{"b": "c"}], "d')
    T.eq(b:complete(), false)
    b:append('": null}')
    T.eq(b:complete(), true)
    local v = b:value()
    T.eq(v.a[3].b, "c")
    T.eq(v.d,      nil) -- JSON null may decode to nil — accept either
end)

T.test("string contents containing braces don't fool the balance counter", function()
    local b = PartialJson.new()
    -- A real-world tool-call where the model encodes a JSON-looking
    -- value inside a string. The detector must NOT close on the inner
    -- `}` since it lives inside the string literal.
    b:append('{"raw": "{\\"x\\": 1}"}')
    T.eq(b:complete(), true)
    local v = b:value()
    T.eq(v.raw, '{"x": 1}')
end)

T.test("escaped quotes inside strings do not flip in_str", function()
    local b = PartialJson.new()
    b:append('{"msg": "He said \\"hi\\" then left"}')
    T.eq(b:complete(), true)
    local v = b:value()
    T.eq(v.msg, 'He said "hi" then left')
end)

T.suite("partial_json — primitive values + reset()")

T.test("a top-level boolean primitive parses once flushed", function()
    local b = PartialJson.new()
    b:append("true")
    T.eq(b:complete(), true)
    T.eq(b:value(),    true)
end)

T.test("a top-level number primitive parses once flushed", function()
    local b = PartialJson.new()
    b:append("42")
    T.eq(b:complete(), true)
    T.eq(b:value(),    42)
end)

T.test("reset() empties the buffer and clears the depth counter", function()
    local b = PartialJson.new()
    b:append('{"a": 1, "b": [')
    T.eq(b:complete(), false, "buffer is mid-array")
    b:reset()
    T.eq(b:string(),   "")
    T.eq(b:complete(), false)
    -- A fresh value can be appended without leaking the previous depth.
    b:append('{"x": 1}')
    T.eq(b:complete(), true)
    T.eq(b:value().x,  1)
end)

T.test("value() returns (nil, err) on malformed JSON even when balanced", function()
    local b = PartialJson.new()
    -- Balanced braces but trailing garbage after the value.
    b:append('{"a": 1, ,}')
    T.eq(b:complete(), true, "brace count is balanced")
    local v, err = b:value()
    T.eq(v, nil)
    T.is_type(err, "string")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Stop — multi-token stop-string detector
-- ════════════════════════════════════════════════════════════════════════

T.suite("Stop — defaults + simple matches")

T.test("default constructor matches the chat-end markers we ship", function()
    local s = Stop.new()
    local list = s:list()
    -- The list is sorted longest-first ; we just verify a few major
    -- tokens are present without enforcing an exact order.
    local set = {}
    for _, x in ipairs(list) do set[x] = true end
    T.eq(set["<|im_end|>"],     true)
    T.eq(set["<|eot_id|>"],     true)
    T.eq(set["<end_of_turn>"],  true)
    T.eq(set["</s>"],           true)
end)

T.test("a single-piece stop string fires immediately", function()
    local s = Stop.new()
    local m = s:feed("hello <|im_end|>")
    T.eq(m, "<|im_end|>")
end)

T.test("a stop string split across two yields is still detected", function()
    local s = Stop.new()
    -- The model emits "<|", then "im_end|>" — the rolling buffer must
    -- glue them together to find the complete stop tag.
    T.eq(s:feed("<|"),       nil)
    T.eq(s:feed("im_end|>"), "<|im_end|>")
end)

T.test("text_before clips the assistant content at the match boundary", function()
    local s = Stop.new()
    s:feed("hello world")
    T.eq(s:feed("<|im_end|>"), "<|im_end|>")
    T.eq(s:text_before("<|im_end|>"), "hello world")
end)

T.test("reset clears the buffer between generations", function()
    local s = Stop.new()
    s:feed("<|im_")
    s:reset()
    T.eq(s:feed("end|>"), nil,
        "reset must drop the partial tag so an unrelated 'end|>' does not match")
end)

T.suite("Stop — custom strings + extra list + add()")

T.test("opts.strings replaces the default list entirely", function()
    local s = Stop.new({ strings = { "[STOP]" } })
    T.eq(s:feed("hi <|im_end|>"), nil,
        "default markers must NOT match when an explicit list is supplied")
    T.eq(s:feed(" [STOP]"), "[STOP]")
end)

T.test("opts.extra adds to the default list", function()
    local s = Stop.new({ extra = { "###END###" } })
    -- Default markers still apply.
    T.eq(s:feed(" </s>"), "</s>")
    s:reset()
    -- Custom marker also fires.
    T.eq(s:feed("hello ###END###"), "###END###")
end)

T.test("add() inserts at runtime and re-sorts longest-first", function()
    local s = Stop.new({ strings = { "END" } })
    s:add("ENDPOINT")
    -- "ENDPOINT" must match before its prefix "END".
    T.eq(s:feed("hit ENDPOINT"), "ENDPOINT",
        "longer stop string must take precedence over its prefix")
end)

T.test("the rolling buffer is bounded — old text cannot match later", function()
    local s = Stop.new({ strings = { "MATCHME" }, buf_size = 16 })
    -- Push the marker beyond the rolling window with arbitrary chars.
    s:feed("MATCHME")
    s:reset()
    s:feed(string.rep("x", 32))
    -- The next feed cannot retroactively find the previous "MATCHME".
    T.eq(s:feed("y"), nil)
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
