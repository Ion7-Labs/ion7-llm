#!/usr/bin/env luajit
--- Pure Lua tests - no model, no libllama.so required.
--- Tests all logic that doesn't depend on FFI or a GPU.
---
--- Run: luajit tests/test_pure.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local T = require "tests.framework"

-- ── Suite 1: Response ─────────────────────────────────────────────────────────

T.suite("Response")

local Response = require "ion7.llm.response"

T.test("Response.new() basic fields", function()
    local r = Response.new("hello", {10, 20, 30}, "stop", { tok_per_s = 42 })
    T.eq(r.text,        "hello")
    T.eq(r.n_tokens,    3)
    T.eq(r.stop_reason, "stop")
    T.eq(r.perf.tok_per_s, 42)
end)

T.test("Response.new() empty defaults", function()
    local r = Response.new()
    T.eq(r.text,        "")
    T.eq(r.n_tokens,    0)
    T.eq(r.stop_reason, "stop")
    T.is_type(r.perf, "table")
    T.is_type(r.tokens, "table")
end)

T.test("Response.new() with think text", function()
    local r = Response.new("answer", {1, 2}, "stop", {}, "thinking stuff")
    T.eq(r:think(), "thinking stuff")
end)

T.test("Response:think() returns nil when no think", function()
    local r = Response.new("answer", {1}, "stop", {})
    T.eq(r:think(), nil)
end)

T.test("Response:summary() format", function()
    local r = Response.new("hi", {1, 2}, "length", { tok_per_s = 12.5 })
    local s = r:summary()
    T.ok(s:find("2 tok"),    "token count in summary")
    T.ok(s:find("12.5"),     "tok/s in summary")
    T.ok(s:find("length"),   "stop_reason in summary")
end)

T.test("Response:summary() with zero perf", function()
    local r = Response.new("x", {1}, "stop", {})
    local s = r:summary()
    T.ok(s:find("0.0"), "zero tok/s")
end)

T.test("tostring(Response) returns text", function()
    local r = Response.new("result text", {1}, "stop", {})
    T.eq(tostring(r), "result text")
end)

T.test("stop_reason 'stop_string'", function()
    local r = Response.new("partial", {1, 2, 3}, "stop_string", {})
    T.eq(r.stop_reason, "stop_string")
end)

T.test("tokens array stored correctly", function()
    local r = Response.new("x", {100, 200, 300, 400}, "stop", {})
    T.eq(r.n_tokens, 4)
    T.eq(r.tokens[1], 100)
    T.eq(r.tokens[4], 400)
end)

-- ── Suite 2: Stop ─────────────────────────────────────────────────────────────

T.suite("Stop")

local Stop = require "ion7.llm.stop"

T.test("Stop.new() returns a Stop object", function()
    local s = Stop.new()
    T.ok(s ~= nil)
    T.ok(type(s.feed) == "function")
    T.ok(type(s.reset) == "function")
    T.ok(type(s.add) == "function")
    T.ok(type(s.list) == "function")
end)

T.test("Stop.new() with extra stop strings", function()
    local s = Stop.new({ extra = { "STOP_NOW", "HALT" } })
    local lst = s:list()
    local found_stop = false
    local found_halt = false
    for _, v in ipairs(lst) do
        if v == "STOP_NOW" then found_stop = true end
        if v == "HALT" then found_halt = true end
    end
    T.ok(found_stop, "STOP_NOW in list")
    T.ok(found_halt, "HALT in list")
end)

T.test("Stop.new() with custom strings replaces defaults", function()
    local s = Stop.new({ strings = { "ONLY_THIS" } })
    local lst = s:list()
    T.eq(#lst, 1)
    T.eq(lst[1], "ONLY_THIS")
end)

T.test("feed() returns nil when no match", function()
    local s = Stop.new({ strings = { "STOP" } })
    local match = s:feed("hello world")
    T.eq(match, nil)
end)

T.test("feed() detects stop string", function()
    local s = Stop.new({ strings = { "<|im_end|>" } })
    s:feed("hello ")
    s:feed("world")
    local match = s:feed("<|im_end|>")
    T.ok(match ~= nil, "should match")
    T.eq(match, "<|im_end|>")
end)

T.test("feed() detects stop across multiple pieces", function()
    local s = Stop.new({ strings = { "<|eot_id|>" } })
    s:feed("<|")
    s:feed("eot")
    s:feed("_id")
    local match = s:feed("|>")
    T.ok(match ~= nil, "cross-token stop detected")
end)

T.test("feed() detects common ChatML stop", function()
    local s = Stop.new()
    s:feed("some text")
    local match = s:feed("<|im_end|>")
    T.ok(match ~= nil)
end)

T.test("text_before() returns text before stop", function()
    local s = Stop.new({ strings = { "END" } })
    s:feed("hello ")
    s:feed("worldEND")
    T.eq(s:text_before("END"), "hello world")
end)

T.test("text_before() with nothing before stop", function()
    local s = Stop.new({ strings = { "STOP" } })
    s:feed("STOP")
    T.eq(s:text_before("STOP"), "")
end)

T.test("reset() clears the buffer", function()
    local s = Stop.new({ strings = { "STOP" } })
    s:feed("STO")
    s:reset()
    -- After reset, "P" alone should not match "STOP"
    local match = s:feed("P")
    T.eq(match, nil, "buffer cleared after reset")
end)

T.test("add() adds a stop string at runtime", function()
    local s = Stop.new({ strings = {} })
    T.eq(s:feed("DYNAMIC_STOP"), nil, "not yet registered")
    s:add("DYNAMIC_STOP")
    s:reset()
    local match = s:feed("DYNAMIC_STOP")
    T.ok(match ~= nil, "now registered")
end)

T.test("list() returns a copy of stop strings", function()
    local s = Stop.new({ strings = { "A", "BB", "CCC" } })
    local lst = s:list()
    T.eq(#lst, 3)
    -- Sorted longest-first
    T.eq(lst[1], "CCC")
    T.eq(lst[2], "BB")
    T.eq(lst[3], "A")
end)

T.test("deduplicate on construction", function()
    local s = Stop.new({ strings = { "X", "X", "Y", "X" } })
    local lst = s:list()
    T.eq(#lst, 2)
end)

T.test("buffer bounded at buf_size", function()
    local s = Stop.new({ strings = { "NEEDLE" }, buf_size = 10 })
    -- Feed more than buf_size bytes - buffer trims to 10 chars
    s:feed("AAAAAAAAAAAAAAAA")  -- 16 chars, trimmed to last 10
    -- NEEDLE is not in the last 10 chars of that feed
    T.eq(s:feed("END"), nil)
end)

T.test("common stops include Llama-3 eot_id", function()
    local s = Stop.new()
    local lst = s:list()
    local found = false
    for _, v in ipairs(lst) do
        if v == "<|eot_id|>" then found = true; break end
    end
    T.ok(found, "<|eot_id|> in default stops")
end)

T.test("common stops include DeepSeek sentinel", function()
    local s = Stop.new()
    local lst = s:list()
    local found = false
    for _, v in ipairs(lst) do
        if v:find("end") and v:find("sentence") then found = true; break end
    end
    T.ok(found, "DeepSeek stop in defaults")
end)

T.test("Stop.new() with buf_size override", function()
    local s = Stop.new({ strings = { "X" }, buf_size = 512 })
    T.ok(s ~= nil)
end)

-- ── Suite 3: Session ──────────────────────────────────────────────────────────

T.suite("Session")

local Session = require "ion7.llm.session"

T.test("Session.new() default fields", function()
    local s = Session.new()
    T.is_type(s.id,        "number")
    T.eq(s.seq_id,         nil)   -- nil = assigned by ContextManager on first use
    T.eq(s.system,         nil)
    T.is_type(s.messages,  "table")
    T.eq(#s.messages,      0)
    T.eq(s.n_past,         0)
    T.eq(s._snapshot,      nil)
    T.eq(s._dirty,         false)
end)

T.test("Session.new() with system prompt", function()
    local s = Session.new({ system = "You are helpful." })
    T.eq(s.system, "You are helpful.")
end)

T.test("Session.new() with seq_id", function()
    local s = Session.new({ seq_id = 3 })
    T.eq(s.seq_id, 3)
end)

T.test("Session IDs are unique and increasing", function()
    local a = Session.new()
    local b = Session.new()
    local c = Session.new()
    T.ok(a.id ~= b.id)
    T.ok(b.id ~= c.id)
    T.ok(b.id > a.id)
    T.ok(c.id > b.id)
end)

T.test("add() appends messages and sets dirty", function()
    local s = Session.new()
    s:add("user", "Hello")
    T.eq(#s.messages, 1)
    T.eq(s.messages[1].role,    "user")
    T.eq(s.messages[1].content, "Hello")
    T.eq(s._dirty, true)
end)

T.test("add() returns self for chaining", function()
    local s = Session.new()
    local ret = s:add("user", "a")
    T.eq(ret, s)
    s:add("assistant", "b"):add("user", "c")
    T.eq(#s.messages, 3)
end)

T.test("add() validates role type", function()
    local s = Session.new()
    T.err(function() s:add(123, "content") end, "role must be a string")
end)

T.test("add() validates content type", function()
    local s = Session.new()
    T.err(function() s:add("user", 456) end, "content must be a string")
end)

T.test("all_messages() with no system prompt", function()
    local s = Session.new()
    s:add("user", "Hi")
    s:add("assistant", "Hello")
    local msgs = s:all_messages()
    T.eq(#msgs, 2)
    T.eq(msgs[1].role, "user")
    T.eq(msgs[2].role, "assistant")
end)

T.test("all_messages() prepends system when set", function()
    local s = Session.new({ system = "Be concise." })
    s:add("user", "Hello")
    local msgs = s:all_messages()
    T.eq(#msgs, 2)
    T.eq(msgs[1].role,    "system")
    T.eq(msgs[1].content, "Be concise.")
    T.eq(msgs[2].role,    "user")
end)

T.test("pending_messages() returns messages array", function()
    local s = Session.new()
    s:add("user", "test")
    local pending = s:pending_messages()
    T.eq(#pending, 1)
end)

T.test("has_snapshot() false by default", function()
    local s = Session.new()
    T.eq(s:has_snapshot(), false)
end)

T.test("_save_snapshot() / has_snapshot() / snapshot()", function()
    local s = Session.new()
    s:_save_snapshot("fake_blob_data", 42)
    T.eq(s:has_snapshot(), true)
    T.eq(s:snapshot(), "fake_blob_data")
    T.eq(s.n_past, 42)
    T.eq(s._dirty, false)
end)

T.test("_save_snapshot() clears dirty flag", function()
    local s = Session.new()
    s:add("user", "hi")
    T.eq(s._dirty, true)
    s:_save_snapshot("blob", 10)
    T.eq(s._dirty, false)
end)

T.test("reset() clears everything", function()
    local s = Session.new()
    s:add("user", "hello")
    s:_save_snapshot("blob", 5)
    s:reset()
    T.eq(#s.messages,  0)
    T.eq(s.n_past,     0)
    T.eq(s._snapshot,  nil)
    T.eq(s._dirty,     false)
    T.eq(s:has_snapshot(), false)
end)

T.test("reset() returns self", function()
    local s = Session.new()
    T.eq(s:reset(), s)
end)

T.test("fork() copies messages", function()
    local s = Session.new({ system = "sys" })
    s:add("user", "a")
    s:add("assistant", "b")
    local child = s:fork()
    T.eq(#child.messages,  2)
    T.eq(child.system,     "sys")
    T.eq(child.messages[1].content, "a")
    T.eq(child.messages[2].content, "b")
end)

T.test("fork() child is independent", function()
    local s = Session.new()
    s:add("user", "original")
    local child = s:fork()
    child:add("user", "forked only")
    T.eq(#s.messages,     1)
    T.eq(#child.messages, 2)
end)

T.test("fork() child has nil seq_id", function()
    local s = Session.new({ seq_id = 2 })
    local child = s:fork()
    T.eq(child.seq_id, nil)
end)

T.test("fork() copies n_past and snapshot", function()
    local s = Session.new()
    s:_save_snapshot("snap", 100)
    local child = s:fork()
    T.eq(child.n_past,    100)
    T.eq(child._snapshot, "snap")
end)

T.test("serialize() returns a plain table", function()
    local s = Session.new({ system = "sys" })
    s:add("user", "hello")
    local t = s:serialize()
    T.is_type(t, "table")
    T.eq(t.system, "sys")
    T.eq(#t.messages, 1)
    T.eq(t.messages[1].content, "hello")
end)

T.test("deserialize() restores session", function()
    local s = Session.new({ system = "sys" })
    s:add("user", "hello")
    s:add("assistant", "hi")
    local t = s:serialize()
    local s2 = Session.deserialize(t)
    T.eq(s2.system, "sys")
    T.eq(#s2.messages, 2)
    T.eq(s2.messages[1].content, "hello")
    T.eq(s2.n_past, 0)
    T.eq(s2._dirty, true)
end)

T.test("deserialize() assigns new id when none in table", function()
    local t = { messages = {}, system = nil }
    local s = Session.deserialize(t)
    T.is_type(s.id, "number")
    T.ok(s.id > 0)
end)

T.test("round-trip serialize/deserialize preserves messages", function()
    local original = Session.new({ system = "You are Joi." })
    original:add("user", "What year is it?")
    original:add("assistant", "2026.")
    original:add("user", "What city?")
    local t = original:serialize()
    local restored = Session.deserialize(t)
    T.eq(restored.system, "You are Joi.")
    T.eq(#restored.messages, 3)
    T.eq(restored.messages[3].content, "What city?")
end)

-- ── Suite 4: sampler_profiles structure ───────────────────────────────────────

T.suite("sampler_profiles - module structure")

local profiles = require "ion7.llm.sampler_profiles"

T.test("module is a table", function()
    T.is_type(profiles, "table")
end)

T.test("balanced profile exists as function", function()
    T.is_type(profiles.balanced, "function")
end)

T.test("precise profile exists as function", function()
    T.is_type(profiles.precise, "function")
end)

T.test("creative profile exists as function", function()
    T.is_type(profiles.creative, "function")
end)

T.test("code profile exists as function", function()
    T.is_type(profiles.code, "function")
end)

T.test("fast profile exists as function", function()
    T.is_type(profiles.fast, "function")
end)

T.test("thinking profile exists as function", function()
    T.is_type(profiles.thinking, "function")
end)

T.test("no unknown keys in profiles", function()
    local known = { balanced=1, precise=1, creative=1, code=1, fast=1, thinking=1, extended=1 }
    for k, _ in pairs(profiles) do
        T.ok(known[k] ~= nil, "unknown profile: " .. tostring(k))
    end
end)

-- ── Suite 5: Response - think field interaction ───────────────────────────────

T.suite("Response - think field")

T.test("think=nil and think() returns nil", function()
    local r = Response.new("out", {1, 2}, "stop", {}, nil)
    T.eq(r:think(), nil)
end)

T.test("think='' and think() returns empty string", function()
    local r = Response.new("out", {1}, "stop", {}, "")
    -- In Lua, "" is truthy: "" or nil = "", so _think stays as "".
    -- An empty think block is preserved (generator only produces non-empty think text).
    T.eq(r:think(), "")
end)

T.test("think text with content", function()
    local r = Response.new("final", {1, 2, 3}, "stop", {}, "I should think about this...")
    T.eq(r:think(), "I should think about this...")
end)

T.test("think does not appear in .text", function()
    -- text field should be stripped output, think field separate
    local r = Response.new("clean output", {1}, "stop", {}, "<think>internal</think>")
    T.ok(not r.text:find("<think>"), "text should not contain think tags")
    T.ok(r:think():find("internal"), "think() should contain the think content")
end)

-- ── Suite 6: Stop - edge cases ────────────────────────────────────────────────

T.suite("Stop - edge cases")

T.test("empty piece does not crash", function()
    local s = Stop.new()
    T.no_error(function() s:feed("") end)
end)

T.test("feed() with very long piece", function()
    local s = Stop.new({ strings = { "END" } })
    local long = string.rep("x", 1000) .. "END"
    local match = s:feed(long)
    T.ok(match ~= nil, "should detect END in long piece")
end)

T.test("add() ignores empty string", function()
    local s = Stop.new({ strings = {} })
    s:add("")
    T.eq(#s:list(), 0)
end)

T.test("add() ignores nil", function()
    local s = Stop.new({ strings = { "A" } })
    s:add(nil)
    T.eq(#s:list(), 1)
end)

T.test("multiple resets are idempotent", function()
    local s = Stop.new()
    s:feed("hello")
    s:reset()
    s:reset()
    s:reset()
    T.eq(s:feed("world"), nil)
end)

T.test("text_before() when stop not in buffer returns whole buffer", function()
    local s = Stop.new({ strings = { "X" } })
    s:feed("hello world")
    -- "X" not in buffer
    T.eq(s:text_before("X"), "hello world")
end)

-- ── Suite 7: Session - multi-message history ──────────────────────────────────

T.suite("Session - multi-message history")

T.test("multi-turn conversation flow", function()
    local s = Session.new({ system = "Be helpful." })
    s:add("user",      "What is 2+2?")
    s:add("assistant", "4.")
    s:add("user",      "And 3+3?")

    local msgs = s:all_messages()
    T.eq(#msgs, 4)  -- system + 3 messages
    T.eq(msgs[1].role, "system")
    T.eq(msgs[2].role, "user")
    T.eq(msgs[3].role, "assistant")
    T.eq(msgs[4].role, "user")
    T.eq(msgs[4].content, "And 3+3?")
end)

T.test("add() many messages", function()
    local s = Session.new()
    for i = 1, 100 do
        s:add("user", "msg " .. i)
    end
    T.eq(#s.messages, 100)
    T.eq(s.messages[100].content, "msg 100")
end)

T.test("dirty flag resets only after _save_snapshot", function()
    local s = Session.new()
    T.eq(s._dirty, false)
    s:add("user", "a")
    T.eq(s._dirty, true)
    s:_save_snapshot("blob", 5)
    T.eq(s._dirty, false)
    s:add("user", "b")
    T.eq(s._dirty, true)
end)

T.test("fork creates deep copy of messages (not reference)", function()
    local s = Session.new()
    s:add("user", "original")
    local child = s:fork()
    -- Modify original after fork
    s.messages[1].content = "modified"
    -- Child should still have original content
    T.eq(child.messages[1].content, "original")
end)

-- ── Suite 8: Session - _msg_kv_ends tracking ─────────────────────────────────

T.suite("Session - _msg_kv_ends tracking")

T.test("new session has empty _msg_kv_ends", function()
    local s = Session.new()
    T.is_type(s._msg_kv_ends, "table")
    T.eq(#s._msg_kv_ends, 0)
end)

T.test("reset() clears _msg_kv_ends", function()
    local s = Session.new()
    s._msg_kv_ends = { 10, 20, 30 }
    s:reset()
    T.eq(#s._msg_kv_ends, 0)
end)

T.test("fork() copies _msg_kv_ends as independent table", function()
    local s = Session.new()
    s._msg_kv_ends = { 10, 20, 30 }
    local child = s:fork()
    T.eq(#child._msg_kv_ends, 3)
    T.eq(child._msg_kv_ends[1], 10)
    T.eq(child._msg_kv_ends[3], 30)
    -- Modify parent - child must not be affected
    s._msg_kv_ends[1] = 999
    T.eq(child._msg_kv_ends[1], 10)
end)

T.test("fork() child _msg_kv_ends is independent", function()
    local s = Session.new()
    s._msg_kv_ends = { 50, 100 }
    local child = s:fork()
    child._msg_kv_ends[1] = 0
    T.eq(s._msg_kv_ends[1], 50)  -- parent unchanged
end)

-- ── Suite 9: ContextManager - constructor options ─────────────────────────────

T.suite("ContextManager - constructor options")

-- Mock ctx and vocab (no real model needed for constructor tests)
local function mock_ctx(n_seq_max, can_shift)
    return {
        n_seq_max  = function() return n_seq_max or 1 end,
        kv_can_shift = function() return can_shift or false end,
    }
end
local mock_vocab = {}

local ContextManager = require "ion7.llm.context_manager"

T.test("default n_sink is 4", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    T.eq(cm:stats().n_sink, 4)
end)

T.test("n_sink=0 disables attention sink", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab, { n_sink = 0 })
    T.eq(cm:stats().n_sink, 0)
end)

T.test("n_sink custom value stored", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab, { n_sink = 8 })
    T.eq(cm:stats().n_sink, 8)
end)

T.test("default eviction is 'message'", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    T.eq(cm:stats().eviction, "message")
end)

T.test("eviction='fifo' stored correctly", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab, { eviction = "fifo" })
    T.eq(cm:stats().eviction, "fifo")
end)

T.test("stats() includes n_sink and eviction", function()
    local cm = ContextManager.new(mock_ctx(2), mock_vocab, { n_sink = 6, eviction = "fifo" })
    local s = cm:stats()
    T.eq(s.n_sink,    6)
    T.eq(s.eviction,  "fifo")
    T.is_type(s.slots_total,   "number")
    T.is_type(s.prefix_tokens, "number")
end)

T.test("has_prefix() false before set_system()", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    T.eq(cm:has_prefix(), false)
end)

T.test("hooks: set/get/clear", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    local fn = function() end
    cm:set_hook("before_encode", fn)
    T.eq(cm:get_hook("before_encode"), fn)
    cm:clear_hook("before_encode")
    T.eq(cm:get_hook("before_encode"), nil)
end)

T.test("set_hook validates name type", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    T.err(function() cm:set_hook(123, function() end) end, "hook name must be a string")
end)

T.test("set_hook validates fn type", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    T.err(function() cm:set_hook("before_encode", "not_a_fn") end, "hook must be a function")
end)

-- ── Suite 10: ContextManager - on_evict hook ────────────────────────────────���─

T.suite("ContextManager - on_evict hook")

T.test("on_evict hook registered and retrievable", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    cm:set_hook("on_evict", function() end)
    T.eq(type(cm:get_hook("on_evict")), "function")
end)

T.test("on_evict hook cleared by clear_hook", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    cm:set_hook("on_evict", function() end)
    cm:clear_hook("on_evict")
    T.eq(cm:get_hook("on_evict"), nil)
end)

T.test("on_evict and before_encode are independent hooks", function()
    local cm = ContextManager.new(mock_ctx(), mock_vocab)
    local fn1 = function() end
    local fn2 = function() end
    cm:set_hook("before_encode", fn1)
    cm:set_hook("on_evict", fn2)
    T.eq(cm:get_hook("before_encode"), fn1)
    T.eq(cm:get_hook("on_evict"), fn2)
    cm:clear_hook("on_evict")
    T.eq(cm:get_hook("before_encode"), fn1)  -- unaffected
    T.eq(cm:get_hook("on_evict"), nil)
end)

T.test("on_evict hook receives evicted messages (simulation)", function()
    -- Simulate the hook contract: fn(evicted_messages, session) -> messages?
    -- We test the contract structure, not the full prepare() (requires model)
    local received = nil
    local hook = function(evicted, _)
        received = evicted
        return nil  -- drop silently
    end
    -- Invoke the hook directly as ContextManager would
    local evicted = { { role = "user", content = "old message" } }
    hook(evicted, {})
    T.ok(received ~= nil, "hook received evicted messages")
    T.eq(#received, 1)
    T.eq(received[1].role, "user")
    T.eq(received[1].content, "old message")
end)

T.test("on_evict hook returning replacement (simulation)", function()
    local hook = function(evicted, _)
        local text = evicted[1].content
        return { { role = "system", content = "[Summary: " .. text .. "]" } }
    end
    local msgs = { { role = "user", content = "past topic" } }
    local replacement = hook(msgs, {})
    T.ok(replacement ~= nil, "hook returned replacement")
    T.eq(#replacement, 1)
    T.eq(replacement[1].role, "system")
    T.ok(replacement[1].content:find("Summary"))
end)

T.test("on_evict hook returning nil drops silently (simulation)", function()
    local hook = function(_, _) return nil end
    local result = hook({ { role = "user", content = "x" } }, {})
    T.eq(result, nil)
end)

-- ── Summary ──────────────────────────────��──────────────────────────────��─────

local ok = T.summary()
os.exit(ok and 0 or 1)
