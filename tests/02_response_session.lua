#!/usr/bin/env luajit
--- @module tests.02_response_session
--- @author  ion7 / Ion7 Project Contributors
---
--- Pure-Lua coverage for the data classes the engine produces and
--- consumes : `Response`, `Session`, and the `util.messages` helper.
---
--- These three modules carry no FFI surface — they store, copy and
--- serialise plain Lua tables. Exercising them without a model gives
--- us :
---
---   - confidence that conversation state moves cleanly between turns
---     (add_user / add_assistant / add_tool_result, fork, reset),
---   - tool-call envelope round-trip via add_assistant({ tool_calls }),
---   - session save / load JSON round-trip,
---   - validation of the `tool_call_id` and `role` constraints the chat
---     template will rely on later.
---
--- When these primitives misbehave silently, every higher-level test
--- becomes flaky for the wrong reason — fixing them here keeps the
--- model-dependent files focused on their actual job.

local T = require "tests.framework"
local H = require "tests.helpers"

local Response = require "ion7.llm.response"
local Session  = require "ion7.llm.session"
local messages = require "ion7.llm.util.messages"

-- ════════════════════════════════════════════════════════════════════════
-- Response — value object semantics
-- ════════════════════════════════════════════════════════════════════════

T.suite("Response — value-object shape")

T.test("new() defaults are sane", function()
    local r = Response.new()
    T.eq(r.content,     "")
    T.eq(r.thinking,    nil)
    T.eq(#r.tool_calls, 0)
    T.eq(#r.tokens,     0)
    T.eq(r.n_tokens,    0)
    T.eq(r.stop_reason, "stop")
    T.is_type(r.perf,   "table")
end)

T.test("new() copies the requested channels", function()
    local r = Response.new({
        content     = "Hello",
        thinking    = "I should greet politely.",
        tool_calls  = { { id = "0", name = "tick", arguments = {} } },
        tokens      = { 10, 20, 30 },
        stop_reason = "stop_string",
        perf        = { tok_per_s = 42.5 },
    })
    T.eq(r.content,           "Hello")
    T.eq(r.thinking,          "I should greet politely.")
    T.eq(#r.tool_calls,       1)
    T.eq(r.tool_calls[1].name, "tick")
    T.eq(r.n_tokens,          3)
    T.eq(r.stop_reason,       "stop_string")
    T.eq(r.perf.tok_per_s,    42.5)
end)

T.test("has_tools reflects the tool_calls array", function()
    T.eq(Response.new():has_tools(),                              false)
    T.eq(Response.new({ tool_calls = {} }):has_tools(),           false)
    T.eq(Response.new({ tool_calls = { { id = "0" } } }):has_tools(), true)
end)

T.test("__tostring returns content for log-friendly printing", function()
    local r = Response.new({ content = "ok" })
    T.eq(tostring(r), "ok")
end)

T.test("summary() includes tokens/throughput/reason", function()
    local r = Response.new({
        tokens      = { 1, 2, 3 },
        stop_reason = "length",
        perf        = { tok_per_s = 12.34 },
    })
    local s = r:summary()
    T.contains(s, "3 tok")
    T.contains(s, "length")
    T.contains(s, "12.3")
end)

-- ════════════════════════════════════════════════════════════════════════
-- util.messages — chat-template interop helpers
-- ════════════════════════════════════════════════════════════════════════

T.suite("util.messages — role validation + builders")

T.test("is_valid_role recognises the four canonical roles", function()
    T.eq(messages.is_valid_role("system"),    true)
    T.eq(messages.is_valid_role("user"),      true)
    T.eq(messages.is_valid_role("assistant"), true)
    T.eq(messages.is_valid_role("tool"),      true)
    T.eq(messages.is_valid_role("admin"),     false)
    T.eq(messages.is_valid_role(nil),         false)
end)

T.test("user / system / assistant builders pass content through", function()
    T.eq(messages.user("hi").role,          "user")
    T.eq(messages.user("hi").content,       "hi")
    T.eq(messages.system("be brief").role,  "system")
    T.eq(messages.assistant("ok").role,     "assistant")
end)

T.test("assistant() carries thinking + tool_calls extras", function()
    local m = messages.assistant("ok", {
        thinking   = "let me think",
        tool_calls = { { id = "0", name = "noop" } },
    })
    T.eq(m.thinking,        "let me think")
    T.eq(#m.tool_calls,     1)
end)

T.test("tool_result encodes table content via the bundled JSON lib", function()
    local m = messages.tool_result("call_42", { value = 7, ok = true })
    T.eq(m.role,         "tool")
    T.eq(m.tool_call_id, "call_42")
    -- The encoder is implementation-defined but must produce a string.
    T.is_type(m.content, "string")
    T.contains(m.content, "7")
    T.contains(m.content, "true")
end)

T.test("tool_result keeps string content as-is", function()
    local m = messages.tool_result("call_1", "raw text")
    T.eq(m.content, "raw text")
end)

T.test("validate raises on bad role / nil content / missing tool_call_id", function()
    T.err(function() messages.validate({ role = "stranger", content = "" }) end,
        "invalid role")
    T.err(function() messages.validate({ role = "user", content = nil }) end,
        "content")
    T.err(function() messages.validate({ role = "tool", content = "x" }) end,
        "tool_call_id")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Session — conversation history + KV bookkeeping
-- ════════════════════════════════════════════════════════════════════════

T.suite("Session — basic mutation + history shape")

T.test("new() carries optional system prompt", function()
    local s = Session.new({ system = "You are concise." })
    T.eq(s.system,        "You are concise.")
    T.eq(#s.messages,     0)
    T.eq(s.n_past,        0)
    T.eq(s._dirty,        false)
    T.eq(s._seq_snapshot, nil)
end)

T.test("each new() instance has a unique id", function()
    local a, b = Session.new(), Session.new()
    T.neq(a.id, b.id, "session ids must be unique")
end)

T.test("add_user marks the session dirty", function()
    local s = Session.new()
    T.eq(s._dirty, false)
    s:add_user("hi")
    T.eq(#s.messages, 1)
    T.eq(s.messages[1].role,    "user")
    T.eq(s.messages[1].content, "hi")
    T.eq(s._dirty, true)
end)

T.test("add_assistant carries thinking + tool_calls envelopes", function()
    local s = Session.new()
    s:add_user("compute 2+2")
    s:add_assistant("the answer is 4", {
        thinking   = "this is straightforward",
        tool_calls = { { id = "0", name = "calc" } },
    })
    local m = s.messages[2]
    T.eq(m.role,            "assistant")
    T.eq(m.content,         "the answer is 4")
    T.eq(m.thinking,        "this is straightforward")
    T.eq(#m.tool_calls,     1)
end)

T.test("add_tool_result tags the message with role=tool + id", function()
    local s = Session.new()
    s:add_tool_result("call_42", { value = 7 })
    T.eq(#s.messages, 1)
    T.eq(s.messages[1].role,         "tool")
    T.eq(s.messages[1].tool_call_id, "call_42")
end)

T.test("set_system mutates the system prompt and resets snapshot state", function()
    local s = Session.new({ system = "old" })
    s:_save_seq_snapshot("BLOB", 12)  -- pretend a previous chat ran
    T.eq(s._seq_snapshot, "BLOB")
    s:set_system("new prompt")
    T.eq(s.system,        "new prompt")
    T.eq(s._seq_snapshot, nil, "system change must invalidate snapshot")
    T.eq(s.n_past,        0)
end)

T.test("set_system with the same text is a no-op (snapshot survives)", function()
    local s = Session.new({ system = "same" })
    s:_save_seq_snapshot("BLOB", 12)
    s:set_system("same")
    T.eq(s._seq_snapshot, "BLOB",
        "no-op set_system must NOT invalidate the snapshot")
end)

T.test("all_messages prepends the system message when set", function()
    local s = Session.new({ system = "be terse" })
    s:add_user("hi")
    s:add_assistant("hello")
    local m = s:all_messages()
    T.eq(#m,         3)
    T.eq(m[1].role,  "system")
    T.eq(m[2].role,  "user")
    T.eq(m[3].role,  "assistant")
end)

T.test("all_messages returns just messages when no system is set", function()
    local s = Session.new()
    s:add_user("hi")
    T.eq(#s:all_messages(), 1)
end)

T.test("reset wipes history and KV bookkeeping but keeps the system", function()
    local s = Session.new({ system = "stay" })
    s:add_user("hi")
    s:_save_seq_snapshot("BLOB", 5)
    s:reset()
    T.eq(s.system,        "stay")
    T.eq(#s.messages,     0)
    T.eq(s.n_past,        0)
    T.eq(s._seq_snapshot, nil)
    T.eq(s._dirty,        false)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Session :_save_seq_snapshot + _has_clean_snapshot
-- ════════════════════════════════════════════════════════════════════════

T.suite("Session — snapshot bookkeeping (kv layer contract)")

T.test("save_snapshot records blob + n_past and clears the dirty flag", function()
    local s = Session.new()
    s:add_user("hi")
    T.eq(s._dirty, true)
    s:_save_seq_snapshot("BLOB", 17)
    T.eq(s._seq_snapshot,     "BLOB")
    T.eq(s.n_past,            17)
    T.eq(s._dirty,            false)
    T.eq(s:_has_clean_snapshot(), true)
end)

T.test("any new message after save invalidates the clean flag", function()
    local s = Session.new()
    s:add_user("hi")
    s:_save_seq_snapshot("BLOB", 10)
    T.eq(s:_has_clean_snapshot(), true)
    s:add_user("again")
    T.eq(s:_has_clean_snapshot(), false,
        "new message must mark snapshot dirty")
end)

T.test("a session with no snapshot reports !has_clean_snapshot", function()
    local s = Session.new()
    T.eq(s:_has_clean_snapshot(), false)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Session :fork
-- ════════════════════════════════════════════════════════════════════════

T.suite("Session — fork() clones history without sharing references")

T.test("fork copies the message list", function()
    local s = Session.new({ system = "stay terse" })
    s:add_user("hi")
    s:add_assistant("hey", { thinking = "easy" })

    local c = s:fork()
    T.eq(c.system,    s.system)
    T.eq(#c.messages, #s.messages)
    T.eq(c.messages[2].thinking, "easy")
end)

T.test("forked messages are independent of the parent's", function()
    local s = Session.new()
    s:add_user("hi")
    local c = s:fork()
    c.messages[1].content = "MUTATED"
    T.eq(s.messages[1].content, "hi",
        "parent's message must not change when fork is mutated")
end)

T.test("fork preserves n_past and _msg_kv_ends but not the snapshot blob", function()
    local s = Session.new()
    s:add_user("hi")
    s.n_past = 12
    s._msg_kv_ends = { 12 }
    s._seq_snapshot = "PARENT_BLOB"

    local c = s:fork()
    T.eq(c.n_past,               12)
    T.eq(#c._msg_kv_ends,        1)
    T.eq(c._seq_snapshot,        nil,
        "fork must not carry the parent's snapshot — kv layer re-snapshots")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Session — JSON persistence round-trip
-- ════════════════════════════════════════════════════════════════════════

T.suite("Session — serialize / deserialize / save / load")

T.test("serialize captures system + messages but not KV state", function()
    local s = Session.new({ system = "cool" })
    s:add_user("hi")
    s:add_assistant("hello", { thinking = "obvious" })
    s:_save_seq_snapshot("BLOB", 5)

    local t = s:serialize()
    T.eq(t.system,              "cool")
    T.eq(#t.messages,           2)
    T.eq(t.messages[1].role,    "user")
    T.eq(t.messages[2].thinking, "obvious")
    T.eq(t.kv,                  nil, "KV blob must NOT round-trip through JSON")
end)

T.test("deserialize rebuilds an equivalent dirty session", function()
    local original = Session.new({ system = "cool" })
    original:add_user("hi")

    local cloned = Session.deserialize(original:serialize())
    T.eq(cloned.system,        "cool")
    T.eq(#cloned.messages,     1)
    T.eq(cloned._dirty,        true,
        "deserialised session must be dirty so the engine re-encodes")
    T.eq(cloned._seq_snapshot, nil)
end)

T.test("save / load round-trips through a temp file", function()
    local p = H.tmpfile("ion7-llm-session-" .. tostring(os.time()) .. ".json")

    local s = Session.new({ system = "stay" })
    s:add_user("hi")
    s:add_assistant("hello")

    local ok, err = s:save(p)
    T.ok(ok, "save error: " .. tostring(err))

    local loaded, lerr = Session.load(p)
    T.ok(loaded, "load error: " .. tostring(lerr))
    T.eq(loaded.system,         "stay")
    T.eq(#loaded.messages,      2)
    T.eq(loaded.messages[2].role, "assistant")

    H.try_remove(p)
end)

T.test("load returns (nil, err) for a missing file", function()
    local r, err = Session.load("/tmp/ion7-no-such-file-" .. os.time() .. ".json")
    T.eq(r,       nil)
    T.is_type(err, "string")
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
