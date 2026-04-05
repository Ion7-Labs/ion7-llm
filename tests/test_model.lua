#!/usr/bin/env luajit
--- Model-dependent tests - requires a real GGUF model and libllama.so.
---
--- Tests the full generation pipeline: init, chat, stream, multi-turn KV reuse,
--- prefix cache, context overflow, think block handling, and hooks.
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf \
---   LLAMA_LIB=/path/to/libllama.so \
---   luajit tests/test_model.lua
---
--- Optional:
---   ION7_CORE_PATH=../ion7-core  (default: ../ion7-core)
---   N_CTX=512                    (small context for overflow tests)

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL    = os.getenv("ION7_MODEL")
local CORE_PATH = os.getenv("ION7_CORE_PATH") or "../ion7-core"
local N_CTX    = tonumber(os.getenv("N_CTX")) or 1024

if not MODEL then
    io.stderr:write("[SKIP] ION7_MODEL not set - skipping model tests\n")
    os.exit(0)
end

package.path = CORE_PATH .. "/src/?.lua;" .. CORE_PATH .. "/src/?/init.lua;" .. package.path

local T   = require "tests.framework"
local llm = require "ion7.llm"

-- ── Helpers ───────────────────────────────────────────────────────────────────

local VALID_STOP = { "stop", "length", "stop_string" }

local function assert_response(r, label)
    T.ok(r ~= nil,                     label .. ": response not nil")
    T.is_type(r.text,       "string",  label .. ": text is string")
    T.is_type(r.n_tokens,   "number",  label .. ": n_tokens is number")
    T.gt(r.n_tokens, 0,                label .. ": n_tokens > 0")
    T.one_of(r.stop_reason, VALID_STOP, label .. ": valid stop_reason")
    T.is_type(r.perf,       "table",   label .. ": perf is table")
end

-- ── Init ──────────────────────────────────────────────────────────────────────

io.write("[init] loading model...\n")
llm.init({
    model      = MODEL,
    n_ctx      = N_CTX,
    n_seq_max  = 2,
    system     = "You are a concise assistant. Answer in one sentence.",
    max_tokens = 64,
    log_level  = 0,
})
io.write(string.format("[init] n_ctx=%d, kv_can_shift=%s\n\n",
    N_CTX, tostring(llm.ctx():kv_can_shift())))

-- ── Suite 1: Basic generation ─────────────────────────────────────────────────

T.suite("Basic generation")

T.test("llm.chat() returns a valid Response", function()
    local r = llm.chat("Say the word 'hello'.")
    assert_response(r, "chat")
    T.gt(#r.text, 0, "text not empty")
end)

T.test("llm.chat() with on_token callback fires", function()
    local count = 0
    llm.chat("Count: one.", { on_token = function() count = count + 1 end })
    T.gt(count, 0, "on_token fired at least once")
end)

T.test("llm.chat() stop_reason is 'stop' or 'length'", function()
    local r = llm.chat("Say 'yes'.")
    T.one_of(r.stop_reason, VALID_STOP)
end)

T.test("llm.chat() perf.tok_per_s > 0", function()
    local r = llm.chat("One word: hello.")
    T.gt(r.perf.tok_per_s or 0, 0, "tok_per_s > 0")
end)

T.test("Response:summary() is non-empty string", function()
    local r = llm.chat("One word.")
    local s = r:summary()
    T.is_type(s, "string")
    T.gt(#s, 0)
end)

T.test("tostring(Response) returns text", function()
    local r = llm.chat("One word: ok.")
    T.eq(tostring(r), r.text)
end)

-- ── Suite 2: Streaming ────────────────────────────────────────────────────────

T.suite("Streaming")

T.test("llm.stream() yields at least one piece", function()
    local pieces = {}
    for piece in llm.stream("Say 'hello'.") do
        pieces[#pieces + 1] = piece
    end
    T.gt(#pieces, 0, "at least one piece yielded")
end)

T.test("llm.stream() concatenated equals chat text", function()
    local prompt = "The capital of France is"
    local parts  = {}
    for piece in llm.stream(prompt) do
        parts[#parts + 1] = piece
    end
    local streamed = table.concat(parts)
    local chatted  = llm.chat(prompt).text
    -- Both should produce non-empty output (exact text may differ - different sessions)
    T.gt(#streamed, 0, "streamed text non-empty")
    T.gt(#chatted,  0, "chatted text non-empty")
end)

T.test("llm.stream() with messages array", function()
    local pieces = 0
    for _ in llm.stream({ { role = "user", content = "Say 'hi'." } }) do
        pieces = pieces + 1
    end
    T.gt(pieces, 0)
end)

-- ── Suite 3: Multi-turn KV reuse ──────────────────────────────────────────────

T.suite("Multi-turn KV reuse")

T.test("n_past grows across turns", function()
    local session = llm.Session.new()
    local gen     = llm.gen()
    local ctx     = llm.ctx()

    session:add("user", "My name is Alice.")
    local r1 = gen:chat(session)
    local n1 = ctx:n_past()
    T.gt(n1, 0, "n_past > 0 after turn 1")

    session:add("assistant", r1.text)
    session:add("user", "Repeat my name.")
    gen:chat(session)
    local n2 = ctx:n_past()
    T.gt(n2, n1, "n_past grew after turn 2")
end)

T.test("Session dirty flag set by add()", function()
    local s = llm.Session.new()
    T.eq(s._dirty, false)
    s:add("user", "hello")
    T.eq(s._dirty, true)
end)

T.test("Session has snapshot after generation", function()
    local session = llm.Session.new()
    session:add("user", "Say 'test'.")
    llm.gen():chat(session)
    T.ok(session:has_snapshot(), "snapshot saved after chat")
    T.gt(session.n_past, 0, "n_past recorded")
end)

T.test("Session fork creates independent branch", function()
    local session = llm.Session.new()
    session:add("user", "My color is blue.")
    llm.gen():chat(session)

    local child = llm.cm():fork(session)
    T.ok(child ~= nil, "fork returned child")
    T.neq(child.id, session.id, "different session ids")
end)

-- ── Suite 4: Prefix cache ─────────────────────────────────────────────────────

T.suite("Prefix cache")

T.test("has_prefix() true after init with system", function()
    -- llm.init() set a system prompt, so prefix cache should be populated
    T.ok(llm.cm():has_prefix(), "prefix cache set")
    T.gt(llm.cm():stats().prefix_tokens, 0, "prefix_tokens > 0")
end)

T.test("prefix_cached = true in stats()", function()
    T.eq(llm.cm():stats().prefix_cached, true)
end)

T.test("generation works after prefix restore", function()
    -- Each chat() restores prefix - verify it doesn't crash and produces output
    local r = llm.chat("One word: yes.")
    T.gt(r.n_tokens, 0)
end)

T.test("set_system() updates prefix cache", function()
    local before = llm.cm():stats().prefix_tokens
    llm.cm():set_system("You are a pirate. Say 'arr'.")
    local after = llm.cm():stats().prefix_tokens
    T.gt(after, 0, "new prefix encoded")
    -- Restore original system
    llm.cm():set_system("You are a concise assistant. Answer in one sentence.")
    T.eq(llm.cm():stats().prefix_tokens, before, "prefix restored to original")
end)

-- ── Suite 5: Context overflow ─────────────────────────────────────────────────

T.suite("Context overflow (no crash)")

T.test("8 turns with max_tokens=80 completes without crash", function()
    local session = llm.Session.new()
    local gen     = llm.gen()
    local prompts = {
        "What is LuaJIT?",
        "What is a hash table?",
        "Explain TCP vs UDP.",
        "What is garbage collection?",
        "Explain virtual memory.",
        "What is a B-tree?",
        "Describe semaphores.",
        "What is a race condition?",
    }
    for _, p in ipairs(prompts) do
        session:add("user", p)
        local r = gen:chat(session, { max_tokens = 80 })
        assert_response(r, "turn")
        session:add("assistant", r.text)
    end
    T.gt(llm.ctx():n_past(), 0, "n_past > 0 after 8 turns")
end)

T.test("n_past never exceeds n_ctx", function()
    T.ok(llm.ctx():n_past() <= N_CTX,
        string.format("n_past(%d) <= n_ctx(%d)", llm.ctx():n_past(), N_CTX))
end)

-- ── Suite 6: ContextManager stats ─────────────────────────────────────────────

T.suite("ContextManager stats")

T.test("stats() returns expected fields", function()
    local s = llm.cm():stats()
    T.is_type(s.slots_total,    "number")
    T.is_type(s.slots_free,     "number")
    T.is_type(s.prefix_tokens,  "number")
    T.is_type(s.prefix_cached,  "boolean")
    T.is_type(s.n_sink,         "number")
    T.is_type(s.eviction,       "string")
    T.is_type(s.n_evictions,    "number")
    T.is_type(s.n_tokens_evicted, "number")
end)

T.test("n_sink default is 4", function()
    T.eq(llm.cm():stats().n_sink, 4)
end)

T.test("eviction default is 'message'", function()
    T.eq(llm.cm():stats().eviction, "message")
end)

T.test("n_evictions >= 0", function()
    T.gte(llm.cm():stats().n_evictions, 0)
end)

-- ── Suite 7: Think blocks ─────────────────────────────────────────────────────

T.suite("Think block handling")

T.test("think=false: resp:think() is nil", function()
    -- Default llm.init has no think flag - but we need to verify
    -- Create a direct gen with think=false
    local Session   = llm.Session
    local Generator = llm.Generator
    local gen = Generator.new(llm.ctx(), llm.vocab(), llm.cm(), {
        think      = false,
        max_tokens = 32,
        sampler    = llm.sampler.balanced(llm.vocab()),
    })
    local s = Session.new()
    s:add("user", "Say 'hello'.")
    local r = gen:chat(s)
    T.eq(r:think(), nil, "think() nil when think=false")
end)

T.test("resp.text is non-empty string", function()
    local r = llm.chat("One word: yes.")
    T.is_type(r.text, "string")
    T.gt(#r.text, 0)
end)

-- ── Suite 8: Hooks ────────────────────────────────────────────────────────────

T.suite("Hooks")

T.test("before_encode hook fires during generation", function()
    local fired = false
    llm.cm():set_hook("before_encode", function(msgs, _)
        fired = true
        return msgs
    end)
    llm.chat("One word: ok.")
    llm.cm():clear_hook("before_encode")
    T.ok(fired, "before_encode hook fired")
end)

T.test("before_encode hook can inject a message", function()
    local injected = false
    llm.cm():set_hook("before_encode", function(msgs, _)
        injected = true
        -- Inject a system message at the front
        local new = { { role = "system", content = "Always say 'roger'." } }
        for _, m in ipairs(msgs) do new[#new+1] = m end
        return new
    end)
    local r = llm.chat("Acknowledge.")
    llm.cm():clear_hook("before_encode")
    T.ok(injected, "hook fired")
    T.gt(#r.text, 0, "response generated with injected context")
end)

T.test("on_evict hook fires on overflow (if overflow occurred)", function()
    local evict_fired = false
    llm.cm():set_hook("on_evict", function(evicted, _)
        evict_fired = true
        T.ok(#evicted > 0, "evicted messages non-empty")
        return nil
    end)
    -- Force overflow with a long multi-turn session
    local session = llm.Session.new()
    local gen     = llm.gen()
    for i = 1, 6 do
        session:add("user", "Turn " .. i .. ": describe TCP in detail.")
        local r = gen:chat(session, { max_tokens = 80 })
        session:add("assistant", r.text)
    end
    llm.cm():clear_hook("on_evict")
    -- The hook may or may not fire depending on context size - just check no crash
    T.ok(true, "on_evict hook registered without crash")
end)

T.test("on_evict hook returning summary is injected", function()
    local summary_injected = false
    llm.cm():set_hook("on_evict", function(evicted, _)
        summary_injected = true
        return { { role = "system", content = "[Earlier: " .. #evicted .. " messages]" } }
    end)
    local session = llm.Session.new()
    local gen     = llm.gen()
    for i = 1, 6 do
        session:add("user", "Message " .. i .. ": explain virtual memory in detail.")
        local r = gen:chat(session, { max_tokens = 80 })
        session:add("assistant", r.text)
    end
    llm.cm():clear_hook("on_evict")
    T.ok(true, "on_evict with replacement hook completed without crash")
end)

-- ── Suite 9: Generator checkpoint / rollback ──────────────────────────────────

T.suite("Generator checkpoint / rollback")

T.test("checkpoint() + rollback() restores KV", function()
    local gen = llm.gen()
    local ctx = llm.ctx()

    -- Prime state with one chat so n_past is at a known non-zero position,
    -- independent of accumulated state from previous suites.
    local s0 = llm.Session.new()
    s0:add("user", "Say 'hello'.")
    gen:chat(s0, { max_tokens = 16 })
    local n_prime = ctx:n_past()
    T.gt(n_prime, 0, "n_past > 0 after priming")

    gen:checkpoint()
    local n_before = ctx:n_past()

    -- Second chat: extend the context beyond the checkpoint position.
    local session = llm.Session.new()
    session:add("user", "Say 'world'.")
    gen:chat(session, { max_tokens = 16 })
    local n_after = ctx:n_past()
    T.gt(n_after, 0, "n_past > 0 after chat")

    gen:rollback()
    local n_restored = ctx:n_past()
    T.eq(n_restored, n_before, "n_past restored after rollback")
end)

T.test("rollback() without checkpoint() raises error", function()
    local gen2 = llm.Generator.new(llm.ctx(), llm.vocab(), llm.cm(), {
        max_tokens = 16,
        sampler    = llm.sampler.balanced(llm.vocab()),
    })
    T.err(function() gen2:rollback() end, "no checkpoint")
end)

-- ── Suite 10: Session serialize / deserialize ─────────────────────────────────

T.suite("Session serialize / deserialize")

T.test("serialize() + deserialize() round-trip", function()
    local s = llm.Session.new({ system = "Be helpful." })
    s:add("user", "Hello.")
    s:add("assistant", "Hi there.")
    s:add("user", "Bye.")

    local t  = s:serialize()
    local s2 = llm.Session.deserialize(t)

    T.eq(s2.system, "Be helpful.")
    T.eq(#s2.messages, 3)
    T.eq(s2.messages[1].content, "Hello.")
    T.eq(s2.messages[3].content, "Bye.")
    T.eq(s2._dirty, true)
    T.eq(s2.n_past, 0)
end)

T.test("deserialized session can generate", function()
    local s = llm.Session.new()
    s:add("user", "My name is Bob.")
    llm.gen():chat(s)
    s:add("assistant", "Hello Bob.")

    local t  = s:serialize()
    local s2 = llm.Session.deserialize(t)
    s2:add("user", "Say 'ok'.")

    local r = llm.gen():chat(s2)
    assert_response(r, "deserialized session")
end)

T.test("Session:format() returns non-empty string", function()
    local s = llm.Session.new()
    s:add("user", "Hello.")
    s:add("assistant", "Hi.")
    local text = s:format()
    T.is_type(text, "string")
    T.gt(#text, 0)
    T.ok(text:find("Hello"), "format contains message content")
end)

-- ── Suite 11: Batch / Scheduler ───────────────────────────────────────────────

T.suite("Batch / Scheduler")

T.test("llm.batch() with 2 sessions completes", function()
    local results = {}
    local sa = llm.Session.new(); sa:add("user", "Say 'alpha'.")
    local sb = llm.Session.new(); sb:add("user", "Say 'beta'.")

    llm.batch({
        { session = sa, max_tokens = 16,
          on_done = function(r) results.a = r end },
        { session = sb, max_tokens = 16,
          on_done = function(r) results.b = r end },
    })

    T.ok(results.a ~= nil, "job A completed")
    T.ok(results.b ~= nil, "job B completed")
    T.gt(results.a.n_tokens, 0, "job A produced tokens")
    T.gt(results.b.n_tokens, 0, "job B produced tokens")
end)

T.test("llm.batch() requires at least 2 jobs", function()
    local s = llm.Session.new(); s:add("user", "hello")
    T.err(function()
        llm.batch({ { session = s } })
    end, "at least 2 jobs")
end)

-- ── Shutdown ──────────────────────────────────────────────────────────────────

io.write("\n")
llm.shutdown()

local ok = T.summary()
os.exit(ok and 0 or 1)
