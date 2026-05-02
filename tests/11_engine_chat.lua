#!/usr/bin/env luajit
--- @module tests.11_engine_chat
--- @author  ion7 / Ion7 Project Contributors
---
--- Single-session inference path : `Engine:chat`, `Engine:complete`,
--- `Engine:cm` accessors. Builds the pipeline once at the top of the
--- file and reuses the model + context across every test for speed.
---
--- What we cover :
---
---   - `chat(session)` returns a `Response` with non-empty content,
---     a `stop_reason` from the documented set, a tokens array.
---   - The session's per-seq `n_past` advances past the prefill +
---     generated tokens.
---   - The session's `_last_response` accessor is wired.
---   - `complete(prompt)` is the ephemeral-session shortcut.
---   - `max_tokens` caps the generation, surfaces as `stop_reason ==
---     "length"` when reached.
---   - `stop_strings` halts on a custom marker.
---   - Two consecutive `chat` calls on the same session preserve the
---     conversation (the second response can reference the first).
---
--- We deliberately avoid asserting on EXACT model output — small
--- models give probabilistic answers and the suite would be flaky.
--- The asserts focus on the contract (shape, length, monotonicity).

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm, ctx, vocab, cm, engine = H.pipeline(model, {
    n_ctx     = 4096,
    -- Multiple slots so each test can spin up its own throwaway
    -- session without colliding with earlier ones. Sessions get
    -- released at test end to keep the pool from filling up.
    n_seq_max = 4,
    headroom  = 256,
})

-- Helper : run `fn` with a Session that gets released back to the
-- slot pool after the test, regardless of whether the body raised.
local function with_session(opts, fn)
    local s = llm.Session.new(opts)
    local ok, err = pcall(fn, s)
    cm:release(s)
    if not ok then error(err, 0) end
end

cm:set_system("You are a concise assistant. Answer in plain English.")

-- ════════════════════════════════════════════════════════════════════════
-- Engine — chat() basic shape
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:chat — Response shape on a happy-path turn")

T.test("returns a Response with non-empty content + tokens array", function()
    with_session(nil, function(s)
        s:add_user("Reply with the single word: ready.")
        local r = engine:chat(s, { max_tokens = 16 })

        T.is_type(r,                "table")
        T.is_type(r.content,        "string")
        T.gt(#r.content,            0)
        T.is_type(r.tokens,         "table")
        T.gt(r.n_tokens,            0)
        T.eq(#r.tokens,             r.n_tokens)
        T.one_of(r.stop_reason, { "stop", "stop_string", "length", "tool_use", "error" })
    end)
end)

T.test("session.n_past advances past prefill + generated tokens", function()
    with_session(nil, function(s)
        s:add_user("Say one word.")
        local before = s.n_past
        local r = engine:chat(s, { max_tokens = 8 })
        T.gt(s.n_past, before, "n_past must move forward after a chat")
        T.gt(s.n_past, r.n_tokens, "n_past also includes the prompt prefill")
    end)
end)

T.test("session.last_response() returns the same Response", function()
    with_session(nil, function(s)
        s:add_user("Reply with: ok.")
        local r = engine:chat(s, { max_tokens = 8 })
        T.eq(s:last_response(), r)
    end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- max_tokens — length cap
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:chat — max_tokens behaviour")

T.test("max_tokens=1 stops after exactly one decode", function()
    with_session(nil, function(s)
        s:add_user("Tell me a long story about a dragon.")
        local r = engine:chat(s, { max_tokens = 1 })
        T.eq(r.n_tokens, 1)
        T.one_of(r.stop_reason, { "length", "stop" })
    end)
end)

T.test("max_tokens caps generation at the requested count", function()
    with_session(nil, function(s)
        s:add_user("Tell me a long story about a dragon.")
        local r = engine:chat(s, { max_tokens = 16 })
        T.gt(r.n_tokens, 0)
        T.eq(r.n_tokens <= 16, true,
            string.format("got %d tokens for max_tokens=16", r.n_tokens))
    end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- stop_strings — halt on a custom marker
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:chat — custom stop strings")

T.test("a generation that crosses a stop string halts with stop_string", function()
    -- We instruct the model to insert the marker. With a precise
    -- sampler the instruction is reasonably reliable, but small
    -- models occasionally paraphrase — accept both outcomes.
    with_session(nil, function(s)
        s:add_user("Output exactly: hello [DONE] then stop.")
        local r = engine:chat(s, {
            max_tokens   = 32,
            sampler      = llm.sampler.profiles.precise(),
            stop_strings = { "[DONE]" },
        })
        if r.stop_reason == "stop_string" then
            T.eq(r.content:find("[DONE]", 1, true), nil,
                "stop string must be clipped from the content")
        end
    end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- complete() — ephemeral session shortcut
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:complete — one-shot helper")

T.test("complete returns a Response without a persistent session", function()
    local r = engine:complete("Reply with the single word: ready.", {
        max_tokens = 8,
        system     = "You are concise.",
    })
    T.gt(#r.content, 0)
    T.gt(r.n_tokens, 0)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Multi-turn — history is preserved across chat calls
-- ════════════════════════════════════════════════════════════════════════

T.suite("Engine:chat — multi-turn conversation")

T.test("a second turn can reference the first turn", function()
    with_session(nil, function(s)
        s:add_user("My favourite colour is teal. Reply with one word: ok.")
        local r1 = engine:chat(s, { max_tokens = 6 })
        s:add_assistant(r1.content, { thinking = r1.thinking })

        s:add_user("What colour did I just say? Reply with one word.")
        local r2 = engine:chat(s, { max_tokens = 6 })
        T.gt(#r2.content, 0)
    end)
end)

T.test("session.messages grows after add_user / add_assistant", function()
    with_session(nil, function(s)
        s:add_user("hi")
        local r = engine:chat(s, { max_tokens = 4 })
        s:add_assistant(r.content)
        T.eq(#s.messages, 2)
        T.eq(s.messages[1].role, "user")
        T.eq(s.messages[2].role, "assistant")
    end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Cleanup
-- ════════════════════════════════════════════════════════════════════════

ctx:free()
local ok = T.summary()
os.exit(ok and 0 or 1)
