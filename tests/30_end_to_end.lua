#!/usr/bin/env luajit
--- @module tests.30_end_to_end
--- @author  ion7 / Ion7 Project Contributors
---
--- End-to-end smoke test : a small but realistic chat session that
--- crosses every layer of the public API.
---
--- Flow :
---
---   1. Boot the backend, load the model.
---   2. Build a context with `n_seq_max = 4`, `kv_unified = true`
---      (auto via H.pipeline) and a 4 K context window.
---   3. Build a `ContextManager`, set a system prompt — the prefix
---      cache encodes it once.
---   4. Create a Session, run three multi-turn user → engine:chat
---      → add_assistant cycles.
---   5. Stream a fourth turn and concatenate the chunks ; assert that
---      the streamed content matches the post-stream Response.
---   6. Fork the session via `cm:fork`, run a divergent turn on the
---      fork ; the parent's history is unchanged.
---   7. Release every session, verify the slot pool is full.
---
--- Asserting on EXACT model output would make this brittle on small
--- weights, so the checks focus on shape + monotonicity (n_past
--- grows, sessions don't share rows, the streaming path matches the
--- batch path on the same call).

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm, ctx, vocab, cm, engine = H.pipeline(model, {
    n_ctx     = 4096,
    n_seq_max = 4,
    headroom  = 256,
})

cm:set_system("You are a concise, helpful assistant. Reply briefly.")

T.suite("End-to-end — three-turn batch chat")

local main = llm.Session.new()

T.test("turn 1 / 3 produces a non-empty response", function()
    main:add_user("Reply with the single word: ready.")
    local r = engine:chat(main, { max_tokens = 8 })
    T.gt(#r.content, 0)
    T.gt(r.n_tokens,  0)
    main:add_assistant(r.content, { thinking = r.thinking })
end)

T.test("turn 2 / 3 grows the history without collapsing", function()
    local before_n_msg = #main.messages
    main:add_user("In one short sentence, what is LuaJIT ?")
    local r = engine:chat(main, { max_tokens = 32 })
    main:add_assistant(r.content, { thinking = r.thinking })
    T.eq(#main.messages, before_n_msg + 2,
        "user + assistant must be appended this turn")
end)

T.test("turn 3 / 3 completes after history accumulation", function()
    main:add_user("Reply with one word: ok.")
    local r = engine:chat(main, { max_tokens = 4 })
    main:add_assistant(r.content)
    T.gt(main.n_past, 0)
end)

T.suite("End-to-end — streaming a follow-up turn")

T.test("the streamed content matches the parsed Response.content prefix", function()
    main:add_user("Reply with one short sentence about LuaJIT.")
    local content_parts = {}
    for chunk in engine:stream(main, { max_tokens = 32 }) do
        if chunk.kind == "content" then
            content_parts[#content_parts + 1] = chunk.text
        end
    end
    local streamed = table.concat(content_parts)
    local resp = main:last_response()
    T.gt(#resp.content, 0)
    T.eq(streamed:sub(1, #resp.content), resp.content,
        "streamed content must agree with the post-parse Response prefix")
    main:add_assistant(resp.content, { thinking = resp.thinking })
end)

T.suite("End-to-end — fork + divergent continuation")

T.test("forked session keeps the parent's history but lands on a fresh seq", function()
    local fork = cm:fork(main)
    T.neq(fork.seq_id, main.seq_id, "fork must own its own KV row")
    T.eq(#fork.messages, #main.messages)

    -- Diverge : ask the fork a new question, leave the parent untouched.
    fork:add_user("Reply with a different word: branch.")
    local rf = engine:chat(fork, { max_tokens = 4 })
    T.gt(#rf.content, 0)

    -- Parent history is unchanged.
    T.eq(main.messages[#main.messages].role, "assistant",
        "parent's last message should still be the assistant turn from before")
    cm:release(fork)
end)

T.suite("End-to-end — release returns slots to the pool")

T.test("releasing every session restores the slot pool", function()
    local before_release = cm:stats().slots_free
    cm:release(main)
    local after_release = cm:stats().slots_free
    T.gte(after_release, before_release + 1,
        "the released slot must be back in the pool")
end)

ctx:free()
local ok = T.summary()
os.exit(ok and 0 or 1)
