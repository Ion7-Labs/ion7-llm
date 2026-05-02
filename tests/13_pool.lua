#!/usr/bin/env luajit
--- @module tests.13_pool
--- @author  ion7 / Ion7 Project Contributors
---
--- Multi-session inference via `Pool`. Each tick packs one row per
--- active slot into a single `llama_decode` call, so N concurrent
--- conversations share the same forward pass. The aggregate throughput
--- is the value-add — a sequential `for s in sessions do engine:chat(s)`
--- loop would issue N independent prompts and N independent generations.
---
--- The tests exercise the public surface :
---
---   - `Pool:add(session, opts)` registers a session, prefills it
---     through the cm, samples its first token. Slots can be added
---     before OR after `run` — both paths must end up consistent.
---
---   - `Pool:run()` drives every slot to a stop condition and writes
---     a `Response` onto each session.
---
---   - The slot list reports `n_active` accurately as sessions
---     terminate one by one.
---
---   - Distinct sessions get distinct seq_ids and distinct content.
---
---   - The streaming `on_chunk` callback receives typed chunks, ending
---     with a `stop` chunk per slot.

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm = require "ion7.llm"

-- The pool needs more than one seq slot ; the helper auto-enables
-- kv_unified when n_seq_max > 1.
local _, ctx, vocab, cm, _ = H.pipeline(model, {
    n_ctx     = 4096,
    n_seq_max = 4,
    headroom  = 256,
})

cm:set_system("You are a concise assistant.")

-- ════════════════════════════════════════════════════════════════════════
-- Pool — basic two-session run
-- ════════════════════════════════════════════════════════════════════════

T.suite("Pool — runs two concurrent sessions to completion")

T.test("each session receives a Response after run()", function()
    local pool = llm.Pool.new(ctx, vocab, cm)

    local s1 = llm.Session.new() ; s1:add_user("Reply with: red.")
    local s2 = llm.Session.new() ; s2:add_user("Reply with: blue.")

    pool:add(s1, { max_tokens = 8 })
    pool:add(s2, { max_tokens = 8 })

    pool:run()

    T.is_type(s1:last_response(),         "table")
    T.is_type(s2:last_response(),         "table")
    T.gt(s1:last_response().n_tokens,     0)
    T.gt(s2:last_response().n_tokens,     0)

    cm:release(s1) ; cm:release(s2)
    pool:free()
end)

T.test("two sessions land on different seq_ids", function()
    local pool = llm.Pool.new(ctx, vocab, cm)

    local s1 = llm.Session.new() ; s1:add_user("Reply: A")
    local s2 = llm.Session.new() ; s2:add_user("Reply: B")
    local s3 = llm.Session.new() ; s3:add_user("Reply: C")

    pool:add(s1, { max_tokens = 4 })
    pool:add(s2, { max_tokens = 4 })
    pool:add(s3, { max_tokens = 4 })

    pool:run()

    -- All seq_ids must be distinct.
    T.neq(s1.seq_id, s2.seq_id)
    T.neq(s2.seq_id, s3.seq_id)
    T.neq(s1.seq_id, s3.seq_id)

    cm:release(s1) ; cm:release(s2) ; cm:release(s3)
    pool:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Pool — n_active drains as sessions terminate
-- ════════════════════════════════════════════════════════════════════════

T.suite("Pool — n_active counter")

T.test("n_active equals the number of registered slots before tick", function()
    local pool = llm.Pool.new(ctx, vocab, cm)
    local s1 = llm.Session.new() ; s1:add_user("Reply: x")
    local s2 = llm.Session.new() ; s2:add_user("Reply: y")
    pool:add(s1, { max_tokens = 4 })
    pool:add(s2, { max_tokens = 4 })

    T.eq(pool:n_active(), 2)

    pool:run()
    T.eq(pool:n_active(), 0,
        "every slot must terminate after run()")

    cm:release(s1) ; cm:release(s2)
    pool:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Pool — streaming chunks per slot
-- ════════════════════════════════════════════════════════════════════════

T.suite("Pool — on_chunk callback fires typed chunks per slot")

T.test("each slot receives content + a final stop chunk", function()
    local pool = llm.Pool.new(ctx, vocab, cm)

    local s1 = llm.Session.new() ; s1:add_user("Reply with: ok.")
    local s2 = llm.Session.new() ; s2:add_user("Reply with: yes.")

    local chunks_per_slot = { [s1] = {}, [s2] = {} }
    local on_chunk = function(slot, chunk)
        chunks_per_slot[slot.session][#chunks_per_slot[slot.session] + 1] = chunk
    end

    pool:add(s1, { max_tokens = 8, on_chunk = on_chunk })
    pool:add(s2, { max_tokens = 8, on_chunk = on_chunk })

    pool:run()

    for _, sess in ipairs({ s1, s2 }) do
        local cs = chunks_per_slot[sess]
        T.gt(#cs, 0, "at least one chunk per slot")
        T.eq(cs[#cs].kind, "stop", "last chunk must be stop")
    end

    cm:release(s1) ; cm:release(s2)
    pool:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Pool — capacity guard
-- ════════════════════════════════════════════════════════════════════════

T.suite("Pool — batch capacity")

T.test("Pool.new clamps to ctx:n_seq_max() by default", function()
    local pool = llm.Pool.new(ctx, vocab, cm)
    T.eq(pool._batch_cap, ctx:n_seq_max())
    pool:free()
end)

ctx:free()
local ok = T.summary()
os.exit(ok and 0 or 1)
