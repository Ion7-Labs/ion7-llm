#!/usr/bin/env luajit
--- @module tests.14_radix
--- @author  ion7 / Ion7 Project Contributors
---
--- RadixAttention prefix cache : `kv.radix` + `cm:prepare` integration.
---
--- The cache is OFF by default to avoid surprising memory growth ;
--- this file flips it on (`opts.radix = true`) and verifies :
---
---   - A fresh `find_longest_prefix` against an empty cache misses.
---   - After one chat through `cm:prepare`, the cache holds at least
---     one blob (the just-prefilled prompt).
---   - A second session whose prompt SHARES the head of the first
---     session's prompt warm-starts from the cached blob ; the
---     `radix.hits` counter increments, the second prepare's decode
---     work is reduced (n_past advances less from base than the first
---     prepare did).
---   - `set_system` clears the cache so a different prior cannot
---     leak its blobs into a new system context.
---   - `min_match` filters out pointlessly-tiny matches.

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm = require "ion7.llm"

-- Pure-Lua cache tests first (no model involvement).
T.suite("kv.radix — pure-Lua cache semantics")

T.test("a fresh cache misses every lookup", function()
    local r = llm.kv.radix.new({ min_match = 4 })
    local n, blob = r:find_longest_prefix({ 1, 2, 3, 4 })
    T.eq(n,    0)
    T.eq(blob, nil)
    T.eq(r:stats().misses, 1)
end)

T.test("insert + find_longest_prefix on the same input hits", function()
    local r = llm.kv.radix.new({ min_match = 1 })
    r:insert({ 10, 20, 30, 40 }, 4, "BLOB_A")
    local n, blob = r:find_longest_prefix({ 10, 20, 30, 40 })
    T.eq(n,    4)
    T.eq(blob, "BLOB_A")
    T.eq(r:stats().hits, 1)
end)

T.test("a partial divergence misses (no block-level sharing yet)", function()
    local r = llm.kv.radix.new({ min_match = 1 })
    r:insert({ 1, 2, 3, 4 }, 4, "BLOB_FULL")
    -- The input shares a 3-token prefix but diverges at position 4.
    -- Restoring BLOB_FULL would put tokens 4 = 4 in the KV that the
    -- new prompt does not have ; we report a miss instead. Block-
    -- level KV sharing (vLLM-style) would let us partial-match here ;
    -- it is a v3 project.
    local n, blob = r:find_longest_prefix({ 1, 2, 3, 9 })
    T.eq(n,    0)
    T.eq(blob, nil)
end)

T.test("an exact match below min_match is rejected", function()
    local r = llm.kv.radix.new({ min_match = 16 })
    r:insert({ 1, 2 }, 2, "BLOB_SHORT")
    local n = r:find_longest_prefix({ 1, 2 })
    T.eq(n, 0, "matched but too short for min_match")
end)

T.test("a strict prefix of a longer cached path also misses", function()
    -- We have a blob at length 4 ; querying with the first 2 tokens
    -- alone would put the kv in a state of length 4 — incorrect.
    local r = llm.kv.radix.new({ min_match = 1 })
    r:insert({ 1, 2, 3, 4 }, 4, "BLOB_FULL")
    local n = r:find_longest_prefix({ 1, 2 })
    T.eq(n, 0)
end)

T.test("an exact-match input restores the blob", function()
    local r = llm.kv.radix.new({ min_match = 1 })
    r:insert({ 7, 8, 9 }, 3, "BLOB")
    local n, blob = r:find_longest_prefix({ 7, 8, 9 })
    T.eq(n,    3)
    T.eq(blob, "BLOB")
end)

T.test("min_match suppresses tiny matches", function()
    local r = llm.kv.radix.new({ min_match = 16 })
    r:insert({ 1, 2 }, 2, "BLOB_SHORT")
    local n, blob = r:find_longest_prefix({ 1, 2, 3, 4 })
    T.eq(n,    0, "match is below min_match")
    T.eq(blob, nil)
end)

T.test("LRU eviction kicks in past max_blobs", function()
    local r = llm.kv.radix.new({ min_match = 1, max_blobs = 2 })
    r:insert({ 100 },  1, "A")
    r:insert({ 200 },  1, "B")
    r:insert({ 300 },  1, "C")    -- evicts the oldest (A)
    T.eq(r:stats().n_blobs,       2)
    T.eq(r:stats().blobs_evicted, 1)
    -- A is gone.
    local n, _ = r:find_longest_prefix({ 100 })
    T.eq(n, 0)
    -- B + C are still there.
    n = r:find_longest_prefix({ 200 }) ; T.eq(n, 1)
    n = r:find_longest_prefix({ 300 }) ; T.eq(n, 1)
end)

T.test("clear() wipes everything", function()
    local r = llm.kv.radix.new({ min_match = 1 })
    r:insert({ 1, 2 }, 2, "X")
    r:clear()
    T.eq(r:stats().n_blobs, 0)
    local n = r:find_longest_prefix({ 1, 2 })
    T.eq(n, 0)
end)

-- Integration : ContextManager + radix.
T.suite("ContextManager — radix cache integration")

T.test("two sessions running the SAME prompt — second one hits", function()
    -- Single-seq context so the radix sees the WHOLE rendered prompt
    -- (system included). Multi-seq mode would need the prefix tokens
    -- to come through the cache too — handled by `kv.Prefix`, not by
    -- the radix.
    local ctx   = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
    local vocab = model:vocab()
    local cm = llm.kv.new(ctx, vocab, {
        headroom = 256,
        radix    = { min_match = 8, max_blobs = 16 },
    })
    local engine = llm.Engine.new(ctx, vocab, cm)
    cm:set_system("You are a concise tutor.")

    -- Two sessions, same exact prompt. The second one SHOULD warm-
    -- start from the first's blob.
    local prompt = "Reply with the single word: ready."

    local s1 = llm.Session.new()
    s1:add_user(prompt)
    engine:chat(s1, { max_tokens = 4 })
    T.gt(cm:stats().radix.n_blobs, 0,
        "first prepare should have inserted at least one blob")
    cm:release(s1)

    local s2 = llm.Session.new()
    s2:add_user(prompt)
    engine:chat(s2, { max_tokens = 4 })
    T.gt(cm:stats().radix.hits, 0,
        "the second prepare should have hit the radix cache")

    cm:release(s2)
    ctx:free()
end)

T.test("set_system clears the radix when the prompt changes", function()
    local ctx   = model:context({ n_ctx = 2048, n_seq_max = 1, n_threads = 4 })
    local vocab = model:vocab()
    local cm = llm.kv.new(ctx, vocab, {
        headroom = 128,
        radix    = { min_match = 8 },
    })
    local engine = llm.Engine.new(ctx, vocab, cm)
    cm:set_system("System prompt A.")

    local s = llm.Session.new()
    s:add_user("hi")
    engine:chat(s, { max_tokens = 4 })
    T.gt(cm:stats().radix.n_blobs, 0)

    cm:release(s)

    -- Switching system text invalidates the cache.
    cm:set_system("System prompt B.")
    T.eq(cm:stats().radix.n_blobs, 0,
        "set_system() should clear the radix cache")
    ctx:free()
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
