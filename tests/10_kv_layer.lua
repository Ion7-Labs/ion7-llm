#!/usr/bin/env luajit
--- @module tests.10_kv_layer
--- @author  ion7 / Ion7 Project Contributors
---
--- KV layer coverage : `Slots`, `Prefix`, `snapshot`, `ContextManager`.
---
--- These four modules are what makes ion7-llm correct in the
--- multi-session case — the previous codebase tracked one global
--- `n_past` per context, which silently corrupted state every time
--- two sessions interleaved decodes. The new layer carries `seq_id`
--- + per-seq snapshots, so a call against session A cannot disturb
--- session B's KV row.
---
--- The tests below load a real model, build a context with
--- `n_seq_max = 4`, and verify each invariant the higher layers (the
--- engine, the pool) take for granted :
---
---   - Slots refuses to allocate more rows than the context has.
---   - Releasing a row makes it reclaimable.
---   - The reserved-slot mode (used by Prefix) keeps user sessions
---     out of seq 0.
---   - Prefix encodes once, then `kv_seq_cp` to user sessions.
---   - ContextManager:prepare goes through the slow path on first
---     call, the fast (snapshot) path on the second.
---   - Forking a session through cm:fork copies KV without re-encoding.

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm = require "ion7.llm"

-- ════════════════════════════════════════════════════════════════════════
-- Slots — pure pool semantics over the context's n_seq_max
-- ════════════════════════════════════════════════════════════════════════

T.suite("Slots — capacity + acquire / release")

T.test("acquire fills up to n_seq_max then returns nil", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local slots = llm.kv.Slots.new(ctx)
    T.eq(slots:capacity(), 4)
    T.eq(slots:n_free(),   4)

    local ids = {}
    for _ = 1, 4 do ids[#ids + 1] = slots:acquire() end
    T.eq(#ids, 4)
    T.eq(slots:n_free(), 0)
    T.eq(slots:acquire(), nil, "5th acquire should return nil")

    -- Releasing one frees a row.
    slots:release(ids[2])
    T.eq(slots:n_free(), 1)
    local id = slots:acquire()
    T.neq(id, nil)
    ctx:free()
end)

T.test("a reserved slot is never handed out by acquire", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local slots = llm.kv.Slots.new(ctx, { reserved = 0 })
    T.eq(slots:capacity(),   3, "capacity excludes the reserved slot")
    T.eq(slots:n_free(),     3, "3 user-facing slots when seq 0 is reserved")
    T.eq(slots:is_reserved(0), true)

    for _ = 1, 3 do
        local id = slots:acquire()
        T.neq(id, 0, "acquire must skip the reserved slot")
    end
    T.eq(slots:acquire(), nil, "fourth acquire must fail — the reserved slot is off-limits")
    ctx:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Prefix — system-prompt prefill cached in seq 0
-- ════════════════════════════════════════════════════════════════════════

T.suite("Prefix — encode once, copy to session rows")

T.test("multi-seq context owns a real prefix slot", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()
    local prefix = llm.kv.Prefix.new(ctx, vocab)

    T.eq(prefix:has_slot(),  true)
    T.eq(prefix:seq_id(),    0)
    T.eq(prefix:n_tokens(),  0, "no system text yet")
    ctx:free()
end)

T.test("set encodes the system text into the reserved slot", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()
    local prefix = llm.kv.Prefix.new(ctx, vocab)

    local n = prefix:set("You are a concise assistant.")
    T.gt(n, 0, "tokenisation should produce at least one token")
    T.eq(prefix:n_tokens(), n)
end)

T.test("copy_to duplicates the prefix into a user row", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()
    local prefix = llm.kv.Prefix.new(ctx, vocab)
    prefix:set("System prompt here.")
    local copied = prefix:copy_to(2)
    T.gt(copied, 0)
    T.eq(copied, prefix:n_tokens())
end)

T.test("single-seq context degrades gracefully — text-only mode", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 1, n_threads = 2 })
    local vocab = model:vocab()
    local prefix = llm.kv.Prefix.new(ctx, vocab)
    T.eq(prefix:has_slot(), false, "no spare slot for the prefix")
    -- set still records the text (returned n is 0 because nothing was encoded
    -- into a dedicated slot).
    local n = prefix:set("System prompt.")
    T.eq(n, 0)
    T.eq(prefix:text(), "System prompt.")
end)

-- ════════════════════════════════════════════════════════════════════════
-- snapshot — per-seq KV state I/O
-- ════════════════════════════════════════════════════════════════════════

T.suite("snapshot — save / restore per-seq blob")

T.test("save returns a non-empty blob for a populated seq", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()

    local toks, n = vocab:tokenize("Hello, world!", true, true)
    ctx:decode(toks, n, 1, 0)

    local blob = llm.kv.snapshot.save(ctx, 1)
    T.is_type(blob, "string")
    T.gt(#blob, 0, "snapshot of a non-empty seq should be non-empty")
    ctx:free()
end)

T.test("restore brings a wiped seq back to its captured state", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()

    local toks, n = vocab:tokenize("This is a test prompt.", true, true)
    ctx:decode(toks, n, 1, 0)
    local blob = llm.kv.snapshot.save(ctx, 1)

    -- Wipe the row entirely.
    ctx:kv_seq_rm(1, 0, -1)

    -- Restore — the blob carries the position + KV layout.
    local restored = llm.kv.snapshot.restore(ctx, blob, 1)
    T.gt(restored, 0, "restore should consume bytes from the blob")
    ctx:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- ContextManager — slot allocation, slow / fast prepare path
-- ════════════════════════════════════════════════════════════════════════

T.suite("ContextManager — slot allocation + prepare")

T.test("prepare allocates a seq_id on first call", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()
    local cm = llm.kv.new(ctx, vocab, { headroom = 64 })
    cm:set_system("Be concise.")

    local s = llm.Session.new()
    s:add_user("Hi.")
    cm:prepare(s)

    T.neq(s.seq_id, nil, "prepare must assign a seq_id on first call")
    T.gt(s.n_past,  0,   "prepare must advance n_past after the prefill")
    T.eq(s._dirty,  false, "prepare clears the dirty flag after a successful encode")
    ctx:free()
end)

T.test("a clean prepare on the same session takes the fast path", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()
    local cm = llm.kv.new(ctx, vocab)
    cm:set_system("Stay brief.")

    local s = llm.Session.new()
    s:add_user("test")
    cm:prepare(s)
    local n1 = s.n_past

    -- Without mutating the session, prepare should be a snapshot-restore
    -- that returns the same n_past and does NOT toggle the dirty flag.
    cm:prepare(s)
    T.eq(s.n_past, n1, "fast path must keep the same n_past")
    T.eq(s._dirty, false)
    ctx:free()
end)

T.test("release returns the seq_id to the pool", function()
    local ctx = model:context({ n_ctx = 1024, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()
    local cm = llm.kv.new(ctx, vocab)

    local s = llm.Session.new()
    s:add_user("hi")
    cm:prepare(s)

    local before = cm:stats().slots_free
    cm:release(s)
    T.eq(cm:stats().slots_free, before + 1, "release should give back one slot")
    T.eq(s.seq_id, nil)
    T.eq(s.n_past, 0)
    ctx:free()
end)

T.test("two sessions on the same context get distinct seq_ids", function()
    local ctx = model:context({ n_ctx = 2048, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()
    local cm = llm.kv.new(ctx, vocab)
    cm:set_system("Be brief.")

    local a = llm.Session.new() ; a:add_user("one")
    local b = llm.Session.new() ; b:add_user("two")
    cm:prepare(a)
    cm:prepare(b)

    T.neq(a.seq_id, b.seq_id, "sessions must occupy different KV rows")
    ctx:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- ContextManager:fork — KV duplication into a fresh slot
-- ════════════════════════════════════════════════════════════════════════

T.suite("ContextManager — fork copies KV without re-encoding")

T.test("fork lands on a different seq_id and inherits the history", function()
    local ctx = model:context({ n_ctx = 2048, n_seq_max = 4, n_threads = 2, kv_unified = true })
    local vocab = model:vocab()
    local cm = llm.kv.new(ctx, vocab)

    local parent = llm.Session.new()
    parent:add_user("hello")
    cm:prepare(parent)

    local child = cm:fork(parent)
    T.neq(child.seq_id,        parent.seq_id, "child must own a separate slot")
    T.eq(#child.messages,      #parent.messages)
    T.eq(child.messages[1].content, "hello")
    T.eq(child.n_past,         parent.n_past, "fork preserves n_past")
    T.is_type(child._seq_snapshot, "string", "fork re-snapshots the child's KV")
    ctx:free()
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
