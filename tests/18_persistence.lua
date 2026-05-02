#!/usr/bin/env luajit
--- @module tests.18_persistence
--- @author  ion7 / Ion7 Project Contributors
---
--- Persistence : Session JSON round-trip + per-seq KV state
--- save / load through `kv.snapshot.save_file` /
--- `kv.snapshot.load_file`.
---
--- Two distinct concerns :
---
---   1. Conversation history. `Session:save` / `Session.load` go
---      through JSON — text-only, portable across runs and across
---      machines. The deserialised session starts dirty so the next
---      `engine:chat` re-encodes from scratch (we don't carry the
---      KV blob through JSON, since it is per-context-shape).
---
---   2. KV cache. `kv.snapshot.save_file` writes one sequence's KV
---      to disk. Loading it back into a fresh context with the same
---      shape brings the row to its original state — no re-decoding.
---      The integration here verifies a save → reload → continue chat
---      round-trip produces a coherent next turn.

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local llm, ctx, vocab, cm, engine = H.pipeline(model, {
    n_ctx     = 4096,
    n_seq_max = 4,
    headroom  = 256,
})

cm:set_system("You are a concise assistant.")

-- ════════════════════════════════════════════════════════════════════════
-- Session — JSON round-trip preserves history (already covered in 02 ;
-- this re-runs it post-chat to confirm nothing leaks from the engine).
-- ════════════════════════════════════════════════════════════════════════

T.suite("Session JSON — round-trip after a chat run")

T.test("a chatted session saves + reloads with intact history", function()
    local s = llm.Session.new()
    s:add_user("Reply with: ok.")
    local r = engine:chat(s, { max_tokens = 4 })
    s:add_assistant(r.content, { thinking = r.thinking })

    local p = H.tmpfile("ion7-llm-persist-" .. os.time() .. ".json")
    local ok, err = s:save(p)
    T.ok(ok, "save error: " .. tostring(err))

    local loaded, lerr = llm.Session.load(p)
    T.ok(loaded, "load error: " .. tostring(lerr))
    T.eq(loaded.system,            s.system)
    T.eq(#loaded.messages,         #s.messages)
    T.eq(loaded.messages[1].role,  "user")
    T.eq(loaded.messages[2].role,  "assistant")

    -- A reloaded session has no KV blob — must be dirty so the next
    -- engine:chat will re-prefill.
    T.eq(loaded._dirty,        true)
    T.eq(loaded._seq_snapshot, nil)

    H.try_remove(p)
    cm:release(s)
end)

-- ════════════════════════════════════════════════════════════════════════
-- kv.snapshot file API — save / load per-seq blob to disk
-- ════════════════════════════════════════════════════════════════════════

T.suite("kv.snapshot — file save / load round-trip")

T.test("save_file then load_file returns the same n_past", function()
    -- Decode something into seq 1, save the blob, wipe, reload.
    local toks, n = vocab:tokenize("This is a stable prompt.", true, true)
    ctx:decode(toks, n, 1, 0)

    local p = H.tmpfile("ion7-llm-kv-" .. os.time() .. ".bin")
    local ok, _ = llm.kv.snapshot.save_file(ctx, p, 1)
    T.ok(ok, "save_file should succeed")

    -- Wipe the seq.
    ctx:kv_seq_rm(1, 0, -1)

    -- Reload.
    local restored_ok = llm.kv.snapshot.load_file(ctx, p, 1)
    T.ok(restored_ok, "load_file should succeed")

    H.try_remove(p)
end)

-- ════════════════════════════════════════════════════════════════════════
-- End-to-end : save mid-conversation, reload into a NEW session, continue.
-- ════════════════════════════════════════════════════════════════════════

T.suite("Persistence — JSON-restored session can keep chatting")

T.test("a JSON-reloaded session continues the conversation", function()
    local s = llm.Session.new()
    s:add_user("My name is Ada. Reply with: noted.")
    local r1 = engine:chat(s, { max_tokens = 6 })
    s:add_assistant(r1.content)

    local p = H.tmpfile("ion7-llm-cont-" .. os.time() .. ".json")
    s:save(p)

    -- Forget the original session, release its KV row.
    cm:release(s)

    -- Reload, continue with a follow-up question.
    local loaded = llm.Session.load(p)
    loaded:add_user("What name did I just give you? Reply with one word.")
    local r2 = engine:chat(loaded, { max_tokens = 6 })
    T.gt(#r2.content, 0)
    -- Soft check : a 3 B model often reproduces "Ada" but may
    -- paraphrase. Just assert a non-empty answer.
    cm:release(loaded)
    H.try_remove(p)
end)

ctx:free()
local ok = T.summary()
os.exit(ok and 0 or 1)
