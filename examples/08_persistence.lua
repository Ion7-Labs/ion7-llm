#!/usr/bin/env luajit
--- @example examples.08_persistence
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 08 — Persistence : Session JSON + per-seq KV save / load
--- ════════════════════════════════════════════════════════════════════════
---
--- Two persistence layers, each with its own scope :
---
---   1. `Session:save / Session.load`      — JSON, cross-process, cross-host.
---      Captures the conversation history (messages + system). Does
---      NOT include the KV blob : the next chat() call re-prefills
---      from the saved messages, since the blob is only meaningful
---      against the same context shape (n_ctx, n_layer, kv quant…).
---
---   2. `kv.snapshot.save_file / load_file` — binary, ONE seq.
---      Captures a single sequence's KV state. Useful when the same
---      process keeps its context alive between calls (e.g. a
---      long-running server) and wants to swap out a session's row
---      without re-decoding. Not portable — different model / quant
---      / context shape will reject the blob.
---
--- The example below shows both : run a turn, save the JSON history,
--- save the KV blob, "shut down", reload both, run a follow-up turn.
---
---   ION7_MODEL=/path/to/chat.gguf luajit examples/08_persistence.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
local llm  = require "ion7.llm"

ion7.init({ log_level = 0 })

local model = ion7.Model.load(MODEL, {
    n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS")) or 0,
})
local ctx   = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
local vocab = model:vocab()
local cm, engine = llm.pipeline(ctx, vocab, { headroom = 256 })

cm:set_system("You are a concise assistant.")

local TMPDIR     = os.getenv("TMPDIR") or "/tmp"
local JSON_PATH  = TMPDIR .. "/ion7-llm-example-session.json"
local KV_PATH    = TMPDIR .. "/ion7-llm-example-session.kv"

-- ── Phase 1 : run a turn, persist both the history and the KV ───────────

print("[phase 1] running first turn")
local s1 = llm.Session.new()
s1:add_user("My name is Ada. Reply with: noted.")
local r1 = engine:chat(s1, { max_tokens = 8 })
s1:add_assistant(r1.content, { thinking = r1.thinking })

print("  - assistant : " .. r1.content)
print("  - n_past    : " .. s1.n_past)

-- JSON history.
assert(s1:save(JSON_PATH), "save() failed")
print("  - history JSON saved to " .. JSON_PATH)

-- KV blob (binary).
assert(llm.kv.snapshot.save_file(ctx, KV_PATH, s1.seq_id),
    "kv.snapshot.save_file() failed")
print("  - KV blob   saved to " .. KV_PATH)

-- Forget the session as far as the cm is concerned.
cm:release(s1)

-- ── Phase 2 : reload the JSON, re-prefill from messages ─────────────────

print("\n[phase 2] reloading from JSON (cold path : re-prefills from history)")
local s2 = assert(llm.Session.load(JSON_PATH))
s2:add_user("What name did I just give you ? Answer in one word.")
local r2 = engine:chat(s2, { max_tokens = 6 })
print("  - assistant : " .. r2.content)
cm:release(s2)

-- ── Phase 3 : reload the KV blob into a fresh seq, no re-decode ─────────

print("\n[phase 3] reloading the KV blob (warm path : no re-prefill)")
local s3 = llm.Session.new({ system = "You are a concise assistant." })
-- Re-create the message history from the JSON copy so the next
-- engine:chat sees a coherent state once the KV is restored.
local saved_session = assert(llm.Session.load(JSON_PATH))
for _, m in ipairs(saved_session.messages) do
    s3.messages[#s3.messages + 1] = m
end
-- Allocate the slot first so we know which seq to load into.
cm:prepare(s3)
-- Drop the prepared row, replace with the saved blob.
ctx:kv_seq_rm(s3.seq_id, 0, -1)
assert(llm.kv.snapshot.load_file(ctx, KV_PATH, s3.seq_id),
    "kv.snapshot.load_file() failed")
-- Tell the session bookkeeping to trust the loaded row : n_past from
-- the original session, snapshot blob set, dirty cleared.
s3.n_past        = saved_session.n_past or s3.n_past
s3._seq_snapshot = nil   -- force the next prepare to re-snapshot

s3:add_user("Confirm in one word that you remember the name.")
local r3 = engine:chat(s3, { max_tokens = 6 })
print("  - assistant : " .. r3.content)
cm:release(s3)

-- ── Cleanup ──────────────────────────────────────────────────────────────

os.remove(JSON_PATH)
os.remove(KV_PATH)

ctx:free()
model:free()
ion7.shutdown()
