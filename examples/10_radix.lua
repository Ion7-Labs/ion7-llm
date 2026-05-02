#!/usr/bin/env luajit
--- @example examples.10_radix
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 10 — RadixAttention prefix cache : warm-start a fresh session from
---       a sibling's KV blob
--- ════════════════════════════════════════════════════════════════════════
---
--- The radix cache (in `kv.radix`) indexes per-seq snapshots by their
--- token sequence. The simple-variant cache we ship in v2 is an
--- EXACT-MATCH cache : it warm-starts a session whose prompt is
--- byte-identical to one already cached, but does NOT (yet) handle
--- partial-match warm-starts (block-level KV sharing, vLLM-style, is
--- a v3 project — restoring a blob whose tail diverges from the live
--- prompt would put unseen tokens in the KV).
---
--- The exact-match cache covers the high-value cases on its own :
---
---   - **Re-issue** : agent loops re-running the same prompt after a
---     reject, A/B testing, regeneration with fresh sampling.
---   - **Forked sessions** rejoining a parent's exact state.
---   - **Idempotent agent steps** that re-render the same context.
---
--- Below we run two prompts twice : the second occurrence of each
--- prompt warm-starts from the cache. A third prompt without a
--- match misses, as expected.
---
---   ION7_MODEL=/path/to/chat.gguf luajit examples/10_radix.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
local llm  = require "ion7.llm"

ion7.init({ log_level = 0 })

local model = ion7.Model.load(MODEL, {
    n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS")) or 0,
})

-- Single-seq context : the radix sees the WHOLE rendered prompt
-- (system + messages) since the prefix cache is in degraded mode.
local ctx   = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
local vocab = model:vocab()
local cm = llm.kv.new(ctx, vocab, {
    headroom = 256,
    -- Enabling the radix is one option. Defaults are fine for casual
    -- workloads ; tweak `min_match` / `max_blobs` for serving setups.
    radix    = { min_match = 32, max_blobs = 16, chunk_size = 32 },
})
local engine = llm.Engine.new(ctx, vocab, cm)

cm:set_system("You are a concise tutor. Reply in one sentence.")

local prompts = {
    "What is the determinant of a 2x2 matrix ?",
    "What is the determinant of a 2x2 matrix ?",  -- exact re-issue → hit
    "Explain photosynthesis in one sentence.",
    "Explain photosynthesis in one sentence.",     -- exact re-issue → hit
    "Define a Banach space in one sentence.",      -- new prompt → miss
}

for i, p in ipairs(prompts) do
    local s = llm.Session.new()
    s:add_user(p)

    local before = cm:stats().radix
    local r = engine:chat(s, { max_tokens = 32 })
    local after  = cm:stats().radix

    local hit = (after.hits or 0) > (before.hits or 0)
    print(string.format("\n[prompt %d] %s", i, p))
    print(string.format("  cache : %s — %d blobs / %d max ; n_past after prepare = %d",
        hit and "HIT" or "miss",
        after.n_blobs, after.max_blobs, s.n_past - r.n_tokens))
    print("  reply : " .. r.content)

    cm:release(s)
end

-- The cache survives across release() calls. The blobs only go away
-- on `set_system` change or on explicit `cm:clear_radix()`.
print("\nfinal stats :")
local s = cm:stats().radix
print(string.format("  hits=%d  misses=%d  blobs=%d  blobs_evicted=%d",
    s.hits, s.misses, s.n_blobs, s.blobs_evicted))

ctx:free()
model:free()
ion7.shutdown()
