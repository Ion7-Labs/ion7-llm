#!/usr/bin/env luajit
--- @example examples.04_pool
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 04 — Multi-session pool : N concurrent conversations, ONE decode
---       per tick
--- ════════════════════════════════════════════════════════════════════════
---
--- `Pool` packs one batch row per active session into a single
--- `llama_decode` call. Throughput-per-clock goes up considerably for
--- small N — the GPU sees a wider matmul, the CPU prefill amortises
--- across rows. The trade-off : every session shares the same KV
--- cache, so total `n_ctx_seq` tightens proportionally.
---
--- This example runs three independent sessions side by side. Each
--- one streams its tokens through an `on_chunk` callback that prefixes
--- output with the slot index, so the interleaved per-tick yield is
--- visible.
---
---   ION7_MODEL=/path/to/chat.gguf luajit examples/04_pool.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
local llm  = require "ion7.llm"

ion7.init({ log_level = 0 })

local model = ion7.Model.load(MODEL, {
    n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS")) or 0,
})

-- Multi-session contexts need `kv_unified = true` (so kv_seq_cp is
-- legal) AND `n_seq_max > 1` (so several rows can coexist).
local ctx   = model:context({
    n_ctx      = 4096,
    n_seq_max  = 4,
    kv_unified = true,
    n_threads  = 4,
})
local vocab = model:vocab()
local cm, engine = llm.pipeline(ctx, vocab, { headroom = 128 })

cm:set_system("You are a concise assistant. Reply in one short sentence.")

local pool = llm.Pool.new(ctx, vocab, cm)

-- ── Three independent sessions ────────────────────────────────────────────

local prompts = {
    "Define a stack data structure.",
    "Define a queue data structure.",
    "Define a hash map data structure.",
}

local sessions = {}
for i, prompt in ipairs(prompts) do
    sessions[i] = llm.Session.new()
    sessions[i]:add_user(prompt)
end

-- A simple per-slot streaming callback. The pool calls it for every
-- typed chunk a slot produces ; we tag the output with the slot id
-- so the interleaving is obvious in stdout.
local function on_chunk(slot, chunk)
    local _, ok = nil, true
    if chunk.kind == "content" then
        io.write(string.format("[s%d] %s\n",
            slot.session.id, chunk.text:gsub("\n", " ")))
        io.flush()
    elseif chunk.kind == "stop" then
        local r = slot.session:last_response()
        io.write(string.format("[s%d] -- done : %s\n", slot.session.id, r:summary()))
    end
end

for _, s in ipairs(sessions) do
    pool:add(s, { max_tokens = 64, on_chunk = on_chunk })
end

print(string.format("[pool] %d active slots — running...", pool:n_active()))
pool:run()

-- ── Final tally ──────────────────────────────────────────────────────────

io.write("\n=== final answers ===\n")
for i, s in ipairs(sessions) do
    local r = s:last_response()
    io.write(string.format("\n[%d] %s\n -> %s\n", i, prompts[i], r.content))
    cm:release(s)
end

pool:free()
ctx:free()
model:free()
ion7.shutdown()
