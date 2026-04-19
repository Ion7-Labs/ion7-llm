#!/usr/bin/env luajit
--- 08_adaptive_speculation.lua - Speculative decoding: n-gram, EAGLE3, entropy-adaptive.
---
--- Demonstrates two speculative decoding directions, both opt-in:
---
---   Direction 1 — Entropy-adaptive speculation (ngram_cache + entropy gate)
---     ctx:entropy(0) is measured after each token sample.
---     If entropy < threshold → model is confident → draft and verify.
---     If entropy > threshold → model is uncertain → skip drafting (avoid wasted
---     batch-decode overhead when drafts would be rejected anyway).
---     CRANE-style: speculation effort proportional to model confidence.
---
---   Direction 2 — EAGLE3 (draft heads on the target model)
---     Uses EAGLE3 speculative heads baked into the model's GGUF metadata.
---     No separate draft model required. ~2-3x speedup on compatible models.
---     Setup via engine:setup_speculative({ type = "eagle3" }).
---
---   Direction 3 — Draft model (small model as draft oracle)
---     A small model (e.g. 1.7B) predicts continuations for a larger target.
---     Configured via setup_speculative({ type = "draft", draft_model = ... }).
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/08_adaptive_speculation.lua
---
--- EAGLE3 (requires compatible model):
---   EAGLE3=1 ION7_MODEL=/path/to/eagle3-capable.gguf luajit examples/08_adaptive_speculation.lua
---
--- Draft model:
---   DRAFT_MODEL=/path/to/small.gguf ION7_MODEL=/path/to/large.gguf luajit examples/08_adaptive_speculation.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL       = os.getenv("ION7_MODEL")
local DRAFT_MODEL = os.getenv("DRAFT_MODEL")
local USE_EAGLE3  = os.getenv("EAGLE3") == "1"

if not MODEL then
    io.stderr:write(
        "Usage: ION7_MODEL=/path/to/model.gguf luajit examples/08_adaptive_speculation.lua\n")
    os.exit(1)
end

local llm = require "ion7.llm"

-- ── Shared prompts for timing comparison ──────────────────────────────────────

local PROMPTS = {
    "List the first 20 Fibonacci numbers.",
    "Write a Lua function that merges two sorted arrays.",
    "Explain the difference between TCP and UDP in two paragraphs.",
    "What are the main advantages of LuaJIT over standard Lua?",
}

local function time_ms()
    -- Relies on os.clock() — wall clock precision varies; good enough for demo.
    return os.clock() * 1000
end

local function run_prompts(engine, label, session_fn)
    io.write(string.format("\n── %s ──\n", label))
    local total_tok = 0
    local total_ms  = 0

    for _, prompt in ipairs(PROMPTS) do
        local s = session_fn(engine, prompt)
        local t0 = time_ms()
        local n  = 0
        local resp = engine:chat(s, {
            max_tokens = 128,
            on_token   = function() n = n + 1 end,
        })
        local elapsed = time_ms() - t0
        total_tok = total_tok + n
        total_ms  = total_ms  + elapsed
        io.write(string.format("  %3d tok  %6.0f ms  %.1f tok/s  │ %s...\n",
            n, elapsed,
            elapsed > 0 and (n / (elapsed / 1000)) or 0,
            resp.text:sub(1, 50):gsub("\n", " ")))
    end

    local avg_tps = total_ms > 0 and (total_tok / (total_ms / 1000)) or 0
    io.write(string.format("  total: %d tok, %.0f ms, %.1f tok/s avg\n",
        total_tok, total_ms, avg_tps))
    return avg_tps
end

local function make_session(engine, prompt)
    local s = engine:session()
    s:add("user", prompt)
    return s
end

-- ── Build engine ──────────────────────────────────────────────────────────────
io.write(string.format("[model] %s\n", MODEL:match("[^/]+$")))
io.write("[init] loading model (no speculation yet)...\n")

local engine = llm.new({
    model  = MODEL,
    system = "You are a concise assistant. Answer in 2–3 sentences max.",
    think  = false,
})

-- ─────────────────────────────────────────────────────────────────────────────
-- Baseline: no speculation
-- ─────────────────────────────────────────────────────────────────────────────
local tps_baseline = run_prompts(engine, "Baseline (no speculation)", make_session)

-- ─────────────────────────────────────────────────────────────────────────────
-- Direction 1a: ngram_cache — always speculate (no entropy gate)
-- ─────────────────────────────────────────────────────────────────────────────
io.write("\n[setup] ngram_cache — always speculate\n")
engine:setup_speculative({
    type    = "ngram_cache",
    n_draft = 6,
})

local tps_ngram = run_prompts(engine, "ngram_cache (always)", make_session)

-- ─────────────────────────────────────────────────────────────────────────────
-- Direction 1b: ngram_cache + entropy-adaptive gate
--
-- The entropy gate reads ctx:entropy(0) after each sample. When the model is
-- uncertain (entropy > threshold), speculation is skipped for that step.
-- This avoids wasting a batch-decode round when the model would reject drafts.
--
-- engine:set_spec_entropy_threshold(1.0) means:
--   entropy < 1.0 nats → draft                (model picking confidently)
--   entropy ≥ 1.0 nats → standard decode       (model uncertain, skip)
--
-- For context: a perfectly uniform distribution over 151k tokens = ~11.9 nats.
-- Most well-predicted tokens sit around 0.1–0.8 nats.
-- ─────────────────────────────────────────────────────────────────────────────
io.write("\n[setup] ngram_cache + entropy_threshold=1.0 nats\n")
io.write("        ctx:entropy(0) checked per step — skips drafting when model uncertain\n")
engine:set_spec_entropy_threshold(1.0)

local tps_adaptive = run_prompts(engine, "ngram_cache + entropy gate (1.0 nats)", make_session)

io.write(string.format("\n[entropy gate] spec acceptance likely higher than always-on\n"))
engine:spec_stats()

-- ─────────────────────────────────────────────────────────────────────────────
-- Direction 2: EAGLE3 (requires compatible model)
-- ─────────────────────────────────────────────────────────────────────────────
local tps_eagle3 = nil

if USE_EAGLE3 then
    io.write("\n[setup] EAGLE3 — draft heads on target model (no separate draft model)\n")
    io.write("        Requires GGUF with EAGLE3 metadata.\n")
    engine:setup_speculative({
        type    = "eagle3",
        n_draft = 6,
        -- Note: no entropy_threshold — EAGLE3 is more accurate, gate less needed.
    })
    tps_eagle3 = run_prompts(engine, "EAGLE3", make_session)
    engine:spec_stats()
else
    io.write("\n[skipped] EAGLE3 — set EAGLE3=1 to enable (requires compatible model)\n")
end

-- ─────────────────────────────────────────────────────────────────────────────
-- Direction 3: Draft model (opt-in, requires DRAFT_MODEL env var)
-- ─────────────────────────────────────────────────────────────────────────────
local tps_draft = nil

if DRAFT_MODEL then
    io.write(string.format("\n[setup] draft model — %s\n", DRAFT_MODEL:match("[^/]+$")))
    io.write("        small model generates candidates, large model verifies.\n")
    engine:setup_speculative({
        type         = "draft",
        draft_model  = DRAFT_MODEL,
        draft_ngl    = 0,    -- CPU for draft; target stays on GPU
        n_draft      = 6,
        -- Entropy gate is optional with draft model:
        entropy_threshold = 1.5,
    })
    tps_draft = run_prompts(engine, "Draft model + entropy gate (1.5 nats)", make_session)
    engine:spec_stats()
else
    io.write("\n[skipped] Draft model — set DRAFT_MODEL=/path/small.gguf to enable\n")
end

-- ─────────────────────────────────────────────────────────────────────────────
-- Summary
-- ─────────────────────────────────────────────────────────────────────────────
io.write("\n" .. string.rep("─", 60) .. "\n[summary]\n")
io.write(string.format("  Baseline           : %5.1f tok/s\n", tps_baseline))
io.write(string.format("  ngram_cache        : %5.1f tok/s  (%.1fx)\n",
    tps_ngram, tps_ngram / tps_baseline))
io.write(string.format("  + entropy gate 1.0 : %5.1f tok/s  (%.1fx)\n",
    tps_adaptive, tps_adaptive / tps_baseline))
if tps_eagle3 then
    io.write(string.format("  EAGLE3             : %5.1f tok/s  (%.1fx)\n",
        tps_eagle3, tps_eagle3 / tps_baseline))
end
if tps_draft then
    io.write(string.format("  Draft model        : %5.1f tok/s  (%.1fx)\n",
        tps_draft, tps_draft / tps_baseline))
end

io.write([[

[notes]
  ngram_cache     — zero overhead, warms up from generated history. Best default.
  entropy gate    — reduces wasted batch-decode at uncertain positions. Tune the
                    threshold (nats) for your model and content type:
                      0.5 = only draft when very confident (few skips)
                      1.0 = balanced (good default)
                      2.0 = draft unless nearly random
  EAGLE3          — draft heads on target model, ~2-3x speedup. Requires model
                    built with EAGLE3 support (check GGUF metadata).
  Draft model     — separate small model as oracle. Best speedup potential but
                    requires a compatible pair (same tokenizer, same vocab).

  All modes are opt-in — llm.new() with no speculative opts = standard decoding.
  Use engine:setup_speculative() or engine:set_spec_entropy_threshold() at any
  point after init, without reloading the model.
]])

engine:shutdown()
