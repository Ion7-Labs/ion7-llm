#!/usr/bin/env luajit
--- 06_eviction_hook.lua - on_evict hook: summarization and external memory.
---
--- Demonstrates how to intercept context overflow to:
---   A) Compress evicted messages into a short summary (keeps context coherent)
---   B) Drop silently while logging what was lost (minimal overhead)
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/06_eviction_hook.lua
---
--- Use a small N_CTX to trigger overflow quickly:
---   N_CTX=512 ION7_MODEL=/path/to/model.gguf luajit examples/06_eviction_hook.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
local N_CTX = tonumber(os.getenv("N_CTX")) or 512

if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/06_eviction_hook.lua\n")
    os.exit(1)
end

local llm = require "ion7.llm"

-- ── Option A: summarization hook ─────────────────────────────────────────────
-- When messages are evicted, compress them into a 1-sentence summary and inject
-- it as a system message. The model keeps a compressed trace of the conversation.

local function make_summarization_hook(engine_ref)
    local evict_count = 0
    return function(evicted, _session)
        evict_count = evict_count + 1

        local text = _session and _session:format(evicted) or (function()
            local parts = {}
            for _, m in ipairs(evicted) do
                parts[#parts + 1] = m.role .. ": " .. m.content
            end
            return table.concat(parts, "\n")
        end)()

        io.write(string.format(
            "\n  [on_evict #%d] %d messages evicted — summarizing...\n",
            evict_count, #evicted))

        local ok, resp = pcall(engine_ref.complete, engine_ref,
            "Summarize the following conversation in 1 sentence:\n\n" .. text,
            { max_tokens = 60 })

        if ok and resp and resp.text ~= "" then
            io.write(string.format("  [summary] %s\n\n", resp.text))
            return { { role = "system", content = "[Earlier: " .. resp.text .. "]" } }
        end

        io.write("  [on_evict] summarization failed — dropping silently\n\n")
        return nil
    end
end

-- ── Option B: silent drop with logging ───────────────────────────────────────
local dropped_log = {}

local function logging_hook(evicted, _session)
    dropped_log[#dropped_log + 1] = {
        ts       = os.time(),
        n_msgs   = #evicted,
        messages = evicted,
    }
    io.write(string.format(
        "\n  [on_evict] %d messages dropped (total events: %d)\n\n",
        #evicted, #dropped_log))
    return nil
end

-- ── Demo ──────────────────────────────────────────────────────────────────────
io.write(string.format("[init] n_ctx=%d\n\n", N_CTX))

local engine = llm.new({
    model  = MODEL,
    n_ctx  = N_CTX,
    system = "You are a helpful assistant. Answer in exactly 2 sentences.",
    think  = true,
})

-- Install Option A. Switch to logging_hook to try Option B.
engine.cm:set_hook("on_evict", make_summarization_hook(engine))

io.write("=== Demo: 8-turn conversation with overflow ===\n\n")

local session = engine:session()

local topics = {
    "What is the Fibonacci sequence?",
    "Explain hash tables.",
    "What is the difference between TCP and UDP?",
    "How does a CPU execute instructions?",
    "What is garbage collection?",
    "Explain virtual memory.",
    "What is a B-tree?",
    "Describe the producer-consumer problem.",
}

for i, prompt in ipairs(topics) do
    session:add("user", prompt)
    local resp = engine:chat(session, { max_tokens = 80 })
    session:add("assistant", resp.text)

    local usage = engine:ctx_usage()
    io.write(string.format("  turn %-2d │ n_past=%d  evictions=%d  tokens_evicted=%d\n",
        i, usage.n_past, usage.n_evictions, usage.n_tokens_evicted))
end

-- ── Final stats ───────────────────────────────────────────────────────────────
io.write("\n" .. string.rep("─", 60) .. "\n")
local usage = engine:ctx_usage()
io.write("[stats]\n")
io.write(string.format("  eviction strategy : %s\n",  usage.eviction))
io.write(string.format("  n_sink            : %d\n",  usage.n_sink))
io.write(string.format("  overflow events   : %d\n",  usage.n_evictions))
io.write(string.format("  tokens evicted    : %d\n",  usage.n_tokens_evicted))
io.write(string.format("  KV fill           : %.1f%%\n", usage.fill_pct))
io.write(string.format("  dropped_log size  : %d (Option B)\n", #dropped_log))

io.write("\n[coherence check]\n")
session:add("user", "What was the first topic we discussed?")
local final = engine:chat(session, { max_tokens = 60 })
io.write("  -> " .. final.text:gsub("\n", " ") .. "\n")

engine:shutdown()
