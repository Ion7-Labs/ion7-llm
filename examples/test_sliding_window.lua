#!/usr/bin/env luajit
--- test_sliding_window.lua - Verify sliding window context management.
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/test_sliding_window.lua
---   N_CTX=512 ION7_MODEL=... luajit examples/test_sliding_window.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
local N_CTX = tonumber(os.getenv("N_CTX")) or 512

if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/test_sliding_window.lua\n")
    os.exit(1)
end

local llm    = require "ion7.llm"
local engine = llm.new({
    model  = MODEL,
    n_ctx  = N_CTX,
    system = "You are a helpful assistant. Answer in exactly 2-3 sentences.",
    think  = true,
})

io.write(string.format("[init] n_ctx=%d  (overflow expected around %d tokens)\n\n",
    N_CTX, N_CTX - 64))

local can_shift = engine.ctx:kv_can_shift()
local usage0    = engine:ctx_usage()

io.write(string.format("[kv_can_shift] %s\n",
    can_shift and "YES — sliding window active" or "NO — hard reset fallback"))
io.write(string.format("[eviction]     %s\n", usage0.eviction))
io.write(string.format("[n_sink]       %d\n", usage0.n_sink))
io.write(string.rep("─", 60) .. "\n\n")

local session = engine:session()

local prompts = {
    "What is the Fibonacci sequence? Give several examples.",
    "Explain what a hash table is and how collision resolution works.",
    "What is the difference between TCP and UDP? Give examples of when to use each.",
    "Describe how a CPU executes instructions step by step.",
    "What is garbage collection and how does a mark-and-sweep algorithm work?",
    "Explain the concept of virtual memory and why operating systems use it.",
    "What is a B-tree and why is it used in databases?",
    "Describe the producer-consumer problem and a solution using semaphores.",
}

local results = {}

for i, prompt in ipairs(prompts) do
    local before    = engine.ctx:n_past()
    local tok_count = 0

    session:add("user", prompt)
    local resp = engine:chat(session, {
        max_tokens = 120,
        on_token   = function() tok_count = tok_count + 1 end,
    })
    session:add("assistant", resp.text)

    local after      = engine.ctx:n_past()
    local overflowed = (before > 0 and after < before + tok_count - 10)

    results[i] = { turn = i, before = before, after = after,
                   generated = tok_count, overflow = overflowed }

    local flag = overflowed and " \27[33m← OVERFLOW\27[0m" or ""
    io.write(string.format("  turn %d │ n_past before=%-4d after=%-4d (+%d tok)%s\n",
        i, before, after, tok_count, flag))

    if overflowed then
        local drop = before - (after - tok_count)
        io.write(string.format("         │ dropped ~%d tokens from history\n", drop))
        if can_shift then
            io.write(after > N_CTX * 0.5
                and "         │ \27[32m✓ recent context preserved (sliding window)\27[0m\n"
                or  "         │ \27[31m✗ more context lost than expected\27[0m\n")
        else
            io.write(after > N_CTX * 0.3
                and "         │ \27[32m✓ recent history re-encoded after reset\27[0m\n"
                or  "         │ \27[33m~ nearly empty (hard reset, minimal history)\27[0m\n")
        end
    end
end

-- ── Summary ───────────────────────────────────────────────────────────────────
io.write("\n" .. string.rep("─", 60) .. "\n[summary]\n")

local n_overflows = 0
for _, r in ipairs(results) do
    if r.overflow then n_overflows = n_overflows + 1 end
end

local final_usage = engine:ctx_usage()
io.write(string.format("  %d turns, %d overflow(s) detected\n", #results, n_overflows))
io.write(string.format("  n_ctx=%d, n_past final=%d (%.1f%% full)\n",
    N_CTX, final_usage.n_past, final_usage.fill_pct))
io.write(string.format("  strategy: kv_can_shift=%s, eviction=%s, n_sink=%d\n",
    tostring(can_shift), final_usage.eviction, final_usage.n_sink))

if n_overflows == 0 then
    io.write("  [!] No overflow triggered — try with N_CTX=256\n")
end

io.write("\n[memory test post-overflow]\n")
session:add("user", "What was the first topic we discussed?")
local final = engine:chat(session, { max_tokens = 60 })
io.write("  -> " .. final.text:gsub("\n", " ") .. "\n")
if n_overflows > 0 then
    io.write("  (oldest turns were evicted; model may not recall the very first topic)\n")
else
    io.write("  (no overflow — full history available)\n")
end

engine:shutdown()
