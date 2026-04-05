#!/usr/bin/env luajit
--- test_sliding_window.lua - Verify sliding window context management.
---
--- How it works:
---   1. Use a small context window (N_CTX tokens)
---   2. Generate many tokens per turn to fill it quickly
---   3. Track n_past before/after each turn
---   4. Sliding window (kv_can_shift=YES): n_past drops then climbs again
---   5. Hard reset    (kv_can_shift=NO):  n_past drops back near prefix_n,
---      then climbs again with trimmed history
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/test_sliding_window.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL  = os.getenv("ION7_MODEL")
local N_CTX  = tonumber(os.getenv("N_CTX")) or 512   -- small to trigger overflow fast

if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/test_sliding_window.lua\n")
    os.exit(1)
end

local llm = require "ion7.llm"

io.write(string.format("[init] n_ctx=%d  (overflow expected around %d tokens)\n\n",
    N_CTX, N_CTX - 64))

llm.init({
    model    = MODEL,
    n_ctx    = N_CTX,
    system   = "You are a helpful assistant. Answer in exactly 2-3 sentences.",
    think    = true,
    -- n_sink=4 (default): first 4 tokens always preserved (attention sink)
    -- eviction="message" (default): evict whole messages, not arbitrary tokens
})

local ctx     = llm.ctx()
local session = llm.Session.new({ system = "You are a helpful assistant. Answer in exactly 2-3 sentences." })
local gen     = llm.gen()

local can_shift = ctx:kv_can_shift()
io.write(string.format("[kv_can_shift] %s\n",
    can_shift and "YES - sliding window active" or "NO - hard reset fallback"))
io.write(string.format("[eviction]     %s\n", llm.cm():stats().eviction))
io.write(string.format("[n_sink]       %d\n", llm.cm():stats().n_sink))
io.write(string.rep("─", 60) .. "\n\n")

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
    local before = ctx:n_past()
    session:add("user", prompt)

    local tok_count = 0
    local resp = gen:chat(session, {
        max_tokens = 120,
        on_token   = function() tok_count = tok_count + 1 end,
    })
    session:add("assistant", resp.text)

    local after      = ctx:n_past()
    local overflowed = (before > 0 and after < before + tok_count - 10)

    results[i] = {
        turn      = i,
        before    = before,
        after     = after,
        generated = tok_count,
        overflow  = overflowed,
    }

    local flag = overflowed and " \27[33m← OVERFLOW\27[0m" or ""
    io.write(string.format("  turn %d │ n_past before=%-4d after=%-4d (+%d tok)%s\n",
        i, before, after, tok_count, flag))

    if overflowed then
        local drop = before - (after - tok_count)
        io.write(string.format("         │ dropped ~%d tokens from history\n", drop))
        if can_shift then
            -- Sliding window: n_past stays high (tokens shifted backward)
            if after > N_CTX * 0.5 then
                io.write("         │ \27[32m✓ recent context preserved (sliding window)\27[0m\n")
            else
                io.write("         │ \27[31m✗ more context lost than expected\27[0m\n")
            end
        else
            -- Hard reset: n_past drops near prefix_n, then re-filled with trimmed history
            -- n_past > 30% of n_ctx means history was re-encoded (not fully empty)
            if after > N_CTX * 0.3 then
                io.write("         │ \27[32m✓ recent history re-encoded after reset\27[0m\n")
            else
                io.write("         │ \27[33m~ context nearly empty (hard reset, minimal history)\27[0m\n")
            end
        end
    end
end

-- ── Summary ───────────────────────────────────────────────────────────────────
io.write("\n" .. string.rep("─", 60) .. "\n")
io.write("[summary]\n")

local n_overflows = 0
for _, r in ipairs(results) do
    if r.overflow then n_overflows = n_overflows + 1 end
end

io.write(string.format("  %d turns, %d overflow(s) detected\n", #results, n_overflows))
io.write(string.format("  n_ctx=%d, n_past final=%d\n", N_CTX, ctx:n_past()))
io.write(string.format("  strategy: kv_can_shift=%s, eviction=%s, n_sink=%d\n",
    tostring(can_shift), llm.cm():stats().eviction, llm.cm():stats().n_sink))

if n_overflows == 0 then
    io.write("  [!] No overflow triggered - try with N_CTX=256\n")
end

-- ── Memory test ───────────────────────────────────────────────────────────────
-- Ask about the first topic. With overflow, the oldest turns get evicted, so
-- the model may not recall the very first topic - this is expected.
-- The test just verifies the model responds coherently (no crash, no empty output).
io.write("\n[memory test post-overflow]\n")
session:add("user", "What was the first topic we discussed?")
local final = gen:chat(session, { max_tokens = 60 })
io.write("  -> " .. final.text:gsub("\n", " ") .. "\n")
if n_overflows > 0 then
    io.write("  (oldest turns were evicted; model may not recall the very first topic)\n")
else
    io.write("  (no overflow - full history available)\n")
end

llm.shutdown()
