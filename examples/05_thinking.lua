#!/usr/bin/env luajit
--- 05_thinking.lua - Reasoning models: think_budget, on_think, checkpoint/rollback.
---
--- Demonstrates:
---   - think = true: strip <think> blocks from output
---   - on_think callback: stream reasoning separately in real time
---   - think_budget: cap the thinking phase to N tokens
---   - resp:think(): access the raw reasoning trace
---   - engine:checkpoint() / engine:rollback(): KV snapshot + restore
---
--- Works with any reasoning model: Qwen3.5, DeepSeek-R1, Phi-4-reasoning, etc.
---
--- Run:
---   ION7_MODEL=/path/to/thinking-model.gguf luajit examples/05_thinking.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/05_thinking.lua\n")
    os.exit(1)
end

local llm    = require "ion7.llm"
local engine = llm.new({
    model        = MODEL,
    sampler      = "thinking",   -- temperature=0.6 (Qwen3 recommended)
    think        = true,         -- strip <think>...</think> from .text
    think_budget = 512,
})
io.write("[model] " .. MODEL:match("[^/]+$") .. "\n\n")

-- ── 1. Basic think stripping ──────────────────────────────────────────────────
io.write("── 1. Reasoning with think stripping ────────────────────────────────\n")

local r1 = engine:complete("What is 17 × 23? Show just the answer, no explanation.", {
    max_tokens = 256,
})
io.write("Answer: " .. r1.text .. "\n")
if r1:think() then
    local preview = r1:think():sub(1, 120):gsub("\n", " ")
    io.write(string.format("Think (%d chars): %s...\n", #r1:think(), preview))
end
io.write(r1:summary() .. "\n\n")

-- ── 2. on_think callback — stream reasoning in real time ─────────────────────
io.write("── 2. on_think callback — dim reasoning, bright answer ──────────────\n")

local s2 = engine:session("Is 97 a prime number? Reason step by step, then answer yes or no.")
io.write("\027[2m[thinking] \027[0m")
local r2 = engine:chat(s2, {
    max_tokens = 384,
    on_think   = function(p) io.write("\027[2m" .. p .. "\027[0m"); io.flush() end,
    on_token   = function(p) io.write(p);                           io.flush() end,
})
io.write("\n" .. r2:summary() .. "\n\n")

-- ── 3. think_budget cap ───────────────────────────────────────────────────────
io.write("── 3. think_budget = 64 tokens ──────────────────────────────────────\n")
io.write("    Model is forced out of the think block after 64 tokens.\n\n")

local r3 = engine:complete("List 3 prime numbers greater than 100.", {
    max_tokens   = 128,
    think_budget = 64,
})
io.write("Answer: " .. r3.text .. "\n")
if r3:think() then
    io.write(string.format("Think (capped, %d chars)\n", #r3:think()))
end
io.write("\n")

-- ── 4. checkpoint() + rollback() ─────────────────────────────────────────────
io.write("── 4. KV checkpoint / rollback ──────────────────────────────────────\n")
io.write("    Generate, validate, rollback if invalid, retry.\n\n")

local session = engine:session()
session:add("user", 'Return: {"status": "ok", "code": <integer 200-599>}')

engine:checkpoint()
io.write("[attempt 1] ")
local r4 = engine:chat(session, {
    max_tokens = 32,
    on_token   = function(p) io.write(p) end,
})
io.write("\n")

local text = r4.text:match("^%s*(.-)%s*$")
if text:sub(1, 1) == "{" and text:sub(-1) == "}" then
    io.write("[valid JSON — keeping]\n")
else
    io.write("[invalid — rolling back and retrying]\n")
    engine:rollback()
    session._dirty = true
    io.write("[attempt 2] ")
    local r4b = engine:chat(session, {
        max_tokens = 32,
        on_token   = function(p) io.write(p) end,
    })
    io.write("\n")
end
io.write("\n")

engine:shutdown()
io.write("[shutdown]\n")
