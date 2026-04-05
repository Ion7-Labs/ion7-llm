#!/usr/bin/env luajit
--- 05_thinking.lua - Reasoning models: think_budget, think() accessor, rollback.
---
--- Demonstrates:
---   - opts.think = true: strip <think> blocks from output
---   - think_budget: cap the thinking phase to N tokens
---   - resp:think(): access the raw reasoning trace
---   - Generator:checkpoint() / Generator:rollback(): KV snapshot + restore
---   - "thinking" sampler profile (temperature=0.6)
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

local llm = require "ion7.llm"

llm.init({
    model        = MODEL,
    sampler      = "thinking",   -- temperature=0.6 (Qwen3 recommended)
    think        = true,         -- strip <think>...</think> from .text
    think_budget = 512,          -- give up on think block after 512 tokens
})
io.write("[model] " .. MODEL:match("[^/]+$") .. "\n\n")

-- ── 1. Basic think - clean output ─────────────────────────────────────────────
io.write("── 1. Reasoning with think stripping ──────────────────────────────\n")

local r1 = llm.chat("What is 17 × 23? Show just the answer, no explanation.", {
    max_tokens = 256,
})
io.write("Answer: " .. r1.text .. "\n")
if r1:think() then
    local think = r1:think()
    local preview = think:sub(1, 120):gsub("\n", " ")
    io.write(string.format("Think (%d chars): %s...\n", #think, preview))
end
io.write(r1:summary() .. "\n\n")

-- ── 2. think_budget - cap reasoning tokens ────────────────────────────────────
io.write("── 2. think_budget = 64 tokens ────────────────────────────────────\n")
io.write("    Model is forced out of the think block after 64 tokens.\n\n")

local r2 = llm.chat("List 3 prime numbers greater than 100.", {
    max_tokens   = 128,
    think_budget = 64,  -- override per-call
})
io.write("Answer: " .. r2.text .. "\n")
if r2:think() then
    io.write(string.format("Think (capped, %d chars)\n", #r2:think()))
end
io.write("\n")

-- ── 3. checkpoint() + rollback() ─────────────────────────────────────────────
io.write("── 3. KV checkpoint / rollback ────────────────────────────────────\n")
io.write("    Generate, validate, rollback if invalid, retry.\n\n")

local gen = llm.gen()
local session = llm.Session.new({
    system = "You are a JSON API. Respond ONLY with valid JSON, nothing else."
})
session:add("user", 'Return: {"status": "ok", "code": <integer 200-599>}')

-- Save KV state before speculative generation
gen:checkpoint()
io.write("[attempt 1] ")
local r3 = gen:chat(session, { max_tokens = 32, on_token = function(p) io.write(p) end })
io.write("\n")

-- Validate (simple check: starts with '{' and ends with '}')
local text = r3.text:match("^%s*(.-)%s*$")
if text:sub(1,1) == "{" and text:sub(-1) == "}" then
    io.write("[valid JSON - keeping]\n")
else
    io.write("[invalid - rolling back and retrying]\n")
    gen:rollback()  -- KV restored to before attempt 1
    session._dirty = true

    io.write("[attempt 2] ")
    local r3b = gen:chat(session, { max_tokens = 32, on_token = function(p) io.write(p) end })
    io.write("\n")
end
io.write("\n")

-- ── 4. Streaming with think ───────────────────────────────────────────────────
io.write("── 4. Streaming - visible tokens only, think filtered ──────────────\n")

io.write("  Answer: ")
for piece in llm.stream("Is 97 a prime number? Answer: yes or no.", {
    max_tokens = 128,
}) do
    io.write(piece)
    io.flush()
end
io.write("\n\n")

llm.shutdown()
io.write("[shutdown]\n")
