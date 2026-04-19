#!/usr/bin/env luajit
--- 02_sessions.lua - Session API: multi-turn KV reuse, fork, serialize.
---
--- Demonstrates:
---   - Multi-turn conversation with KV cache reuse
---   - engine:fork() for branching dialogues
---   - Session:serialize() / Session.deserialize() for persistence
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/02_sessions.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/02_sessions.lua\n")
    os.exit(1)
end

local llm    = require "ion7.llm"
local engine = llm.new({
    model  = MODEL,
    system = "You are a concise assistant. One sentence per answer.",
})
io.write("[model] " .. MODEL:match("[^/]+$") .. "\n\n")

-- ── 1. Multi-turn with KV reuse ───────────────────────────────────────────────
io.write("── 1. Multi-turn KV reuse ─────────────────────────────────────────\n")

local session = engine:session()
session:add("user", "My name is Louis and I work on LuaJIT LLMs.")

io.write("[turn 1] ")
local r1 = engine:chat(session, {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n")
session:add("assistant", r1.text)

session:add("user", "What's my name and what am I building?")
io.write("[turn 2] ")
local r2 = engine:chat(session, {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n")
io.write("  " .. r2:summary() .. "\n\n")

-- ── 2. engine:fork() — branch a dialogue ─────────────────────────────────────
io.write("── 2. engine:fork() — two independent branches ─────────────────────\n")

local base = engine:session()
base:add("user", "What is Lua?")

local branch_a = engine:fork(base)
local branch_b = engine:fork(base)
branch_a:add("user", "Explain it to a C programmer.")
branch_b:add("user", "Explain it to a Python developer.")

io.write("[branch A - C programmer]\n")
local ra = engine:chat(branch_a, {
    on_token   = function(p) io.write(p); io.flush() end,
    max_tokens = 80,
})
io.write("\n\n")

io.write("[branch B - Python developer]\n")
local rb = engine:chat(branch_b, {
    on_token   = function(p) io.write(p); io.flush() end,
    max_tokens = 80,
})
io.write("\n\n")

-- Release KV slots when done
engine:release(branch_a)
engine:release(branch_b)

-- ── 3. Serialize / deserialize ────────────────────────────────────────────────
io.write("── 3. Session serialize / deserialize ──────────────────────────────\n")

local saved = engine:session()
saved:add("user",      "What year was Lua created?")
saved:add("assistant", "1993.")

local serialized = saved:serialize()
io.write(string.format("  Serialized: %d messages\n", #serialized.messages))

local restored = llm.Session.deserialize(serialized)
restored:add("user", "By whom?")
io.write("[restored session, new turn] ")
local r_restored = engine:chat(restored, {
    on_token   = function(p) io.write(p); io.flush() end,
    max_tokens = 32,
})
io.write("\n  " .. r_restored:summary() .. "\n\n")

engine:shutdown()
io.write("[shutdown]\n")
