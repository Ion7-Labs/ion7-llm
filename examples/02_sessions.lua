#!/usr/bin/env luajit
--- 02_sessions.lua - Session API: multi-turn KV reuse, fork, serialize.
---
--- Demonstrates:
---   - Multi-turn conversation with KV cache reuse (no re-prefill of prior turns)
---   - Session:fork() for branching dialogues
---   - Session:serialize() / Session.deserialize() for persistence
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/02_sessions.lua
---
--- @author Ion7-Labs

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/02_sessions.lua\n")
    os.exit(1)
end

local llm = require "ion7.llm"

llm.init({
    model  = MODEL,
    system = "You are a concise assistant. One sentence per answer.",
})
io.write("[model] " .. MODEL:match("[^/]+$") .. "\n\n")

-- ── 1. Multi-turn with KV reuse ───────────────────────────────────────────────
io.write("── 1. Multi-turn KV reuse ─────────────────────────────────────────\n")
io.write("    Each turn reuses the KV cache from the previous turn.\n")
io.write("    No re-prefill - only the new tokens are processed.\n\n")

local session = llm.Session.new()
session:add("user", "My name is Louis and I work on LuaJIT LLMs.")

io.write("[turn 1] ")
local r1 = llm.gen():chat(session, {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n")
session:add("assistant", r1.text)

session:add("user", "What's my name and what am I building?")
io.write("[turn 2] ")
local r2 = llm.gen():chat(session, {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n")
io.write("        " .. r2:summary() .. "\n\n")

-- ── 2. Session:fork() - branch a dialogue ────────────────────────────────────
io.write("── 2. Session fork - two independent branches ─────────────────────\n")
io.write("    Fork at 'What is Lua?' - each branch answers differently.\n\n")

local base = llm.Session.new()
base:add("user", "What is Lua?")

local branch_a = base:fork()
local branch_b = base:fork()
branch_a:add("user", "Explain it to a C programmer.")
branch_b:add("user", "Explain it to a Python developer.")

io.write("[branch A - C programmer]\n")
local ra = llm.gen():chat(branch_a, {
    on_token = function(p) io.write(p); io.flush() end,
    max_tokens = 80,
})
io.write("\n\n")

io.write("[branch B - Python developer]\n")
local rb = llm.gen():chat(branch_b, {
    on_token = function(p) io.write(p); io.flush() end,
    max_tokens = 80,
})
io.write("\n\n")

-- ── 3. serialize / deserialize ────────────────────────────────────────────────
io.write("── 3. Session serialize / deserialize ──────────────────────────────\n")

local saved_session = llm.Session.new({ system = "Be concise." })
saved_session:add("user", "What year was Lua created?")
saved_session:add("assistant", "1993.")

local serialized = saved_session:serialize()
io.write(string.format("    Serialized: %d messages, system=%s\n",
    #serialized.messages, tostring(serialized.system)))

local restored = llm.Session.deserialize(serialized)
restored:add("user", "By whom?")
io.write("[restored session, new turn] ")
local r_restored = llm.gen():chat(restored, {
    on_token = function(p) io.write(p); io.flush() end,
    max_tokens = 32,
})
io.write("\n")
io.write("    " .. r_restored:summary() .. "\n\n")

llm.shutdown()
io.write("[shutdown]\n")
