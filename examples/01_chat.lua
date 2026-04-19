#!/usr/bin/env luajit
--- 01_chat.lua - Minimal ion7-llm pipeline: one-shot, stream, multi-turn.
---
--- Run: ION7_MODEL=/path/to/model.gguf luajit examples/01_chat.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/01_chat.lua\n")
    os.exit(1)
end

local llm    = require "ion7.llm"
local engine = llm.new({
    model  = MODEL,
    system = "You are a helpful assistant. Be concise.",
    think  = true,   -- strip <think> blocks (Qwen3.5, DeepSeek-R1, etc.)
})
io.write("[model] " .. MODEL:match("[^/]+$") .. "\n\n")

-- ── 1. One-shot chat ──────────────────────────────────────────────────────────
io.write("── 1. engine:complete() ────────────────────────────────────────────\n")
io.write("[Assistant] ")
local resp = engine:complete("What is LuaJIT and why is it fast?", {
    on_token = function(piece) io.write(piece); io.flush() end,
})
io.write("\n" .. resp:summary() .. "\n\n")

-- ── 2. Streaming ──────────────────────────────────────────────────────────────
io.write("── 2. engine:stream() ──────────────────────────────────────────────\n")
io.write("[Assistant] ")
local s2 = engine:session("Name three advantages of LuaJIT over standard Lua.")
for piece in engine:stream(s2) do
    io.write(piece); io.flush()
end
io.write("\n\n")

-- ── 3. Multi-turn with KV reuse ───────────────────────────────────────────────
io.write("── 3. Multi-turn with KV reuse ─────────────────────────────────────\n")
local session = engine:session()
session:add("user", "My name is Louis.")
io.write("[turn 1] ")
local r1 = engine:chat(session, {
    on_token = function(p) io.write(p); io.flush() end,
})
io.write("\n")
session:add("assistant", r1.text)
session:add("user", "What's my name?")
io.write("[turn 2] ")
local r2 = engine:chat(session, {
    on_token = function(p) io.write(p); io.flush() end,
})
io.write("\n" .. r2:summary() .. "\n\n")

engine:shutdown()
io.write("[shutdown]\n")
