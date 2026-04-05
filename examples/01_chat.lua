#!/usr/bin/env luajit
--- 01_chat.lua - Minimal ion7-llm pipeline: init, chat, stream, shutdown.
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

local llm = require "ion7.llm"
llm.init({
    model  = MODEL,
    system = "You are a helpful assistant. Be concise.",
    think  = true,   -- strip <think> blocks (Qwen3.5, DeepSeek-R1, etc.)
})
io.write("[model] " .. MODEL:match("[^/]+$") .. "\n\n")

-- ── 1. One-shot chat ──────────────────────────────────────────────────────────
io.write("── 1. llm.chat() ──────────────────────────────────────────────────\n")
io.write("[Assistant] ")
local resp = llm.chat("What is LuaJIT and why is it fast?", {
    on_token = function(piece) io.write(piece); io.flush() end,
})
io.write("\n" .. resp:summary() .. "\n\n")

-- ── 2. Streaming ──────────────────────────────────────────────────────────────
io.write("── 2. llm.stream() ────────────────────────────────────────────────\n")
io.write("[Assistant] ")
for piece in llm.stream("Name three advantages of LuaJIT over standard Lua.") do
    io.write(piece); io.flush()
end
io.write("\n\n")

-- ── 3. Multi-turn ─────────────────────────────────────────────────────────────
io.write("── 3. Multi-turn with KV reuse ─────────────────────────────────────\n")
local session = llm.Session.new()
session:add("user", "My name is Louis.")
io.write("[turn 1] ")
local r1 = llm.gen():chat(session, {
    on_token = function(p) io.write(p); io.flush() end,
})
io.write("\n")
session:add("assistant", r1.text)
session:add("user", "What's my name?")
io.write("[turn 2] ")
local r2 = llm.gen():chat(session, {
    on_token = function(p) io.write(p); io.flush() end,
})
io.write("\n" .. r2:summary() .. "\n\n")

llm.shutdown()
io.write("[shutdown]\n")
