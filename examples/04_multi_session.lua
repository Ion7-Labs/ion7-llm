#!/usr/bin/env luajit
--- 04_multi_session.lua - Generator vs Scheduler: automatic routing.
---
--- Usage: ION7_MODEL=/path/to/model.gguf luajit examples/04_multi_session.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/04_multi_session.lua\n")
    os.exit(1)
end

local llm = require "ion7.llm"

io.write("[init] loading...\n")
llm.init({
    model     = MODEL,
    system    = "You are a concise assistant. One paragraph max.",
    sampler   = "precise",
    n_seq_max = 2,
    think     = true,  -- allow batch of 2 sessions
})
io.write("[init] ready\n\n")

-- ── 1 session -> llm.chat() -> Generator (optimal path) ───────────────────────
io.write("── 1 session: llm.chat() -> Generator ──────────────────────────────\n")
local r = llm.chat("What is LuaJIT?", {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n" .. r:summary() .. "\n\n")

-- ── Multi-turn with KV reuse ─────────────────────────────────────────────────
io.write("── Multi-turn: KV reuse across turns ──────────────────────────────\n")
local session = llm.Session.new()
session:add("user", "My name is Louis.")
local r1 = llm.gen():chat(session, {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n")
session:add("assistant", r1.text)
session:add("user", "What's my name?")
local r2 = llm.gen():chat(session, {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n" .. r2:summary() .. "\n\n")

-- ── 2 sessions -> llm.batch() -> Scheduler (1 GPU decode/step) ─────────────────
io.write("── 2 sessions: llm.batch() -> Scheduler (true parallel) ────────────\n")
local sa = llm.Session.new(); sa:add("user", "Say 'hello' in French. One word only.")
local sb = llm.Session.new(); sb:add("user", "Say 'hello' in Spanish. One word only.")
local out = { a = {}, b = {} }

llm.batch({
    { session = sa, max_tokens = 8,
      on_piece  = function(p) out.a[#out.a+1] = p end,
      on_done   = function(r) io.write("[A] " .. (r and r.text or table.concat(out.a)) .. "\n") end },
    { session = sb, max_tokens = 8,
      on_piece  = function(p) out.b[#out.b+1] = p end,
      on_done   = function(r) io.write("[B] " .. (r and r.text or table.concat(out.b)) .. "\n") end },
})
io.write("[1 GPU decode per step - both sessions ran in parallel]\n\n")

llm.shutdown()
io.write("[shutdown]\n")
