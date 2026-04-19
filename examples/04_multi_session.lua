#!/usr/bin/env luajit
--- 04_multi_session.lua - Generator vs Scheduler: single vs parallel sessions.
---
--- Run: ION7_MODEL=/path/to/model.gguf luajit examples/04_multi_session.lua

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/04_multi_session.lua\n")
    os.exit(1)
end

local llm    = require "ion7.llm"
local engine = llm.new({
    model     = MODEL,
    system    = "You are a concise assistant. One paragraph max.",
    sampler   = "precise",
    n_seq_max = 2,
    think     = true,
})
io.write("[model] " .. MODEL:match("[^/]+$") .. "\n\n")

-- ── One session → engine:complete() ──────────────────────────────────────────
io.write("── Single session: engine:complete() ──────────────────────────────\n")
local r = engine:complete("What is LuaJIT?", {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n" .. r:summary() .. "\n\n")

-- ── Multi-turn with KV reuse ─────────────────────────────────────────────────
io.write("── Multi-turn: KV reuse across turns ──────────────────────────────\n")
local session = engine:session()
session:add("user", "My name is Louis.")
local r1 = engine:chat(session, {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n")
session:add("assistant", r1.text)
session:add("user", "What's my name?")
local r2 = engine:chat(session, {
    on_token = function(p) io.write(p); io.flush() end
})
io.write("\n" .. r2:summary() .. "\n\n")

-- ── 2 sessions → engine:batch() → 1 GPU decode/step ─────────────────────────
io.write("── 2 sessions: engine:batch() → true parallel decode ───────────────\n")
local sa = engine:session(); sa:add("user", "Say 'hello' in French. One word only.")
local sb = engine:session(); sb:add("user", "Say 'hello' in Spanish. One word only.")
local out = { a = {}, b = {} }

engine:batch({
    { session    = sa,
      max_tokens = 8,
      on_piece   = function(p) out.a[#out.a + 1] = p end,
      on_done    = function(r) io.write("[A] " .. (r and r.text or table.concat(out.a)) .. "\n") end },
    { session    = sb,
      max_tokens = 8,
      on_piece   = function(p) out.b[#out.b + 1] = p end,
      on_done    = function(r) io.write("[B] " .. (r and r.text or table.concat(out.b)) .. "\n") end },
})
io.write("[1 GPU decode per step — both sessions ran in parallel]\n\n")

engine:shutdown()
io.write("[shutdown]\n")
