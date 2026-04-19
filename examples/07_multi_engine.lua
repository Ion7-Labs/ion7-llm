#!/usr/bin/env luajit
--- 07_multi_engine.lua - Multiple independent LLM engines, tool loop, on_think.
---
--- Demonstrates:
---   - llm.new() → LLM object (multiple engines, no global state)
---   - engine:session() → Session
---   - engine:chat(session) → Response (session-based API)
---   - engine:ctx_usage() → KV fill stats
---   - engine:fork(session) → branch a conversation
---   - engine:checkpoint() / engine:rollback()
---   - engine:chat_with_tools() → ReAct tool loop
---   - opts.on_think → stream reasoning tokens separately
---
--- Two-engine pattern: a large engine for generation, a small engine for
--- tool results / reranking — on the same GPU without reinitialising.
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/07_multi_engine.lua
---
--- @author Ion7-Labs

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/07_multi_engine.lua\n")
    os.exit(1)
end

local llm = require "ion7.llm"

-- ── 1. llm.new() - object-oriented, no global state ──────────────────────────
io.write("── 1. llm.new() — OOP engine ──────────────────────────────────────\n")

local engine = llm.new({
    model  = MODEL,
    system = "You are a concise assistant. One sentence per answer.",
    think  = true,    -- strip <think> blocks
    n_ctx  = 4096,
})

-- engine:session() creates a Session bound to this engine's system prompt
local session = engine:session()
session:add("user", "What is LuaJIT?")

io.write("[turn 1] ")
local r1 = engine:chat(session, {
    on_token = function(p) io.write(p); io.flush() end,
    on_think = function(p) io.write("\027[2m" .. p .. "\027[0m"); io.flush() end,
    max_tokens = 128,
})
io.write("\n")
session:add("assistant", r1.text)

session:add("user", "Why is it faster than standard Lua?")
io.write("[turn 2] ")
local r2 = engine:chat(session, {
    on_token = function(p) io.write(p); io.flush() end,
    max_tokens = 128,
})
io.write("\n")
io.write("  " .. r2:summary() .. "\n\n")

-- ── 2. ctx_usage() — KV fill stats ───────────────────────────────────────────
io.write("── 2. ctx_usage() ─────────────────────────────────────────────────\n")
local usage = engine:ctx_usage()
io.write(string.format("  KV: %d/%d tokens (%.1f%% full)\n",
    usage.n_past, usage.n_ctx, usage.fill_pct))
io.write(string.format("  prefix: %d tokens cached | evictions: %d\n\n",
    usage.prefix_tokens, usage.n_evictions))

-- ── 3. fork() — branch a conversation ────────────────────────────────────────
io.write("── 3. engine:fork() — branch ──────────────────────────────────────\n")

local base = engine:session()
base:add("user", "Recommend one programming language for game modding.")

local branch_a = engine:fork(base)
local branch_b = engine:fork(base)
branch_a:add("user", "For a C++ engine.")
branch_b:add("user", "For a Python engine.")

io.write("[branch A - C++] ")
local ra = engine:chat(branch_a, { max_tokens = 64,
    on_token = function(p) io.write(p); io.flush() end })
io.write("\n")

io.write("[branch B - Python] ")
local rb = engine:chat(branch_b, { max_tokens = 64,
    on_token = function(p) io.write(p); io.flush() end })
io.write("\n\n")

-- ── 4. checkpoint() / rollback() ─────────────────────────────────────────────
io.write("── 4. checkpoint() / rollback() ───────────────────────────────────\n")

local s4 = engine:session("Return a single integer between 1 and 10.")
engine:checkpoint()

local r4a = engine:chat(s4, { max_tokens = 8,
    on_token = function(p) io.write(p); io.flush() end })
io.write("\n")

local num = tonumber(r4a.text:match("%d+"))
if not num or num < 1 or num > 10 then
    io.write("  [out of range — rolling back]\n  [retry] ")
    engine:rollback()
    s4._dirty = true
    local r4b = engine:chat(s4, { max_tokens = 8,
        on_token = function(p) io.write(p); io.flush() end })
    io.write("\n\n")
else
    io.write("  [valid: " .. num .. "]\n\n")
end

-- ── 5. chat_with_tools() — ReAct loop ────────────────────────────────────────
io.write("── 5. chat_with_tools() — ReAct ───────────────────────────────────\n")
io.write("  Model calls tools; engine manages the session automatically.\n\n")

-- Toy grammar (JSON tool call schema)
local Grammar = llm.Grammar
local tool_grammar
if Grammar then
    tool_grammar = Grammar.from_json_schema({
        type = "object",
        properties = {
            tool = { type = "string", enum = { "calc", "none" } },
            expr = { type = "string" },
        },
        required = { "tool" },
    })
end

-- Simple parse_fn: look for {"tool": ...} in response
local function parse_tool(text)
    local tool = text:match('"tool"%s*:%s*"([^"]+)"')
    if not tool or tool == "none" then return nil end
    local expr = text:match('"expr"%s*:%s*"([^"]+)"')
    return { name = tool, args = expr or "" }
end

-- Simple handler: evaluate arithmetic expression
local function handler(name, args, _session)
    if name == "calc" then
        local fn = load("return " .. args)
        if fn then
            local ok, result = pcall(fn)
            return ok and tostring(result) or ("error: " .. result)
        end
        return "error: invalid expression"
    end
    return "unknown tool"
end

local tool_session = engine:session()
tool_session:add("user",
    'Calculate 17 * 23 + 5. Respond with JSON: {"tool":"calc","expr":"<expression>"} '
    .. 'or {"tool":"none"} if you know the answer directly.')

local resp, turns = engine:chat_with_tools(tool_session, {
    parse_fn   = parse_tool,
    handler    = handler,
    max_turns  = 4,
    max_tokens = 64,
    on_token   = function(p) io.write(p); io.flush() end,
})
io.write(string.format("\n  [%d turn(s)] final: %s\n\n", turns, resp.text))

-- ── 6. engine:shutdown() ─────────────────────────────────────────────────────
io.write("── 6. Cleanup ──────────────────────────────────────────────────────\n")
engine:shutdown()
io.write("[engine shut down]\n")
