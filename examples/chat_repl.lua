#!/usr/bin/env luajit
--- chat_repl.lua - Interactive multi-turn chat REPL.
---
--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit examples/chat_repl.lua
---
--- Options:
---   SYSTEM="You are a helpful assistant."    Custom system prompt
---   SAMPLER=thinking                         Sampler profile (default: balanced)
---   THINK=1                                  Strip <think> blocks

package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               package.path

local MODEL   = os.getenv("ION7_MODEL")
local SYSTEM  = os.getenv("SYSTEM")  or "You are a helpful assistant."
local SAMPLER = os.getenv("SAMPLER") or "balanced"
local THINK   = os.getenv("THINK")   == "1"

if not MODEL then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/chat_repl.lua\n")
    os.exit(1)
end

local llm    = require "ion7.llm"
local engine = llm.new({
    model   = MODEL,
    system  = SYSTEM,
    sampler = SAMPLER,
    think   = THINK,
})
io.write(string.format("[model] %s\n", MODEL:match("[^/]+$")))
io.write(string.format("[system] %s\n", SYSTEM))
io.write("[type your message, Enter to send — Ctrl+C or /exit to quit]\n")
io.write(string.rep("─", 60) .. "\n\n")

local session = engine:session()

while true do
    io.write("\27[1;32mYou>\27[0m ")
    io.flush()

    local input = io.read("*l")

    if not input or input == "/exit" or input == "/quit" then
        io.write("\n[bye]\n")
        break
    end

    if input == "/reset" then
        session = engine:session()
        io.write("\27[33m[session reset]\27[0m\n\n")
        goto continue
    end

    if input == "/stats" then
        local u = engine:ctx_usage()
        io.write(string.format(
            "\27[2m[KV %d/%d (%.0f%%) | evictions %d | prefix %d tok]\27[0m\n\n",
            u.n_past, u.n_ctx, u.fill_pct, u.n_evictions, u.prefix_tokens))
        goto continue
    end

    if input == "/help" then
        io.write("  /reset   — clear conversation history\n")
        io.write("  /stats   — show KV cache usage\n")
        io.write("  /exit    — quit\n\n")
        goto continue
    end

    if input:match("^%s*$") then goto continue end

    session:add("user", input)

    io.write("\27[1;34mAssistant>\27[0m ")
    io.flush()

    local resp = engine:chat(session, {
        on_token = function(piece) io.write(piece); io.flush() end,
        on_think = THINK and function()
            -- think tokens suppressed — use on_think to show a spinner or log
        end or nil,
    })

    io.write("\n")

    if THINK and resp:think() then
        local think = resp:think():match("^%s*(.-)%s*$")
        if #think > 0 then
            io.write(string.format("\27[2m[think: %d chars]\27[0m\n", #think))
        end
    end

    io.write(string.format("\27[2m%s\27[0m\n\n", resp:summary()))

    session:add("assistant", resp.text)

    ::continue::
end

engine:shutdown()
