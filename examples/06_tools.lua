#!/usr/bin/env luajit
--- @example examples.06_tools
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 06 — Tool calling : declarative `Tool`s + interleaved-thinking loop
--- ════════════════════════════════════════════════════════════════════════
---
--- Three pieces wire the tool-calling pattern :
---
---   1. `Tool` declarative definitions — name, description, JSON
---      schema for arguments, optional Lua handler.
---   2. `ToolSet` registry — the engine consults it to build the
---      `tools:` block in the chat-template render.
---   3. `tools.loop.run(engine, session, opts)` — orchestrates the
---      multi-turn dispatch :
---
---        - chat → if response carries tool_calls : append the
---          assistant turn (preserving thinking + tool_calls), run
---          each tool's handler, append the result as a tool message,
---          loop ;
---        - chat → no tool_calls : return the final Response.
---
--- The example below registers two tools (`get_weather`, `get_time`)
--- and runs the loop. Whether the model actually calls them depends
--- on the model's tool-use training ; this script will work either
--- way (single content turn or multi-turn dispatch).
---
---   ION7_MODEL=/path/to/chat.gguf luajit examples/06_tools.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
local llm  = require "ion7.llm"

ion7.init({ log_level = 0 })

local model = ion7.Model.load(MODEL, {
    n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS")) or 0,
})
local ctx   = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
local vocab = model:vocab()
local cm, engine = llm.pipeline(ctx, vocab, { headroom = 256 })

cm:set_system(
    "You are a helpful assistant. Use tools when the answer requires " ..
    "real-time data. Only emit a tool_call when a tool is the right answer.")

-- ── 1. Declare the tools ────────────────────────────────────────────────

local Tool    = llm.tools.Tool
local ToolSet = llm.tools.ToolSet

local toolset = ToolSet.new({
    Tool.new({
        name        = "get_weather",
        description = "Return the current weather for a city.",
        schema      = {
            type       = "object",
            properties = { city = { type = "string" } },
            required   = { "city" },
        },
        handler     = function(args)
            -- A real implementation would call a weather API. We just
            -- return a deterministic stub so the example is self-contained.
            return {
                city        = args.city,
                temperature = 22,
                conditions  = "partly cloudy",
                units       = "celsius",
            }
        end,
    }),

    Tool.new({
        name        = "get_time",
        description = "Return the current local time as ISO-8601.",
        schema      = {
            type       = "object",
            properties = {},
        },
        handler     = function() return { iso = os.date("!%Y-%m-%dT%H:%M:%SZ") } end,
    }),
})

-- ── 2. Run the tool-use loop ─────────────────────────────────────────────

local session = llm.Session.new()
session:add_user("What is the weather in Paris right now ?")

print("[loop] starting tool-use loop")
local final, n_turns = llm.tools.loop(engine, session, {
    tools     = toolset,
    max_turns = 4,
    on_call   = function(call, result)
        print(string.format("  ↳ tool '%s' called with %s -> %s",
            call.name,
            require("ion7.vendor.json").encode(call.arguments or {}),
            require("ion7.vendor.json").encode(result)))
    end,
})

print(string.format("\n[loop] finished in %d turn(s)", n_turns))
print("\nFinal answer :")
print(final.content)
print("\n" .. final:summary())

cm:release(session)
ctx:free()
model:free()
ion7.shutdown()
