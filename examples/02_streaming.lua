#!/usr/bin/env luajit
--- @example examples.02_streaming
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 02 — Streaming : typed-chunk iterator, content / thinking / stop demux
--- ════════════════════════════════════════════════════════════════════════
---
--- `engine:stream(session, opts)` returns a coroutine iterator that
--- yields ONE chunk per generation event :
---
---   { kind = "content",  text = "Hello" }
---   { kind = "thinking", text = "let me think…" }   -- reasoning models
---   { kind = "stop",     reason = "stop" | "length" | "stop_string" | ... }
---
--- The consumer prints `content` chunks live for a typewriter effect,
--- routes `thinking` chunks to a side channel (here : a dimmer ANSI
--- colour), and reacts to the final `stop` chunk to print a perf tally.
---
---   ION7_MODEL=/path/to/chat.gguf luajit examples/02_streaming.lua

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

cm:set_system("You are a concise, helpful assistant.")

-- ── ANSI colours for the two output channels ────────────────────────────
local function dim(s)  return "\27[2m" .. s .. "\27[0m" end
local function bold(s) return "\27[1m" .. s .. "\27[0m" end

-- ── Streaming run ────────────────────────────────────────────────────────
local session = llm.Session.new()
session:add_user("In two short sentences, explain what an embedding is.")

io.write(bold("Assistant : "))
io.flush()

for chunk in engine:stream(session, { max_tokens = 128 }) do
    if     chunk.kind == "content"  then
        io.write(chunk.text) ; io.flush()
    elseif chunk.kind == "thinking" then
        -- Reasoning models like Qwen3 emit `<think>...</think>` blocks ;
        -- we route them to a dim side channel so the user sees the
        -- model's working alongside the answer.
        io.write(dim(chunk.text)) ; io.flush()
    elseif chunk.kind == "stop" then
        io.write("\n")
        local resp = session:last_response()
        io.write(string.format("[%s | %s]\n", chunk.reason, resp:summary()))
    end
end

session:add_assistant(session:last_response().content)

cm:release(session)
ctx:free()
model:free()
ion7.shutdown()
