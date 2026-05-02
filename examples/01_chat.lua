#!/usr/bin/env luajit
--- @example examples.01_chat
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 01 — Hello, chat : minimal pipeline + Session + engine:chat
--- ════════════════════════════════════════════════════════════════════════
---
--- The minimum viable use of ion7-llm : load a model, build the
--- pipeline (context manager + engine), drop a user turn into a
--- Session, get a Response back. Eight steps, ~30 lines of actual code.
---
---   ION7_MODEL=/path/to/chat.gguf luajit examples/01_chat.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
local llm  = require "ion7.llm"

-- ── 1. Bring the backend up ──────────────────────────────────────────────
ion7.init({ log_level = 0 })

-- ── 2. Load the model ; default to CPU-only for portability. ─────────────
local n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS")) or 0
local model = ion7.Model.load(MODEL, { n_gpu_layers = n_gpu_layers })
print(string.format("[loaded] %s", model:desc()))

-- ── 3. Context + vocab + pipeline. `llm.pipeline` builds the
--      ContextManager and the Engine in one call. ──────────────────────
local ctx   = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
local vocab = model:vocab()
local cm, engine = llm.pipeline(ctx, vocab, { headroom = 256 })

-- ── 4. Set the system prompt. The cm encodes it once into the prefix
--      slot when the context has spare seqs ; otherwise it falls back
--      to text-only mode and prepends the system message at render time.
cm:set_system("You are a concise, helpful assistant.")

-- ── 5. Build a session and add the user turn. The session carries
--      conversation history + per-seq KV bookkeeping. ─────────────────
local session = llm.Session.new()
session:add_user("Explain LuaJIT in one sentence.")

-- ── 6. Run the chat ; the engine handles prefill, sampling, halt
--      detection, and post-parse demux of content / thinking / tools.
local response = engine:chat(session, { max_tokens = 96 })

-- ── 7. Show the result. Response is a value object — just read fields.
io.write("Assistant : " .. response.content .. "\n")
io.write("(" .. response:summary() .. ")\n")

-- ── 8. Persist the assistant turn so a follow-up call would see the
--      full history. Then tear down. ──────────────────────────────────
session:add_assistant(response.content, { thinking = response.thinking })

cm:release(session)
ctx:free()
model:free()
ion7.shutdown()
