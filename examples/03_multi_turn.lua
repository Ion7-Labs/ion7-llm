#!/usr/bin/env luajit
--- @example examples.03_multi_turn
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 03 — Multi-turn conversation : KV reuse via per-seq snapshot
--- ════════════════════════════════════════════════════════════════════════
---
--- A four-turn dialogue. Between turns we call `session:add_assistant`
--- so the chat template sees the full history on the next render. The
--- KV layer's snapshot fast path means each subsequent prepare() does
--- NOT re-decode the whole conversation — only the new user turn is
--- appended on top of the snapshot.
---
--- The script also prints the running `cm:stats()` so you can watch
--- the slot pool stay full and the prefix cache stay hot.
---
---   ION7_MODEL=/path/to/chat.gguf luajit examples/03_multi_turn.lua

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

cm:set_system("You are a concise tutor. Reply in one short sentence.")

local session = llm.Session.new()

-- Helper : run one turn, print the user/assistant exchange + stats.
local function turn(question)
    session:add_user(question)
    local r = engine:chat(session, { max_tokens = 60 })
    session:add_assistant(r.content, { thinking = r.thinking })

    local s = cm:stats()
    print(string.format(
        "\n\27[1mU\27[0m %s\n\27[1mA\27[0m %s\n   [%s | n_past=%d | %d/%d slots free | prefix=%d tok]",
        question, r.content, r:summary(), session.n_past,
        s.slots_free, s.slots_total, s.prefix_tokens))
end

turn("What is RAM ?")
turn("And what is ROM ?")
turn("Which one keeps its contents when power is off ?")
turn("Give me one practical example of each.")

-- The four turns share the prefill of the system prompt + previous
-- turns. Per-seq snapshot bookkeeping kept us off the slow path on
-- every turn except the first.
print(string.format(
    "\n[end] %d messages stored, n_past = %d / n_ctx = %d",
    #session.messages, session.n_past, ctx:n_ctx()))

cm:release(session)
ctx:free()
model:free()
ion7.shutdown()
