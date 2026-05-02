#!/usr/bin/env luajit
--- @example examples.07_thinking
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 07 — Reasoning models : thinking demux + reasoning budget
--- ════════════════════════════════════════════════════════════════════════
---
--- Modern reasoning models (Qwen3, DeepSeek-R1, Magistral, etc.) emit
--- a `<think>...</think>` block before their visible reply. The
--- engine plumbs this through several layers :
---
---   - `chat.thinking` is the per-token state machine that splits the
---     stream into `content` and `thinking` chunks at tag boundaries.
---   - `opts.thinking = true / false` toggles the chat template's
---     `enable_thinking` flag (when the template supports it).
---   - `opts.think_budget = N` caps the tokens spent inside a
---     `<think>` block ; once the budget is exhausted the engine
---     force-closes the block and lets the model emit the answer.
---
--- A non-thinking model simply ignores the toggles — the streaming
--- channels stay on `content` only. With Qwen3 / R1 you should see
--- the dimmed reasoning text appear before the answer.
---
---   ION7_MODEL=/path/to/qwen3-or-r1.gguf luajit examples/07_thinking.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
local llm  = require "ion7.llm"

ion7.init({ log_level = 0 })

local model = ion7.Model.load(MODEL, {
    n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS")) or 0,
})
local ctx   = model:context({ n_ctx = 8192, n_seq_max = 1, n_threads = 4 })
local vocab = model:vocab()

local cm, engine = llm.pipeline(ctx, vocab, {
    headroom        = 256,
    -- The `thinking` profile widens top_p and lowers the temperature
    -- moderately — a reasonable default for reasoning models. Pair
    -- with `opts.thinking = true` per call.
    default_sampler = llm.sampler.profiles.thinking(),
})

cm:set_system("You are a careful reasoner. Show your working.")

-- ── ANSI helpers ─────────────────────────────────────────────────────────
local function dim(s)  return "\27[2m" .. s .. "\27[0m" end
local function bold(s) return "\27[1m" .. s .. "\27[0m" end

-- ── Streaming with thinking + budget ─────────────────────────────────────

local session = llm.Session.new()
session:add_user("If 7 birds sit on a wire and 3 fly away, how many remain ? Reason briefly, then answer.")

io.write(bold("Streaming response :\n"))
local in_thinking = false
for chunk in engine:stream(session, {
    thinking     = true,    -- request reasoning when the template supports it
    think_budget = 256,     -- cap reasoning at 256 tokens
    max_tokens   = 384,
}) do
    if     chunk.kind == "thinking" then
        if not in_thinking then io.write(dim("\n[thinking] ")) ; in_thinking = true end
        io.write(dim(chunk.text)) ; io.flush()
    elseif chunk.kind == "content" then
        if in_thinking then io.write("\n" .. bold("[answer] ")) ; in_thinking = false end
        io.write(chunk.text) ; io.flush()
    elseif chunk.kind == "stop" then
        io.write("\n")
        local r = session:last_response()
        io.write(string.format("\n%s\n", r:summary()))
        if r.thinking and #r.thinking > 0 then
            io.write(string.format("(thinking trace : %d chars)\n", #r.thinking))
        else
            io.write("(no thinking trace — non-reasoning model or template did not emit one)\n")
        end
    end
end

cm:release(session)
ctx:free()
model:free()
ion7.shutdown()
