# ion7-llm

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![LuaJIT](https://img.shields.io/badge/LuaJIT-2.1-orange.svg)](https://luajit.org/)
[![ion7-core](https://img.shields.io/badge/ion7--core-1.0-brightgreen.svg)](https://github.com/Ion7-Labs/ion7-core)

**LuaJIT LLM pipeline - chat, streaming, sessions, prefix cache, parallel generation.**

```lua
local llm = require "ion7.llm"
llm.init({ model = "/path/to/model.gguf" })

-- One call, streaming output
local resp = llm.chat("What is LuaJIT?", {
    on_token = function(p) io.write(p); io.flush() end
})
print(resp:summary())  -- [42 tok | 38.2 tok/s | stop]
```

---

## Why ion7-llm?

ion7-core gives you raw primitives: Model, Context, Sampler, Vocab. ion7-llm gives you the generation pipeline on top:

- **Session management** - track message history, KV position, dirty flag
- **Prefix cache** - encode system prompt once, restore in ~15ms instead of ~300ms for every new session
- **Sliding window + attention sink** - on context overflow, oldest messages are evicted; the first N tokens are always preserved (StreamingLLM, ICLR 2024)
- **Message-aligned eviction** - eviction happens at message boundaries, never mid-sentence
- **Coroutine streaming** - `for piece in gen:stream(session) do` - no callbacks required
- **Grammar constraints** - pass a Grammar_obj from ion7-grammar directly into `opts.grammar`
- **True parallel generation** - Scheduler batches N sessions into one `llama_decode()` call per step
- **Think block handling** - strip `<think>...</think>` from reasoning models automatically
- **Eviction hooks** - `on_evict` lets developers plug in summarization, embeddings, or any external memory strategy

---

## Quick start

```bash
git clone https://github.com/ion7-labs/ion7-llm
cd ion7-llm
luajit examples/01_chat.lua  # requires ION7_MODEL
```

```bash
ION7_MODEL=/path/to/model.gguf luajit examples/01_chat.lua
```

---

## Usage

### Single-turn chat

```lua
local llm = require "ion7.llm"
llm.init({ model = MODEL })

local resp = llm.chat("Explain prefix caching in one sentence.")
print(resp.text)
```

### Multi-turn with KV reuse

```lua
local session = llm.Session.new({ system = "Be concise." })
session:add("user", "My name is Louis.")
local r1 = llm.gen():chat(session)
session:add("assistant", r1.text)
session:add("user", "What's my name?")
local r2 = llm.gen():chat(session)
-- KV cache reused - prior turns not re-processed
```

### Streaming

```lua
for piece in llm.stream("Write a haiku about LuaJIT.") do
    io.write(piece); io.flush()
end
```

### Structured generation (requires ion7-grammar)

```lua
local Grammar = require "ion7.grammar"
local g = Grammar.from_type({ name = "string", score = "integer" })

local resp = llm.structured("Extract from: 'Alice scored 95 points.'", g)
print(resp.text)  -- guaranteed valid JSON
```

### Reasoning models (thinking mode)

```lua
llm.init({
    model   = MODEL,
    sampler = "thinking",   -- temperature=0.6, tuned for Qwen3.5/DeepSeek-R1
    think   = true,         -- strip <think> blocks from output
    think_budget = 512,     -- max tokens inside a think block
})

local resp = llm.chat("Solve: 17 × 23")
print(resp.text)         -- clean answer
print(resp:think())      -- raw reasoning trace
```

### Parallel generation

```lua
llm.init({ model = MODEL, n_seq_max = 4 })

local sessions = {}
for i = 1, 4 do
    sessions[i] = llm.Session.new()
    sessions[i]:add("user", "Count to " .. i)
end

llm.batch({
    { session = sessions[1], on_done = function(r) print("[A]", r.text) end },
    { session = sessions[2], on_done = function(r) print("[B]", r.text) end },
    { session = sessions[3], on_done = function(r) print("[C]", r.text) end },
    { session = sessions[4], on_done = function(r) print("[D]", r.text) end },
})
-- All 4 run in parallel - one GPU decode per step, not four
```

### Context injection hook (ion7-rag / ion7-memory)

```lua
llm.cm():set_hook("before_encode", function(messages, session)
    local context = rag:retrieve(messages[#messages].content)
    table.insert(messages, 1, { role = "system", content = context })
    return messages
end)
```

### Eviction hook - custom memory strategy

Called when messages are about to be dropped due to context overflow.
The developer decides what to do: summarize, store in external memory, log, or ignore.

```lua
-- Option A: summarization (keep a compressed trace of evicted content)
llm.cm():set_hook("on_evict", function(evicted, session)
    local text = format_messages(evicted)
    local r = llm.gen():complete("Summarize in 1 sentence: " .. text, { max_tokens = 40 })
    return { { role = "system", content = "[Earlier: " .. r.text .. "]" } }
end)

-- Option B: embedding-based external memory (ion7-embed / RAG)
llm.cm():set_hook("on_evict", function(evicted, session)
    embed_and_store(evicted, session.id)  -- push to vector DB
    return nil  -- drop from context; retrieved later via before_encode
end)

-- Option C: silent drop (default when no hook set)
-- nothing to do
```

### KV quantization (4x VRAM reduction)

```lua
-- q4_0 is lossless on Qwen3.5 - 32k -> ~128k effective context on the same VRAM
llm.init({ model = MODEL, kv_type = "q4_0" })
-- Use kv_type_k / kv_type_v to set K and V independently
```

### Attention sink and eviction strategy

```lua
llm.init({
    model    = MODEL,
    n_sink   = 4,         -- default: 4 - never evict first N tokens
    eviction = "message", -- default: "message" - evict whole messages, not tokens
    -- eviction = "fifo"  -- evict arbitrary oldest tokens (raw FIFO)
})
```

### KV rollback

```lua
local gen = llm.gen()
gen:checkpoint()
local resp = gen:chat(session, { max_tokens = 64 })
if not validator:ok(resp.text) then
    gen:rollback()  -- KV restored - try again with different sampler
end
```

---

## Context management

ion7-llm implements state-of-the-art context overflow handling transparently:

| Model type | Overflow strategy | Notes |
|---|---|---|
| Standard RoPE (Llama 3, Mistral, ...) | Sliding window via `kv_seq_shift` | Oldest messages evicted, positions shifted backward |
| M-RoPE / recurrent (Qwen3.5, Mamba, RWKV) | Hard reset + history re-encode | KV cleared, trimmed history re-encoded from prefix |

Both paths:
- Preserve the system prompt (prefix cache)
- Preserve attention sink tokens (`n_sink`, default 4)
- Evict at message boundaries (`eviction="message"`, default)
- Call the `on_evict` hook before dropping messages

---

## Module structure

```
src/ion7/llm/
├── init.lua             - llm.init(), llm.chat(), llm.stream(), llm.batch()
├── session.lua          - Session: message history, KV position, fork, serialize
├── context_manager.lua  - KV orchestration: prefix cache, sliding window, hooks
├── generator.lua        - Generation loop: stream(), chat(), checkpoint/rollback
├── stop.lua             - Multi-token stop string detection (rolling buffer)
├── response.lua         - Response: text, tokens, stop_reason, perf, think()
├── scheduler.lua        - Parallel generation via llama.cpp batch API
└── sampler_profiles.lua - Named sampler presets: balanced, precise, creative, code, fast, thinking
```

The public API contract is documented in [`spec/PUBLIC_API.md`](spec/PUBLIC_API.md).

---

## Compatibility

| Component | Requirement |
|-----------|-------------|
| LuaJIT | 2.1+ |
| ion7-core | 1.0+ |
| llama.cpp | b8600+ (via ion7-core) |
| OS | Linux, macOS |

### Known constraints

- Grammar sampler must be first in the chain - `Generator` handles this automatically.
- Pass `vocab` table (not `vocab._ptr`) to `:grammar(gbnf, root, vocab)`.
- `llama_sampler_sample()` already calls `accept()` internally - **never call `sampler:accept()` after `sample()`**.
- Sliding window (`kv_seq_shift`) requires RoPE-based models. M-RoPE and recurrent models fall back to hard reset automatically.
- `llm.batch()` requires `n_seq_max >= n_jobs` - set in `llm.init()`.
- KV quantization (`kv_type="q4_0"`) auto-enables Flash Attention.

---

## Roadmap

### Currently available

The following context management techniques are fully implemented:

| Technique | Status | Notes |
|---|---|---|
| Prefix cache (system prompt) | ✅ | ~20x TTFT improvement for multi-session workloads |
| Sliding window + `kv_seq_shift` | ✅ | RoPE models (Llama 3, Mistral, ...) |
| Hard reset + history trimming | ✅ | M-RoPE / recurrent models (Qwen3.5, Mamba, ...) |
| Attention sink (`n_sink`) | ✅ | StreamingLLM - prevents collapse at context boundary |
| Message-aligned eviction | ✅ | No mid-sentence cuts |
| KV quantization (`kv_type`) | ✅ | f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl |
| `on_evict` hook | ✅ | Extension point for summarization, RAG, external memory |
| `before_encode` hook | ✅ | Extension point for ion7-rag / ion7-memory context injection |
| KV checkpoint / rollback | ✅ | Speculative generation with undo |

### Pending llama.cpp integration

The following techniques require llama.cpp to expose attention matrices or other internals not yet available via the public API. They will be integrated into ion7-llm as the underlying support lands:

| Technique | Waiting for | Expected gain |
|---|---|---|
| SnapKV / H2O importance scoring | Attention matrices during prefill | ~50% cache reduction at same quality |
| PyramidKV layer-adaptive budget | Per-layer KV control | ~88% cache reduction |
| LASER-KV anti-greedy selection | Attention matrices + block-wise accumulation | +10% accuracy on 128k+ contexts |
| Cascading 2-tier KV cache | Multiple `seq_id` slots + stable EMA scoring | Exponential context extension at same VRAM |
| TurboQuant 3-bit KV | Merge of TurboQuant into llama.cpp | 4.9x VRAM vs FP16 |
| `kv_seq_shift` on Qwen3.5 | Fix of M-RoPE shift in llama.cpp (#20225) | True sliding window for Qwen3.5 |

---

## 📄 Licensing

ion7-llm is dual-licensed.

### Open Source - AGPLv3

Free to use under the [GNU Affero General Public License v3](LICENSE).

### Commercial License

For proprietary or closed-source use without AGPLv3 obligations.

**-> [Contact for commercial licensing](mailto:5inq@kiji.dev)**
