# ion7-llm examples

Progressive examples from minimal to advanced.

## Setup

```bash
cd /path/to/ion7-llm

export ION7_MODEL=/path/to/model.gguf
export ION7_CORE_PATH=../ion7-core   # default: ../ion7-core
```

Run from the **project root**:

```bash
luajit examples/01_chat.lua
```

## Examples

| File | Concepts | Difficulty |
|------|----------|------------|
| [01_chat.lua](01_chat.lua) | `llm.init`, `llm.chat`, streaming, `resp:summary()` | ★☆☆☆ |
| [02_sessions.lua](02_sessions.lua) | `Session`, multi-turn KV reuse, `fork`, serialize/deserialize | ★★☆☆ |
| [03_grammar.lua](03_grammar.lua) | `llm.structured`, `llm.stream_structured`, `from_type`, `from_json_schema`, `from_enum`, `GrammarContext`, tool-call grammar | ★★☆☆ |
| [04_multi_session.lua](04_multi_session.lua) | `llm.gen`, `llm.batch`, Generator vs Scheduler, routing logic | ★★★☆ |
| [05_thinking.lua](05_thinking.lua) | `think=true`, `think_budget`, `resp:think()`, `checkpoint/rollback`, "thinking" profile | ★★★☆ |
| [06_eviction_hook.lua](06_eviction_hook.lua) | `on_evict` hook, `Session:format()`, summarization, silent drop + logging, eviction stats | ★★★☆ |

## What each example shows

**01 - Chat**: the minimal pipeline. Init, call, print, shutdown. Shows streaming and the `resp:summary()` one-liner.

**02 - Sessions**: the core of multi-turn. `Session:add()` + `gen:chat(session)` reuses the KV cache across turns - the model only processes new tokens. `fork()` creates independent branches from any point. `serialize/deserialize` for cross-session persistence.

**03 - Grammar**: how ion7-llm integrates with ion7-grammar. Every output is guaranteed to match the grammar - enforced token-by-token on the GPU. Shows 7 patterns: enum, JSON Schema, type annotation, `opts.grammar`, streaming, `GrammarContext`, tool calls.

**04 - Multi-session**: `llm.batch()` routes to the Scheduler when N ≥ 2 sessions. At each generation step, the Scheduler sends 1 token per session in a single `llama_decode()` call. All sessions run in parallel - not interleaved coroutines.

**05 - Thinking**: reasoning models (Qwen3.5, DeepSeek-R1, Phi-4). `think=true` strips `<think>...</think>` from `resp.text`. `resp:think()` returns the raw trace. `think_budget` caps the thinking phase. `checkpoint/rollback` saves and restores the KV cache for speculative generation.

**06 - Eviction hook**: how to intercept context overflow events. Option A: compress evicted messages into a 1-sentence summary via `on_evict` + `gen:complete()` - keeps the conversation coherent across context resets. Option B: silent drop with an audit log - zero inference cost, plug into external memory (ion7-embed / RAG) via the log. Shows `Session:format()`, eviction stats (`n_evictions`, `n_tokens_evicted`), and the `[Earlier: ...]` injection pattern.

## Notes

- Example 03 requires `ion7-grammar` at `../ion7-grammar` (or set `ION7_GRAMMAR`).
- All examples require `ION7_MODEL` to be set.
- `n_seq_max` in example 04 must match the number of batch jobs.
- Example 06: set `N_CTX=512` to trigger overflow quickly and observe the eviction hook firing.
