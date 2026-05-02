# ion7-llm examples

Nine standalone scripts that walk every layer of the public API. Each
one is runnable from the project root with no extra setup beyond a
model path.

## Setup

```bash
export ION7_MODEL=/path/to/chat.gguf       # required for examples 01-08
export ION7_EMBED=/path/to/embed.gguf      # required for example 09
export ION7_GPU_LAYERS=99                  # optional, override n_gpu_layers
```

ion7-llm depends on `ion7.core`. The examples expect the in-tree
`src/?.lua` to find both, so run them from the **project root** :

```bash
luajit examples/01_chat.lua
```

If you are working off a built ion7-core dist tarball, point
`ION7_LIBLLAMA_PATH`, `ION7_LIBGGML_PATH` and `ION7_BRIDGE_PATH` at
the bundled `.so` files (or use the dist's `bin/ion7-load.lua`
preamble) so the FFI loader finds them.

## What each example shows

| File | Topic | Concepts demonstrated |
|------|-------|----------------------|
| [01_chat.lua](01_chat.lua) | minimal pipeline | `llm.pipeline`, `Session`, `engine:chat`, `Response` |
| [02_streaming.lua](02_streaming.lua) | typed-chunk stream | `engine:stream`, content / thinking / `tool_call_*` / stop chunks |
| [03_multi_turn.lua](03_multi_turn.lua) | conversation history | KV reuse via per-seq snapshot, `cm:stats()` |
| [04_pool.lua](04_pool.lua) | concurrent sessions | `Pool`, one-batch-per-tick across N slots, ~6× aggregate speed-up |
| [05_grammar.lua](05_grammar.lua) | structured output | `ion7-grammar` `from_type` / `from_json_schema` → constrained sampler |
| [06_tools.lua](06_tools.lua) | tool calling | `Tool`, `ToolSet`, `tools.loop.run`, interleaved-thinking dispatch |
| [07_thinking.lua](07_thinking.lua) | reasoning models | `opts.thinking`, `opts.think_budget`, `<think>` demux |
| [08_persistence.lua](08_persistence.lua) | save / load | JSON history + per-seq KV file round-trip |
| [09_embed.lua](09_embed.lua) | embeddings | `Embed`, cosine similarity, nearest-neighbour query |
| [10_radix.lua](10_radix.lua) | exact-match prefix cache | `kv.radix`, warm-start identical prompts, LRU stats |

## Reading order

The progression is incremental — each file assumes the patterns of
the previous one without re-explaining. If you only have time for
three, read **01**, **02** and **05** : minimal chat, streaming, and
constrained output cover the typical happy path.

## Contracts every example follows

- Reads paths from env vars only ; no hardcoded fallbacks.
- Calls `ion7.init({ log_level = 0 })` / `ion7.shutdown()` at the
  top and bottom.
- Frees its `Engine`, `Pool`, `Embed`, `Context`, `Model` before
  exit (via `cm:release` + `ctx:free` + `model:free`).
- Defaults `n_gpu_layers = 0` so every script runs on CPU-only
  laptops, but honours `ION7_GPU_LAYERS` when set.
- Uses the model's embedded chat template via `cm:set_system` +
  `Session:add_user` / `add_assistant` rather than hand-rolling the
  prompt format.

## Notes on small models

The example output we ship with these scripts was produced against a
3 B-parameter chat model. A 3 B model is enough to demonstrate the
plumbing, but it is sometimes too small to follow tool-call protocols
reliably (example 06) or to fill a complex JSON Schema with rich data
(example 05). Re-running with a 7-30 B chat model gives noticeably
better output without changing any code.

## A handful of one-liners

Drop these into a Lua REPL after `package.path = "./src/?.lua;./src/?/init.lua;" .. package.path` to explore the API interactively.

```lua
local llm  = require "ion7.llm"

-- Smallest possible call : engine + session + chat.
local cm, engine = llm.pipeline(ctx, vocab)
print(engine:complete("hi", { max_tokens = 12 }).content)

-- Streaming generator for a typewriter UI.
for chunk in engine:stream(session, { max_tokens = 64 }) do
    if chunk.kind == "content" then io.write(chunk.text) ; io.flush() end
end

-- Schema-constrained one-shot.
local r = engine:complete("Describe Linus Torvalds.", {
    schema = { type = "object",
               properties = { name = { type = "string" }, role = { type = "string" } },
               required = { "name", "role" } },
})
```
