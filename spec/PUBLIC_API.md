# ion7-llm - Public API

This document is the stability contract for `ion7-llm`.

Everything listed here is **stable**: signature and behaviour will not change without a major version bump.
Everything not listed here is internal and may change at any time.

---

## Module: `ion7.llm`

Entry point. Call `llm.init()` before anything else.

### `llm.init(opts)` -> `llm`

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `opts.model` | `string` | **required** | Path to `.gguf` file |
| `opts.n_gpu_layers` | `number?` | auto-fit | GPU offload layers |
| `opts.n_ctx` | `number?` | auto-fit or 4096 | Context window |
| `opts.n_seq_max` | `number?` | 1 | Max parallel sessions (for `llm.batch()`) |
| `opts.system` | `string?` | nil | Default system prompt (stored in prefix cache) |
| `opts.sampler` | `string?` | `"balanced"` | Sampler profile name or pre-built Sampler |
| `opts.max_tokens` | `number?` | 2048 | Default max tokens per generation |
| `opts.think` | `bool?` | false | Strip `<think>...</think>` blocks from output |
| `opts.think_budget` | `number?` | nil | Max tokens inside a think block before forcing exit |
| `opts.log_level` | `number?` | 0 | 0 = silent |
| `opts.kv_type` | `string?` | `"f16"` | KV cache quantization: `"f16"`, `"q8_0"`, `"q4_0"`, etc. `"q4_0"` is lossless on Qwen3.5 (4x VRAM reduction). Use `kv_type_k`/`kv_type_v` to set K and V independently. |
| `opts.n_sink` | `number?` | 4 | Attention sink: number of first tokens always preserved during sliding window eviction. Set 0 to disable. |
| `opts.eviction` | `string?` | `"message"` | Overflow eviction strategy: `"message"` (evict whole messages from the front, avoids mid-sentence cuts) or `"fifo"` (evict arbitrary oldest tokens). |
| `opts.headroom` | `number?` | auto | Tokens reserved for generation after prefill. Default: `min(max_tokens, n_ctx_seq * 0.25)`. Override if generation regularly hits the context wall. |

### `llm.shutdown()`

Free all resources (Context, Model). Call once at program end.

### Convenience API

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `llm.chat` | `(text, opts?) -> Response` | One-shot blocking chat. `text` is string or `{role,content}[]` |
| `llm.stream` | `(text, opts?) -> iterator` | Coroutine iterator, yields string pieces |
| `llm.structured` | `(text, grammar, opts?) -> Response` | Chat with grammar constraint |
| `llm.stream_structured` | `(text, grammar, opts?) -> iterator` | Stream with grammar constraint |
| `llm.batch` | `(jobs) -> Scheduler` | Parallel generation. `jobs` is array of `{session, sampler?, max_tokens?, on_piece?, on_done?}`. Requires `n_seq_max >= #jobs` |

### Accessors (require `init()` first)

| Symbol | Returns |
|--------|---------|
| `llm.model()` | `Model` - ion7-core Model |
| `llm.vocab()` | `Vocab` - ion7-core Vocab |
| `llm.ctx()` | `Context` - ion7-core Context |
| `llm.cm()` | `ContextManager` |
| `llm.gen()` | `Generator` (default, shared) |

### Sub-modules (loaded at `require "ion7.llm"`)

| Symbol | Type | Notes |
|--------|------|-------|
| `llm.Grammar` | `table?` | `ion7.grammar` if installed, else `nil` |
| `llm.Session` | `table` | `ion7.llm.session` module |
| `llm.Generator` | `table` | `ion7.llm.generator` module |
| `llm.ContextManager` | `table` | `ion7.llm.context_manager` module |
| `llm.Scheduler` | `table` | `ion7.llm.scheduler` module |
| `llm.Stop` | `table` | `ion7.llm.stop` module |
| `llm.Response` | `table` | `ion7.llm.response` module |
| `llm.sampler` | `table` | `ion7.llm.sampler_profiles` module |

**`opts` for `llm.chat` / `llm.stream`:**

| Key | Type | Notes |
|-----|------|-------|
| `on_token` | `function?` | Called with each decoded piece (chat only) |
| `max_tokens` | `number?` | Override max tokens |
| `sampler` | `cdata?` | Override sampler |
| `grammar` | `Grammar_obj\|string?` | Grammar_obj (ion7-grammar) or raw GBNF |
| `think_budget` | `number?` | Override think_budget for this call |

---

## `ion7.llm.response` - Response

Returned by every generation call.

### Fields

| Field | Type | Notes |
|-------|------|-------|
| `text` | `string` | Full generated text. Think blocks stripped if `think=true`. |
| `tokens` | `table` | Array of token IDs. |
| `n_tokens` | `number` | `#tokens`. |
| `stop_reason` | `string` | `"stop"` \| `"length"` \| `"stop_string"` \| `"error"` |
| `perf` | `table` | `{ tok_per_s, n_eval, t_eval_ms, n_p_eval }` |

### Methods

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Response:think` | `() -> string?` | Raw `<think>...</think>` content. `nil` if `think=false` or no think block. |
| `Response:summary` | `() -> string` | One-line `[N tok | X tok/s | stop_reason]` |
| `tostring(r)` | `string` | Returns `r.text` |

---

## `ion7.llm.session` - Session

Tracks one logical conversation: message history, KV position, snapshot.

### `Session.new(opts?)` -> `Session`

| Param | Type | Notes |
|-------|------|-------|
| `opts.system` | `string?` | System prompt (prepended to `all_messages()`). Managed by ContextManager. |
| `opts.seq_id` | `number?` | KV sequence ID. `nil` = assigned by ContextManager on first use. |

### Instance methods

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Session:add` | `(role, content) -> self` | Append a message. Sets `_dirty = true`. Fluent. |
| `Session:all_messages` | `() -> table` | All messages including system (if set). |
| `Session:pending_messages` | `() -> table` | The `messages` array (same as `self.messages`). |
| `Session:has_snapshot` | `() -> bool` | True if a KV snapshot exists. |
| `Session:snapshot` | `() -> string?` | Raw KV blob from `ctx:snapshot()`. |
| `Session:reset` | `() -> self` | Clear messages, n_past, snapshot, dirty flag. |
| `Session:fork` | `() -> Session` | Copy history + snapshot into a new independent session. `seq_id = nil`. |
| `Session:format` | `(msgs?) -> string` | Format messages as plain text (`"role: content"` per line). `msgs` defaults to `self.messages`. Useful for building summarization prompts inside `on_evict` hooks. |
| `Session:serialize` | `() -> table` | Plain table (id, system, messages, n_past). Snapshot excluded. |
| `Session:save` | `(path) -> bool, err?` | Save serialized session to JSON file (requires `dkjson` or `cjson`). |

### Static

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Session.deserialize` | `(t) -> Session` | Restore from serialized table. KV must be re-prefilled. `_dirty = true`. |
| `Session.load` | `(path) -> Session?, err?` | Load from JSON file. |

### Internal (called by ContextManager/Generator, not user code)

| Symbol | Notes |
|--------|-------|
| `Session:_save_snapshot(blob, n_past)` | Store KV blob + position. Clears `_dirty`. |

### Fields

| Field | Type | Notes |
|-------|------|-------|
| `id` | `number` | Unique, auto-increment. |
| `seq_id` | `number?` | KV sequence slot. `nil` until ContextManager assigns. |
| `system` | `string?` | System prompt. |
| `messages` | `table` | `{role, content}` array. |
| `n_past` | `number` | KV cache fill position after last prefill. |
| `_dirty` | `bool` | True when messages added since last prefill. |
| `_msg_kv_ends` | `table` | Internal: absolute KV end positions for each encoded message. Populated by ContextManager after prefill. Used for message-aligned eviction. |

---

## `ion7.llm.context_manager` - ContextManager

KV cache orchestration: prefix cache, session slots, sliding window, hooks.

### `ContextManager.new(ctx, vocab, opts?)` -> `ContextManager`

| Param | Type | Notes |
|-------|------|-------|
| `ctx` | `Context` | ion7-core Context |
| `vocab` | `Vocab` | ion7-core Vocab |
| `opts.max_sessions` | `number?` | Max concurrent sessions. Default: `ctx:n_seq_max()`. |
| `opts.headroom` | `number?` | Tokens reserved for generation after prefill. Default: 256. |
| `opts.n_sink` | `number?` | Attention sink: always-preserved first N tokens. Default: 4. Set 0 to disable. |
| `opts.eviction` | `string?` | Overflow eviction strategy: `"message"` (default) or `"fifo"`. |

### Methods

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ContextManager:set_system` | `(text) -> number` | Encode system prompt, save KV snapshot. Returns token count. |
| `ContextManager:has_prefix` | `() -> bool` | True if prefix cache snapshot exists. |
| `ContextManager:restore_prefix` | `() -> bool` | Restore prefix snapshot. Returns false if no cache. |
| `ContextManager:prepare` | `(session) -> number` | Prepare KV for a session's next decode. Returns n_past. |
| `ContextManager:release` | `(session)` | Free this session's KV slot. Call when session is done. |
| `ContextManager:fork` | `(session) -> Session` | Fork session. Copies KV via `kv_seq_cp`. |
| `ContextManager:stats` | `() -> table` | `{ slots_total, slots_free, prefix_tokens, prefix_cached, n_sink, eviction, n_evictions, n_tokens_evicted }` |
| `ContextManager:set_hook` | `(name, fn)` | Install a named hook. See hooks below. |
| `ContextManager:clear_hook` | `(name)` | Remove a hook. |
| `ContextManager:get_hook` | `(name) -> function?` | Return the hook function or nil. |

### Hooks

| Name | Signature | Notes |
|------|-----------|-------|
| `"before_encode"` | `fn(messages, session) -> messages?` | Called before tokenizing messages. Return modified array (or nil to keep original). Used by ion7-rag / ion7-memory to inject retrieved context. |
| `"on_evict"` | `fn(evicted_messages, session) -> messages?` | Called when messages are dropped due to context overflow (both sliding window and hard reset paths). `evicted_messages` is a `{role,content}[]` array about to be lost. Return replacement messages (e.g. a summary) to insert in their place, or `nil` to drop silently. If the replacement doesn't fit in the available space it is silently ignored. Extension point for EpiCache-style summarization, embedding-based memory (ion7-embed), or any custom eviction strategy. |

### `stats()` fields

| Field | Type | Notes |
|-------|------|-------|
| `slots_total` | `number` | Max concurrent sessions. |
| `slots_free` | `number` | Currently available session slots. |
| `prefix_tokens` | `number` | Token count of the cached system prompt. 0 if no prefix. |
| `prefix_cached` | `bool` | True if a prefix KV snapshot exists. |
| `n_sink` | `number` | Attention sink size (first N tokens always preserved). |
| `eviction` | `string` | Active eviction strategy: `"message"` or `"fifo"`. |
| `n_evictions` | `number` | Cumulative number of overflow events where tokens were dropped. |
| `n_tokens_evicted` | `number` | Cumulative token count removed across all eviction events. |

### Overflow strategy

When the context fills up, ContextManager chooses:
1. **Sliding window** (`ctx:kv_can_shift() = true`): remove oldest content, shift positions. Respects `n_sink` (first N tokens never evicted) and `eviction` strategy.
2. **Hard reset** (Mamba, RWKV, Qwen3.5 M-RoPE): restore prefix or `kv_clear()`, then re-encode trimmed message history.

**Eviction strategies (for sliding window path):**
- `"message"` (default): evict whole messages from the front. Avoids mid-sentence cuts. Requires message position tracking stored in `session._msg_kv_ends`.
- `"fifo"`: evict arbitrary oldest tokens (half the movable window). Simpler, slightly less coherent.

**Attention sink (`n_sink`, default 4):** The first `n_sink` tokens after the system prompt are never evicted. LLMs disproportionately attend to these tokens regardless of content (StreamingLLM, ICLR 2024); evicting them causes performance collapse.

---

## `ion7.llm.generator` - Generator

Generation loop: streaming, grammar constraints, KV checkpoints.

### `Generator.new(ctx, vocab, cm, opts?)` -> `Generator`

| Param | Type | Notes |
|-------|------|-------|
| `ctx` | `Context` | |
| `vocab` | `Vocab` | |
| `cm` | `ContextManager` | |
| `opts.sampler` | `cdata?` | Pre-built Sampler. Mutually exclusive with `opts.sampler_opts`. |
| `opts.sampler_opts` | `table?` | `{ temp, top_k, top_p, min_p, seed }` |
| `opts.grammar` | `Grammar_obj\|string?` | Default grammar for every generation |
| `opts.max_tokens` | `number?` | Default: 2048 |
| `opts.stop` | `table?` | Extra stop strings |
| `opts.think` | `bool?` | Strip `<think>` blocks. Default: false |
| `opts.think_budget` | `number?` | Max tokens inside a think block. Default: nil (unlimited) |

### Methods

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Generator:stream` | `(session, opts?) -> iterator` | Coroutine iterator, yields string pieces. |
| `Generator:chat` | `(session, opts?) -> Response` | Blocking: consume stream, return Response. |
| `Generator:complete` | `(prompt, opts?) -> Response` | Wrap raw string in a Session and chat. |
| `Generator:checkpoint` | `() -> self` | Save KV snapshot for rollback. |
| `Generator:rollback` | `() -> self` | Restore to last `checkpoint()`. Requires prior call to `checkpoint()`. |

**Per-call `opts`:**

| Key | Type | Notes |
|-----|------|-------|
| `on_token` | `function?` | Called per piece (chat only) |
| `max_tokens` | `number?` | Override |
| `sampler` | `cdata?` | Override sampler |
| `grammar` | `Grammar_obj\|string?` | Override grammar |
| `think_budget` | `number?` | Override think_budget |

---

## `ion7.llm.stop` - Stop

Multi-token stop string detection via rolling buffer.

### `Stop.new(opts?)` -> `Stop`

| Param | Type | Notes |
|-------|------|-------|
| `opts.strings` | `table?` | Custom stop strings. Replaces defaults if provided. |
| `opts.extra` | `table?` | Additional stop strings merged with defaults. |
| `opts.buf_size` | `number?` | Rolling buffer size in chars. Default: 256. |

### Methods

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Stop:feed` | `(piece) -> string?, number?` | Feed decoded piece. Returns matched stop string + offset, or nil. |
| `Stop:text_before` | `(stop) -> string` | Text in buffer before the matched stop. Call after feed() returns a match. |
| `Stop:reset` | `()` | Clear rolling buffer. Call at start of each generation. |
| `Stop:add` | `(s)` | Add a stop string at runtime. |
| `Stop:list` | `() -> table` | All registered stop strings (sorted longest-first). |

### Default stop strings

ChatML `<|im_end|>`, Llama-3 `<|eot_id|>` / `<|end_of_turn|>`, Gemma `<end_of_turn>`, generic `</s>` / `[INST]` / `[/INST]`, DeepSeek `<｜end▁of▁sentence｜>`.

---

## `ion7.llm.scheduler` - Scheduler

True parallel generation: N sessions share a single GPU decode per step.

### `Scheduler.new(ctx, vocab, cm)` -> `Scheduler`

### Methods

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Scheduler:submit` | `(session, opts) -> self` | Queue a session for generation. |
| `Scheduler:run` | `() -> self` | Run all jobs until completion. Blocks. |
| `Scheduler:n_jobs` | `() -> number` | Pending (not-yet-run) job count. |

**`opts` for `submit()`:**

| Key | Type | Notes |
|-----|------|-------|
| `sampler` | `cdata` | **required** - Sampler for this session |
| `max_tokens` | `number?` | Default: 2048 |
| `stop` | `table?` | Extra stop strings |
| `on_piece` | `function?` | Called with each decoded piece |
| `on_done` | `function?` | Called with `Response` when done |

### Architecture

Phase 1 - Sequential prefill (fast with prefix cache).
Phase 2 - Batched generation: 1 token/session per `llama_decode()` call. All sessions run in parallel on the GPU.

Use `Scheduler` when you have 2+ sessions. For single sessions, use `Generator`.

---

## `ion7.llm.sampler_profiles`

Named Sampler chains for common use cases.

| Profile | `temp` | Notes |
|---------|--------|-------|
| `balanced` | 0.8 | Default. Sensible for most tasks. |
| `precise` | 0.3 | Low temperature. Factual Q&A. |
| `creative` | 1.1 | Higher temperature. Relaxed constraints. |
| `code` | 0.2 | Low temperature + repetition penalty. |
| `fast` | - | Greedy. Fully deterministic, maximum speed. |
| `thinking` | 0.6 | Qwen3.5 / DeepSeek-R1 reasoning models. Use with `opts.think = true`. |

All profiles take a `vocab` argument and return a built Sampler:
```lua
local sampler = require("ion7.llm.sampler_profiles").thinking(vocab)
```

---

---

## Compatibility

| Component | Requirement |
|-----------|-------------|
| LuaJIT | 2.1+ |
| ion7-core | 1.0+ |
| llama.cpp | b8600+ (via ion7-core) |
| OS | Linux, macOS |

## Known constraints

- Grammar samplers must be first in the sampler chain - enforced by `_grammar_sampler()`.
- Pass `vocab` table (not `vocab._ptr`) to `:grammar(gbnf, root, vocab)`.
- `llama_sampler_sample()` on a chain already calls `accept()` internally - **never call `sampler:accept()` separately**.
- `kv_seq_shift()` is not supported on recurrent or M-RoPE models (Mamba, RWKV, Qwen3.5). ContextManager falls back to hard reset automatically via `ctx:kv_can_shift()`.
- `Scheduler` requires `n_seq_max >= n_jobs`. Set this in `llm.init({ n_seq_max = N })`.
- KV quantization (`kv_type="q4_0"`) requires Flash Attention - enabled automatically by ion7-core when quantized KV types are requested.
- Message-aligned eviction (`eviction="message"`) tracks per-message KV positions at prefill time. On the first call after init or after a hard reset, positions are recomputed. The tracking uses per-message single-message template tokenization as an approximation; boundaries are accurate to within a few tokens.
