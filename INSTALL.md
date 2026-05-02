# Installing ion7-llm

ion7-llm is a pure-Lua library that depends on
[`ion7-core`](https://github.com/Ion7-Labs/ion7-core). No native code,
no compilation step — it's literally a `src/ion7/llm/` tree of
`.lua` files plus tests and examples.

That means installation reduces to two questions :

1. **Where is ion7-core ?** ion7-llm has to find both its Lua sources
   AND its native libraries (`libllama.so`, optionally
   `ion7_bridge.so`).
2. **Where is your model ?** The chat-suite tests and most examples
   read a path from the `ION7_MODEL` env var.

The rest is path bookkeeping. Three layouts cover most setups.

---

## Layout A : sibling source checkouts (developer mode)

This is the layout the test suite assumes by default. Two checkouts
live next to each other under any parent directory :

```
my-workspace/
├── ion7-core/
│   ├── src/ion7/core/...
│   ├── vendor/llama.cpp/build/bin/libllama.so
│   ├── bridge/build/ion7_bridge.so
│   └── ...
└── ion7-llm/
    ├── src/ion7/llm/...
    ├── tests/
    └── examples/
```

Workflow :

```bash
cd my-workspace/
git clone --recurse-submodules https://github.com/Ion7-Labs/ion7-core
git clone https://github.com/Ion7-Labs/ion7-llm

cd ion7-core
make build                                # builds llama.cpp + bridge

cd ../ion7-llm
ION7_MODEL=/path/to/chat.gguf bash tests/run_all.sh
```

The test helper at [`tests/helpers.lua`](tests/helpers.lua) walks up
the directory tree and prepends `../ion7-core/src` (and
`../../ion7-core/src`) onto `package.path` automatically. No env vars
needed — `require "ion7.core"` and `require "ion7.llm"` both resolve.

If you keep ion7-core in a non-sibling location, set `ION7_CORE_SRC`
to the absolute path of its `src/` directory :

```bash
export ION7_CORE_SRC=/opt/ion7-core/src
ION7_MODEL=... bash tests/run_all.sh
```

---

## Layout B : ion7-core release tarball + cloned ion7-llm

Pre-built bundles ship with [`ion7-core`
releases](https://github.com/Ion7-Labs/ion7-core/releases). Each
tarball includes the Lua runtime AND the platform-specific shared
libraries.

```bash
# 1. Download an ion7-core release.
curl -L -o ion7.tgz \
  https://github.com/Ion7-Labs/ion7-core/releases/latest/download/ion7-core-linux-x86_64-cpu.tar.gz
tar xzf ion7.tgz
export ION7_CORE_DIR=$PWD/ion7-core-*/

# 2. Clone ion7-llm.
git clone https://github.com/Ion7-Labs/ion7-llm
cd ion7-llm

# 3. Wire the paths : tarball ships its own loader preamble that does
#    package.path + ION7_LIBLLAMA_PATH + ION7_BRIDGE_PATH in one call.
export ION7_CORE_SRC="$ION7_CORE_DIR/src"

# 4. Run the tests.
ION7_MODEL=/path/to/chat.gguf \
ION7_LIBLLAMA_PATH=$ION7_CORE_DIR/lib/libllama.so \
ION7_LIBGGML_PATH=$ION7_CORE_DIR/lib/libggml.so \
ION7_BRIDGE_PATH=$ION7_CORE_DIR/lib/ion7_bridge.so \
bash tests/run_all.sh
```

Available tarball targets : Linux x86_64 (CPU, Vulkan), Linux aarch64
(CPU), macOS arm64 (Metal), macOS x86_64 (CPU). For Windows / CUDA /
ROCm see [`ion7-core/INSTALL.md`](https://github.com/Ion7-Labs/ion7-core/blob/main/INSTALL.md).

---

## Layout C : luarocks + system-installed ion7-core

Once ion7-core ships a luarocks rockspec (planned), this becomes the
production setup :

```bash
luarocks install ion7-core
luarocks install ion7-llm
```

With both modules on the system `package.path`, your script just
imports them :

```lua
local ion7 = require "ion7.core"
local llm  = require "ion7.llm"

ion7.init()

local model    = ion7.Model.load(os.getenv("ION7_MODEL"), { n_gpu_layers = 99 })
local cm, eng  = llm.pipeline(model:context({ n_ctx = 8192 }), model:vocab())
cm:set_system("You are a concise assistant.")
local s = llm.Session.new() ; s:add_user("hi")
print(eng:chat(s).content)
```

---

## Environment variables

Read by the test suite, the example scripts, and (for the native ones)
ion7-core's FFI loader.

| Variable           | Used by | Purpose |
|--------------------|---------|---------|
| `ION7_MODEL`       | tests, examples | Chat-tuned GGUF path. Required for everything model-dependent. |
| `ION7_EMBED`       | embed tests + 09_embed.lua | Path to an embedding-tuned GGUF. |
| `ION7_DRAFT`       | speculative experiments | Optional draft model for spec-decoding. |
| `ION7_GPU_LAYERS`  | tests, examples | Override `n_gpu_layers` (default 0 = pure CPU). |
| `ION7_CORE_SRC`    | helpers.lua | Absolute path to ion7-core's `src/` directory when not a sibling. |
| `ION7_LIBLLAMA_PATH` | ion7-core FFI loader | Pin a specific `libllama.so` (otherwise system search). |
| `ION7_LIBGGML_PATH`  | ion7-core FFI loader | Pin a specific `libggml.so`. |
| `ION7_BRIDGE_PATH`   | ion7-core FFI loader | Pin a specific `ion7_bridge.so`. |
| `ION7_SKIP`        | tests/run_all.sh | Whitespace-separated list of files to skip. |

`ION7_MODEL`, `ION7_EMBED`, `ION7_DRAFT` and `ION7_LIB*` come from
[`feedback_no_hardcoded_fallbacks`](spec/) — every script that needs
a path reads it from the env, never from a built-in default.

---

## Compatibility matrix

| Component   | Requirement                                          |
|-------------|------------------------------------------------------|
| LuaJIT      | 2.1 (any post-2017 build)                            |
| ion7-core   | matched release ; the public-API spec lives in       |
|             | [`spec/PUBLIC_API.md`](spec/PUBLIC_API.md)           |
| OS          | whatever ion7-core builds on (Linux glibc, macOS 12+) |
| Models      | any chat-tuned GGUF llama.cpp can load               |
| Templates   | the model's embedded chat template (Jinja2 via the bridge) |

ion7-llm itself has no platform-specific code, so the OS / arch
matrix is dictated entirely by ion7-core. If ion7-core builds on
your machine, ion7-llm will run.

---

## Troubleshooting

### `ion7.core failed to load` from the tests

The bootstrap could not find the libraries. Confirm that one of these
holds :

- `ION7_CORE_SRC` is set and points at a directory containing
  `ion7/core/init.lua`.
- A sibling `../ion7-core/src/` checkout exists and is built (the
  helper probes there automatically).
- `ION7_LIBLLAMA_PATH` points at a real `.so` file (use
  `ldd $ION7_LIBLLAMA_PATH` to verify it resolves).

### `chat template failed (rc=-2)`

The chat-template render rejected the message list. Most often this
fires when a session contains a malformed `assistant` message
(missing `content`, mismatched role alternation). Run
[`02_response_session.lua`](tests/02_response_session.lua) to confirm
the helpers behave, then inspect the offending session's `messages`
array. ion7-llm's `_compute_msg_positions` already guards against
templates that reject single-message renders ; if you hit this in
`engine:chat`, it usually means the message **content** itself is
malformed.

### `seq_cp() is only supported for full KV buffers`

Multi-session paths (`Pool`, `cm:fork`, the prefix cache) require a
context built with `kv_unified = true` AND `n_seq_max > 1`. The
test-suite helper does this automatically ; if you build your own
context, pass both flags :

```lua
local ctx = model:context({
    n_ctx      = 4096,
    n_seq_max  = 4,
    kv_unified = true,
})
```

### `[ion7.llm.kv] no free seq slot`

You reached `n_seq_max` concurrent sessions. Either bump `n_seq_max`
on the context, or release sessions you are done with via
`cm:release(session)`. In tests, every test body wraps its session in
a `with_session` / `cm:release` to keep the pool drained.

### Streaming output looks garbled mid-codepoint

A token can split a multi-byte UTF-8 codepoint. ion7-core's
`util.utf8` provides `is_complete(buf)` — the typical fix is to
buffer pieces until `utf8.is_complete(buffer)` returns true, then
flush. ion7-llm does NOT do this for you because the right granularity
depends on your transport (a curses TUI doesn't care, an SSE response
does).

### Model emits `[TOOL_CALLS]` but the loop ends after one turn

The bridge's `common_chat_parse` does template-aware tool-call
extraction. If your model uses a non-standard format the parser
doesn't recognise, the tool block will appear as raw `content` and
the dispatcher will not fire. Workarounds : pick a model whose chat
template advertises the JSON tool-call schema, or pre-process the
content yourself and call `Session:add_tool_result` manually.
