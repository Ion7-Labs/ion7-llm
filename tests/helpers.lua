--- @module tests.helpers
--- @author  ion7 / Ion7 Project Contributors
---
--- Shared scaffolding for the ion7-llm test suite. Mirrors the
--- ion7-core helper module — same env vars, same skip-don't-fail
--- contract, same `boot(T)` shortcut.
---
--- Path bootstrap : tests are launched from the repo root, but the
--- in-tree sources live under `src/ion7/...`. We prepend the local
--- `src/?.lua` roots, AND we look for a sibling ion7-core checkout so
--- a developer working from `_updates/ion7-llm/tests/run_all.sh` can
--- pick up the matching `_updates/ion7-core/src/...` without having
--- to `luarocks make ion7-core` after every change.
---
--- Environment :
---   ION7_MODEL    Required for model-dependent suites. Path to a chat
---                 GGUF (e.g. Qwen3-3B, Ministral-3B-Instruct).
---   ION7_EMBED    Required for the embedding suite (`17_embed.lua`).
---   ION7_DRAFT    Optional — speculative-decoding draft model.
---   ION7_GPU_LAYERS  Override `n_gpu_layers` (default 0).
---   ION7_CORE_SRC Optional. Override the ion7-core source root used
---                 for path bootstrap. Defaults to walking up two
---                 levels and looking for `../ion7-core/src/`.

-- ── package.path bootstrap ────────────────────────────────────────────────

local function _add_path(prefix)
    local extras = prefix .. "/?.lua;" .. prefix .. "/?/init.lua"
    if not package.path:find(extras, 1, true) then
        package.path = extras .. ";" .. package.path
    end
end

local function _probe_sibling(env_var, marker_path, candidates)
    local override = os.getenv(env_var)
    if override and override ~= "" then
        _add_path(override)
        return true
    end
    for _, candidate in ipairs(candidates) do
        local f = io.open(candidate .. "/" .. marker_path, "r")
        if f then
            f:close()
            _add_path(candidate)
            return true
        end
    end
    return false
end

local function _bootstrap_paths()
    -- Local ion7-llm sources first.
    _add_path("./src")

    -- ion7-core sources : honour ION7_CORE_SRC, otherwise probe the
    -- two sibling layouts the project actually uses
    -- (`_updates/ion7-core/src/` next to `_updates/ion7-llm/`, and
    -- the production `../ion7-core/src/`).
    _probe_sibling("ION7_CORE_SRC", "ion7/core/init.lua", {
        "../ion7-core/src",
        "../../ion7-core/src",
    })

    -- ion7-grammar sources : same probe pattern, optional. Files that
    -- do not require ion7.grammar simply ignore the missing path.
    _probe_sibling("ION7_GRAMMAR_SRC", "ion7/grammar/init.lua", {
        "../ion7-grammar/src",
        "../../ion7-grammar/src",
    })
end

_bootstrap_paths()

local M = {}

-- ── Environment helpers ───────────────────────────────────────────────────

local function _env(name)
    local v = os.getenv(name)
    if v == nil or v == "" then return nil end
    return v
end

M._env = _env

function M.model_path() return _env("ION7_MODEL") end

function M.require_model(T)
    local p = M.model_path()
    if not p then
        T.skip("(this whole file)",
            "ION7_MODEL not set — export ION7_MODEL=/path/to/model.gguf")
        T.summary()
        os.exit(0)
    end
    return p
end

function M.require_embed_model(T)
    local p = _env("ION7_EMBED")
    if not p then
        T.skip("(this whole file)",
            "ION7_EMBED not set — embedding tests skipped")
        T.summary()
        os.exit(0)
    end
    return p
end

function M.draft_model_path() return _env("ION7_DRAFT") end

function M.gpu_layers()
    return tonumber(_env("ION7_GPU_LAYERS") or "0") or 0
end

-- ── Filesystem helpers ────────────────────────────────────────────────────

function M.tmpfile(basename)
    local dir = _env("TMPDIR") or _env("TEMP") or "/tmp"
    return dir .. "/" .. basename
end

function M.try_remove(path) pcall(os.remove, path) end

-- ── Backend bring-up ──────────────────────────────────────────────────────

--- Require the native ion7-core stack (libllama, optional bridge),
--- skipping the whole file gracefully when the libraries cannot be
--- loaded. Returns the `ion7.core` module with `init({ log_level = 0 })`
--- already executed.
function M.require_backend(T)
    local ok, ion7 = pcall(require, "ion7.core")
    if not ok then
        T.skip("(this whole file)",
            "ion7.core failed to load — build vendor/llama.cpp + bridge first " ..
            "(or set ION7_CORE_SRC). Underlying error: " ..
            tostring(ion7):sub(1, 200))
        T.summary()
        os.exit(0)
    end

    local init_ok, err = pcall(ion7.init, { log_level = 0 })
    if not init_ok then
        T.skip("(this whole file)",
            "ion7.init() failed: " .. tostring(err):sub(1, 200))
        T.summary()
        os.exit(0)
    end

    return ion7
end

--- Boot the suite : require backend, load model, return `(ion7, model)`.
--- Default `n_gpu_layers` comes from `ION7_GPU_LAYERS`.
function M.boot(T, opts)
    local ion7 = M.require_backend(T)
    opts = opts or {}
    if opts.n_gpu_layers == nil then opts.n_gpu_layers = M.gpu_layers() end
    local path  = M.require_model(T)
    local model = ion7.Model.load(path, opts)
    return ion7, model
end

-- ── ion7-llm ergonomics ───────────────────────────────────────────────────

--- Build a single-session pipeline (context + cm + engine) on the
--- given model. Most chat-suite test files start with :
---
---   local ion7, model = H.boot(T)
---   local llm, ctx, vocab, cm, engine = H.pipeline(model, { n_ctx = 4096 })
---
--- so the test body can focus on the behaviour under check rather
--- than re-do the boilerplate plumbing.
---
--- @param  model ion7.core.Model
--- @param  opts  table? `n_ctx`, `n_seq_max`, `n_threads`, `headroom`...
--- @return table   ion7.llm module
--- @return ion7.core.Context
--- @return ion7.core.Vocab
--- @return ion7.llm.kv.ContextManager
--- @return ion7.llm.Engine
function M.pipeline(model, opts)
    opts = opts or {}
    local llm   = require "ion7.llm"
    local n_seq_max = opts.n_seq_max or 1
    local ctx   = model:context({
        n_ctx      = opts.n_ctx     or 4096,
        n_seq_max  = n_seq_max,
        n_threads  = opts.n_threads or 4,
        -- llama.cpp's seq_cp / seq_snapshot path requires the unified
        -- KV layout once a context juggles more than one sequence.
        kv_unified = (n_seq_max > 1) or opts.kv_unified or false,
    })
    local vocab = model:vocab()
    local cm, engine = llm.pipeline(ctx, vocab, {
        headroom        = opts.headroom or 256,
        default_sampler = opts.default_sampler,
        max_tokens      = opts.max_tokens,
    })
    return llm, ctx, vocab, cm, engine
end

return M
