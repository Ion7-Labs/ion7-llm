--- @module ion7.llm
--- SPDX-License-Identifier: AGPL-3.0-or-later
--- High-level LLM pipeline for LuaJIT. Built on ion7-core / llama.cpp.
---
--- @author Ion7-Labs
--- @version 0.1.0

local llm = { _VERSION = "0.1.0" }

llm.Session        = require "ion7.llm.session"
llm.Generator      = require "ion7.llm.generator"
llm.ContextManager = require "ion7.llm.context_manager"
llm.Scheduler      = require "ion7.llm.scheduler"
llm.Stop           = require "ion7.llm.stop"
llm.Response       = require "ion7.llm.response"
llm.sampler        = require "ion7.llm.sampler_profiles"

local _state = {}

--- Initialise ion7-llm. Must be called before anything else.
---
--- @param  opts  table
---   opts.model         string   Path to .gguf file. Required.
---   opts.n_gpu_layers  number?  GPU offload layers. Default: auto-fit.
---   opts.n_ctx         number?  Context window. Default: auto-fit or 4096.
---   opts.n_seq_max     number?  Max parallel sessions. Default: 1.
---   opts.system        string?  Default system prompt (prefix cache).
---   opts.sampler       string?  Sampler profile name. Default: "balanced".
---   opts.max_tokens    number?  Default max tokens. Default: 2048.
---   opts.think         bool?    Strip <think> blocks. Default: false.
---   opts.think_budget  number?  Max tokens inside a think block. Default: nil.
---   opts.kv_type       string?  KV quantization: "f16" (default), "q8_0", "q4_0", etc.
---   opts.kv_type_k     string?  K cache type independently.
---   opts.kv_type_v     string?  V cache type independently.
---   opts.n_sink        number?  Attention sink size. Default: 4.
---   opts.eviction      string?  "message" (default) or "fifo".
---   opts.headroom      number?  Override KV headroom reserved for generation.
---   opts.log_level     number?  0 = silent (default).
--- @return llm
function llm.init(opts)
    assert(type(opts) == "table" and type(opts.model) == "string",
        "[ion7.llm] opts.model (path to .gguf) is required")

    local ion7 = require "ion7.core"
    ion7.init({ log_level = opts.log_level or 0 })

    local ngl   = opts.n_gpu_layers
    local n_ctx = opts.n_ctx or 4096
    if ngl == nil then
        local fit = ion7.Model.fit_params(opts.model, { n_ctx = n_ctx })
        if fit then ngl = fit.n_gpu_layers; n_ctx = fit.n_ctx
        else        ngl = 0 end
    end

    local model = ion7.Model.load(opts.model, { n_gpu_layers = ngl })
    local vocab = model:vocab()
    local ctx   = model:context({
        n_ctx     = n_ctx,
        n_seq_max = opts.n_seq_max or 1,
        kv_type   = opts.kv_type,
        kv_type_k = opts.kv_type_k,
        kv_type_v = opts.kv_type_v,
    })

    local cm = llm.ContextManager.new(ctx, vocab, {
        headroom = opts.headroom or math.min(
            opts.max_tokens or 256,
            math.floor(ctx:n_ctx_seq() * 0.25)
        ),
        n_sink   = opts.n_sink,
        eviction = opts.eviction,
    })
    if opts.system then cm:set_system(opts.system) end

    local profile = opts.sampler or "balanced"
    local sampler = type(profile) == "string" and llm.sampler[profile](vocab) or profile

    local gen = llm.Generator.new(ctx, vocab, cm, {
        sampler      = sampler,
        max_tokens   = opts.max_tokens or 2048,
        think        = opts.think or false,
        think_budget = opts.think_budget,
    })

    _state = { ion7 = ion7, model = model, vocab = vocab, ctx = ctx, cm = cm, gen = gen, opts = opts }
    return llm
end

--- Free all resources.
function llm.shutdown()
    if _state.ctx   then _state.ctx:free()      end
    if _state.model then _state.model:free()    end
    if _state.ion7  then _state.ion7.shutdown() end
    _state = {}
end

-- ── Accessors ─────────────────────────────────────────────────────────────────

function llm.model() assert(_state.model, "[ion7.llm] call init() first"); return _state.model end
function llm.vocab() assert(_state.vocab, "[ion7.llm] call init() first"); return _state.vocab end
function llm.ctx()   assert(_state.ctx,   "[ion7.llm] call init() first"); return _state.ctx   end
function llm.cm()    assert(_state.cm,    "[ion7.llm] call init() first"); return _state.cm    end
function llm.gen()   assert(_state.gen,   "[ion7.llm] call init() first"); return _state.gen   end

-- ── Convenience API ───────────────────────────────────────────────────────────

local function _session(text)
    local s = llm.Session.new({ system = _state.opts and _state.opts.system })
    if type(text) == "string" then
        s:add("user", text)
    else
        for _, m in ipairs(text) do s:add(m.role, m.content) end
    end
    return s
end

--- One-shot blocking chat.
--- @param  text  string|table
--- @param  opts  table?  { on_token, max_tokens, sampler, grammar, think_budget }
--- @return Response
function llm.chat(text, opts)
    return _state.gen:chat(_session(text), opts or {})
end

--- Streaming coroutine. Yields string pieces.
--- @param  text  string|table
--- @param  opts  table?
--- @return function
function llm.stream(text, opts)
    return _state.gen:stream(_session(text), opts)
end

--- Chat with grammar constraint. Output guaranteed to match grammar.
--- @param  text     string|table
--- @param  grammar  table|string  Grammar_obj or GBNF string.
--- @param  opts     table?
--- @return Response
function llm.structured(text, grammar, opts)
    opts = opts or {}
    opts.grammar = grammar
    return llm.chat(text, opts)
end

--- Streaming structured generation.
--- @param  text     string|table
--- @param  grammar  table|string
--- @param  opts     table?
--- @return function
function llm.stream_structured(text, grammar, opts)
    opts = opts or {}
    opts.grammar = grammar
    return llm.stream(text, opts)
end

--- Parallel generation for N sessions via Scheduler.
--- Requires n_seq_max >= #jobs (set in llm.init).
--- @param  jobs  table  Array of { session, sampler?, max_tokens?, on_piece?, on_done? }
--- @return Scheduler  (after run() - all jobs completed)
function llm.batch(jobs)
    assert(type(jobs) == "table" and #jobs >= 2,
        "[ion7.llm] llm.batch() requires at least 2 jobs")
    local n_seq = tonumber(_state.ctx:n_seq_max())
    assert(n_seq >= #jobs, string.format(
        "[ion7.llm] n_seq_max=%d but %d jobs submitted", n_seq, #jobs))

    for i, job in ipairs(jobs) do job.session.seq_id = i - 1 end

    local sched = llm.Scheduler.new(_state.ctx, _state.vocab, _state.cm)
    for _, job in ipairs(jobs) do
        sched:submit(job.session, {
            sampler    = job.sampler or _state.gen._sampler,
            max_tokens = job.max_tokens or 2048,
            on_piece   = job.on_piece,
            on_done    = job.on_done,
            stop       = job.stop,
        })
    end
    return sched:run()
end

-- Optional ion7-grammar integration
llm.Grammar = (function()
    local ok, g = pcall(require, "ion7.grammar")
    return ok and g or nil
end)()

return llm
