--- @module ion7.llm
--- SPDX-License-Identifier: MIT
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
---   opts.model           string   Path to .gguf file. Required.
---   opts.n_gpu_layers    number?  GPU offload layers. Default: auto-fit.
---   opts.n_ctx           number?  Context window. Default: auto-fit or 4096.
---   opts.n_seq_max       number?  Max parallel sessions. Default: 1.
---   opts.system          string?  Default system prompt (prefix cache).
---   opts.sampler         string?  Sampler profile: "balanced" (default), "precise",
---                                 "creative", "code", "fast", "thinking", "extended".
---                                 "extended" uses CSampler with DRY, XTC, mirostat.
---   opts.max_tokens      number?  Default max tokens. Default: 2048.
---   opts.think           bool?    Strip <think> blocks. Default: false.
---   opts.think_budget    number?  Hard limit on tokens inside <think> (via native
---                                 reasoning_budget sampler). Default: nil.
---   opts.kv_type         string?  KV quantization: "f16" (default), "q8_0", "q4_0".
---   opts.kv_type_k       string?  K cache type independently.
---   opts.kv_type_v       string?  V cache type independently.
---   opts.n_sink          number?  Attention sink size. Default: 4.
---   opts.eviction        string?  "message" (default) or "fifo".
---   opts.headroom        number?  Override KV headroom reserved for generation.
---   opts.log_level       number?  0 = silent (default).
---   opts.speculative     string?  Speculative decoding type: "ngram_cache" (default
---                                 when enabled), "ngram_simple", "ngram_map_k".
---                                 Set to false/nil to disable. Enabled when this
---                                 field is a non-false string.
---   opts.n_draft         number?  Max draft tokens per speculative step. Default: 5.
---   -- Extended sampler params (used when sampler = "extended" or set explicitly):
---   opts.temp            number?  Temperature (default: 0.8).
---   opts.top_k           number?  Top-K (default: 40).
---   opts.top_p           number?  Top-P (default: 0.95).
---   opts.min_p           number?  Min-P (default: 0.05).
---   opts.repeat_penalty  number?  Repetition penalty (default: 1.05 for extended).
---   opts.repeat_last_n   number?  Penalty window (default: 64).
---   opts.dry_mult        number?  DRY multiplier (default: 0.8 for extended, 0=off).
---   opts.dry_base        number?  DRY base (default: 1.75).
---   opts.dry_allowed_len number?  DRY min sequence length (default: 2).
---   opts.xtc_probability number?  XTC fire probability (default: 0.1 for extended, 0=off).
---   opts.xtc_threshold   number?  XTC logit threshold (default: 0.1).
---   opts.mirostat        number?  0=off, 1=v1, 2=v2 (default: 0).
---   opts.mirostat_tau    number?  Mirostat target entropy (default: 5.0).
---   opts.mirostat_eta    number?  Mirostat learning rate (default: 0.1).
---   opts.logit_bias      table?   { [token_id] = delta_logit, ... }
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
        no_think = not (opts.think ~= false),  -- true when think=false
    })
    if opts.system then cm:set_system(opts.system) end

    -- ── Sampler construction ──────────────────────────────────────────────────
    -- "extended" profile or explicit advanced params → CSampler (DRY, XTC, mirostat).
    -- All other named profiles → classic Sampler.chain().
    -- Pre-built sampler passed directly → used as-is.

    local sampler
    local native_budget = false

    -- Detect whether any advanced CSampler param is explicitly set
    local use_extended = (opts.sampler == "extended")
        or opts.dry_mult ~= nil
        or opts.xtc_probability ~= nil
        or opts.mirostat ~= nil
        or opts.logit_bias ~= nil

    if type(opts.sampler) == "table" then
        -- Pre-built Sampler or CSampler passed directly
        sampler = opts.sampler

    elseif use_extended then
        -- CSampler path — supports DRY, XTC, mirostat, logit_bias
        local sampler_opts = {
            temp            = opts.temp or opts.temperature or 0.8,
            top_k           = opts.top_k           or 40,
            top_p           = opts.top_p           or 0.95,
            min_p           = opts.min_p           or 0.05,
            repeat_penalty  = opts.repeat_penalty  or 1.05,
            freq_penalty    = opts.freq_penalty    or 0.0,
            pres_penalty    = opts.pres_penalty    or 0.0,
            repeat_last_n   = opts.repeat_last_n   or 64,
            dry_mult        = opts.dry_mult        or 0.8,
            dry_base        = opts.dry_base        or 1.75,
            dry_allowed_len = opts.dry_allowed_len or 2,
            dry_last_n      = opts.dry_last_n      or -1,
            xtc_probability = opts.xtc_probability or 0.1,
            xtc_threshold   = opts.xtc_threshold   or 0.1,
            mirostat        = opts.mirostat        or 0,
            mirostat_tau    = opts.mirostat_tau    or 5.0,
            mirostat_eta    = opts.mirostat_eta    or 0.1,
            logit_bias      = opts.logit_bias,
        }
        sampler = ion7.Sampler.common(model, sampler_opts)

    else
        -- Classic Sampler.chain() path via named profiles
        local profile = opts.sampler or "balanced"
        sampler = llm.sampler[profile](vocab)
    end

    -- Native reasoning budget: add ion7_reasoning_budget_init() to a chain
    -- wrapping the existing sampler. Only for non-CSampler paths and when budget > 0.
    -- CSampler doesn't support reasoning_budget insertion post-construction,
    -- so it falls back to the Lua soft-limit in generator.lua.
    if opts.think_budget and opts.think_budget > 0 and not use_extended
    and type(opts.sampler) ~= "table" then
        -- Build a new chain: reasoning_budget first, then the existing sampler steps
        -- Re-build with reasoning_budget prepended via a fresh chain
        local s = opts  -- alias for sampler opts
        local chain = ion7.Sampler.chain()
            :reasoning_budget(model, opts.think_budget)
            :top_k(s.top_k or 40)
            :top_p(s.top_p or 0.95, 1)
            :min_p(s.min_p or 0.05, 1)
            :temperature(s.temp or s.temperature or 0.8)
            :dist()
            :build(vocab)
        sampler = chain
        native_budget = true
    end

    -- ── Speculative decoding ─────────────────────────────────────────────────
    -- Enabled when opts.speculative is a string (type name) or true (→ ngram_cache).
    -- Disabled when opts.speculative is nil/false.
    -- Not compatible with grammar samplers (grammar constrains every logit position;
    -- speculative batch decoding would require grammar state at each draft position).
    local spec_engine = nil
    if opts.speculative and opts.speculative ~= false then
        local Speculative = ion7.Speculative
        local spec_type   = type(opts.speculative) == "string"
                            and opts.speculative
                            or  "ngram_cache"
        local ok, s = pcall(Speculative.new, Speculative, ctx, nil, {
            type    = spec_type,
            n_draft = opts.n_draft or 5,
        })
        if ok then
            spec_engine = s
        else
            io.stderr:write("[ion7.llm] speculative init failed: " .. tostring(s) .. "\n")
        end
    end

    local gen = llm.Generator.new(ctx, vocab, cm, {
        sampler        = sampler,
        max_tokens     = opts.max_tokens or 2048,
        think          = opts.think or false,
        think_budget   = opts.think_budget,
        native_budget  = native_budget,
        speculative    = spec_engine,
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
