--- @module ion7.llm
--- SPDX-License-Identifier: MIT
--- High-level LLM pipeline for LuaJIT. Built on ion7-core / llama.cpp.
---
--- Usage:
---   local llm = require "ion7.llm"
---   local engine = llm.new({ model = "/path/model.gguf", system = "..." })
---
---   -- One-shot
---   local resp = engine:complete("What is LuaJIT?")
---   print(resp.text)
---
---   -- Multi-turn
---   local session = engine:session()
---   session:add("user", "My name is Louis.")
---   local r1 = engine:chat(session)
---   session:add("assistant", r1.text)
---   session:add("user", "What's my name?")
---   local r2 = engine:chat(session)
---
--- @author Ion7-Labs
--- @version 0.3.0

local llm = { _VERSION = "0.2.0" }

-- Sub-module exports (accessible for advanced / low-level use)
llm.Session        = require "ion7.llm.session"
llm.Generator      = require "ion7.llm.generator"
llm.ContextManager = require "ion7.llm.context_manager"
llm.Scheduler      = require "ion7.llm.scheduler"
llm.Stop           = require "ion7.llm.stop"
llm.Response       = require "ion7.llm.response"
llm.sampler        = require "ion7.llm.sampler_profiles"

-- ── LLM ───────────────────────────────────────────────────────────────────────

--- @class LLM
--- Full-lifecycle LLM engine. Multiple independent instances are supported.
--- @field model   Model
--- @field vocab   Vocab
--- @field ctx     Context
--- @field cm      ContextManager
--- @field gen     Generator
local LLM = {}
LLM.__index = LLM

--- @param  opts  table
---   opts.model           string   Path to .gguf file. Required.
---   opts.n_gpu_layers    number?  GPU offload layers. Default: auto-fit.
---   opts.n_ctx           number?  Context window. Default: auto-fit or 4096.
---   opts.n_seq_max       number?  Max parallel sessions. Default: 1.
---   opts.system          string?  Default system prompt (prefix cache).
---   opts.sampler         string?  Profile: "balanced" (default), "precise",
---                                 "creative", "code", "fast", "thinking", "extended".
---   opts.max_tokens      number?  Default max tokens. Default: 2048.
---   opts.think           bool?    Strip <think> blocks. Default: false.
---   opts.think_budget    number?  Hard limit on tokens inside <think>. Default: nil.
---   opts.kv_type         string?  KV quantization: "f16" (default), "q8_0", "q4_0".
---   opts.kv_type_k       string?  K cache type independently.
---   opts.kv_type_v       string?  V cache type independently.
---   opts.n_sink          number?  Attention sink tokens. Default: 4.
---   opts.eviction        string?  "message" (default) or "fifo".
---   opts.headroom        number?  KV headroom reserved for generation.
---   opts.log_level       number?  0 = silent (default).
---   opts.speculative      string?  "ngram_cache" | "ngram_simple" | "ngram_map_k" |
---                                  "eagle3" | "draft" | false.
---   opts.n_draft          number?  Max draft tokens per speculative step. Default: 5.
---   opts.draft_model      string?  Path to draft .gguf (required when speculative="draft").
---   opts.draft_n_gpu_layers number? GPU layers for draft model. Default: 0.
---   opts.entropy_threshold number?  Entropy gate (nats): skip drafting when model uncertain.
---                                   nil = always speculate. Typical range: 0.5–2.0.
---   -- Extended sampler params (sampler="extended" or any of these set explicitly):
---   opts.temp            number?  Temperature (default: 0.8).
---   opts.top_k           number?  Top-K (default: 40).
---   opts.top_p           number?  Top-P (default: 0.95).
---   opts.min_p           number?  Min-P (default: 0.05).
---   opts.repeat_penalty  number?  Repetition penalty (default: 1.05 for extended).
---   opts.repeat_last_n   number?  Penalty window (default: 64).
---   opts.dry_mult        number?  DRY multiplier (default: 0.8 for extended).
---   opts.dry_base        number?  DRY base (default: 1.75).
---   opts.dry_allowed_len number?  DRY min sequence length (default: 2).
---   opts.xtc_probability number?  XTC fire probability (default: 0.1 for extended).
---   opts.xtc_threshold   number?  XTC threshold (default: 0.1).
---   opts.mirostat        number?  0=off, 1=v1, 2=v2 (default: 0).
---   opts.mirostat_tau    number?  Target entropy (default: 5.0).
---   opts.mirostat_eta    number?  Learning rate (default: 0.1).
---   opts.logit_bias      table?   { [token_id] = delta_logit, ... }
--- @return LLM
function LLM.new(opts)
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
    local vocab  = model:vocab()
    local ctx    = model:context({
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
        no_think = not (opts.think ~= false),
    })
    if opts.system then cm:set_system(opts.system) end

    -- ── Sampler ───────────────────────────────────────────────────────────────
    local sampler
    local native_budget = false

    local use_extended = (opts.sampler == "extended")
        or opts.dry_mult ~= nil
        or opts.xtc_probability ~= nil
        or opts.mirostat ~= nil
        or opts.logit_bias ~= nil

    if type(opts.sampler) == "table" then
        sampler = opts.sampler

    elseif use_extended then
        sampler = ion7.Sampler.common(model, {
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
        })

    else
        sampler = llm.sampler[opts.sampler or "balanced"](vocab)
    end

    if opts.think_budget and opts.think_budget > 0 and not use_extended
    and type(opts.sampler) ~= "table" then
        sampler = ion7.Sampler.chain()
            :reasoning_budget(model, opts.think_budget)
            :top_k(opts.top_k or 40)
            :top_p(opts.top_p or 0.95, 1)
            :min_p(opts.min_p or 0.05, 1)
            :temperature(opts.temp or opts.temperature or 0.8)
            :dist()
            :build(vocab)
        native_budget = true
    end

    -- ── Speculative ───────────────────────────────────────────────────────────
    local spec_engine  = nil
    local draft_model_ = nil
    local ctx_dft_     = nil

    if opts.speculative and opts.speculative ~= false then
        local spec_type = type(opts.speculative) == "string"
                          and opts.speculative or "ngram_cache"

        -- "draft" type requires a separate smaller model loaded as draft context.
        -- "eagle3" uses built-in draft heads on the target model — no draft ctx.
        if spec_type == "draft" then
            if opts.draft_model then
                local ok_dft, dft_m = pcall(ion7.Model.load, opts.draft_model, {
                    n_gpu_layers = opts.draft_n_gpu_layers or 0,
                })
                if ok_dft then
                    draft_model_ = dft_m
                    local ok_ctx, dft_c = pcall(dft_m.context, dft_m, {
                        n_ctx     = n_ctx,
                        n_seq_max = 1,
                    })
                    if ok_ctx then
                        ctx_dft_ = dft_c
                    else
                        io.stderr:write("[ion7.llm] draft context failed: " .. tostring(dft_c) .. "\n")
                        draft_model_ = nil
                    end
                else
                    io.stderr:write("[ion7.llm] draft model load failed: " .. tostring(dft_m) .. "\n")
                end
            else
                io.stderr:write("[ion7.llm] speculative='draft' requires opts.draft_model\n")
            end
        end

        local ok, s = pcall(ion7.Speculative.new, ion7.Speculative, ctx, ctx_dft_, {
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
        sampler            = sampler,
        max_tokens         = opts.max_tokens or 2048,
        think              = opts.think or false,
        think_budget       = opts.think_budget,
        native_budget      = native_budget,
        speculative        = spec_engine,
        entropy_threshold  = opts.entropy_threshold,
    })

    return setmetatable({
        model         = model,
        vocab         = vocab,
        ctx           = ctx,
        cm            = cm,
        gen           = gen,
        _ion7         = ion7,
        _opts         = opts,
        _spec         = spec_engine,
        _draft_model  = draft_model_,
        _ctx_dft      = ctx_dft_,
    }, LLM)
end

-- ── Lifecycle ─────────────────────────────────────────────────────────────────

--- Free all resources (VRAM, KV cache, model weights).
function LLM:shutdown()
    if self.gen      then self.gen:close()       end
    if self._spec    then self._spec:free()      end
    if self._ctx_dft then self._ctx_dft:free()   end
    if self.ctx      then self.ctx:free()        end
    if self._draft_model then self._draft_model:free() end
    if self.model    then self.model:free()      end
    if self._ion7    then self._ion7.shutdown()  end
    self.model = nil; self.vocab = nil
    self.ctx   = nil; self.cm   = nil; self.gen = nil
    self._draft_model = nil; self._ctx_dft = nil; self._spec = nil
end

-- ── Session management ────────────────────────────────────────────────────────

--- Create a Session pre-loaded with this engine's system prompt.
--- @param  init_msgs  string|table?  Optional: user string or { {role,content}, ... }.
--- @return Session
function LLM:session(init_msgs)
    local s = llm.Session.new({ system = self._opts.system })
    if type(init_msgs) == "string" then
        s:add("user", init_msgs)
    elseif type(init_msgs) == "table" then
        for _, m in ipairs(init_msgs) do s:add(m.role, m.content) end
    end
    return s
end

--- Fork a session: child shares KV history up to this point via kv_seq_cp.
--- @param  session  Session
--- @return Session
function LLM:fork(session)
    return self.cm:fork(session)
end

--- Release a session's KV slot. Call when the session is permanently done.
--- @param  session  Session
function LLM:release(session)
    self.cm:release(session)
end

-- ── KV checkpoint / rollback ──────────────────────────────────────────────────

--- Save a KV snapshot. Allows undoing generation back to this point.
--- @return LLM  self
function LLM:checkpoint()
    self.gen:checkpoint()
    return self
end

--- Restore to the last checkpoint().
--- @return LLM  self
function LLM:rollback()
    self.gen:rollback()
    return self
end

-- ── Speculative decoding ──────────────────────────────────────────────────────

--- Configure (or replace) speculative decoding at runtime.
--- Call after engine:new() to attach speculation without reloading the model.
---
--- @param  opts  table
---   opts.type              string   "ngram_cache" | "ngram_simple" | "ngram_map_k" |
---                                   "eagle3" | "draft". Default: "ngram_cache".
---   opts.n_draft           number?  Max draft tokens per step. Default: 5.
---   opts.draft_model       string?  Path to draft .gguf (required for type="draft").
---   opts.draft_ngl         number?  GPU layers for draft model. Default: 0.
---   opts.entropy_threshold number?  Entropy gate in nats. nil = always draft.
--- @return LLM  self
function LLM:setup_speculative(opts)
    opts = opts or {}
    local ion7 = self._ion7

    -- Free existing spec resources
    if self._spec    then self._spec:free();     self._spec    = nil end
    if self._ctx_dft then self._ctx_dft:free();  self._ctx_dft = nil end
    if self._draft_model then
        self._draft_model:free(); self._draft_model = nil
    end

    local spec_type = opts.type or "ngram_cache"
    local ctx_dft   = nil

    if spec_type == "draft" then
        assert(opts.draft_model,
            "[ion7.llm] setup_speculative: opts.draft_model required for type='draft'")
        local dft_m = ion7.Model.load(opts.draft_model, {
            n_gpu_layers = opts.draft_ngl or 0,
        })
        self._draft_model = dft_m
        ctx_dft           = dft_m:context({ n_ctx = self.ctx:n_ctx_seq(), n_seq_max = 1 })
        self._ctx_dft     = ctx_dft
    end

    local ok, s = pcall(ion7.Speculative.new, ion7.Speculative, self.ctx, ctx_dft, {
        type    = spec_type,
        n_draft = opts.n_draft or 5,
    })

    if not ok then
        io.stderr:write("[ion7.llm] setup_speculative: " .. tostring(s) .. "\n")
        return self
    end

    self._spec = s
    self.gen:set_speculative(s)

    if opts.entropy_threshold ~= nil then
        self:set_spec_entropy_threshold(opts.entropy_threshold)
    end

    return self
end

--- Enable entropy-adaptive speculative gating.
--- After each token sample, ctx:entropy(0) is measured. If entropy > threshold,
--- the speculative draft step is skipped — normal single-token decoding is used.
---
--- Low entropy (model confident) → drafts accepted → throughput increase.
--- High entropy (model uncertain) → drafts rejected → overhead avoided.
---
--- Typical values:
---   0.5 nats — aggressive, drafts only when very confident
---   1.0 nats — balanced (good default)
---   2.0 nats — conservative, drafts unless nearly uniform distribution
---   nil       — disabled, always speculate (default behaviour)
---
--- @param  threshold  number?
--- @return LLM  self
function LLM:set_spec_entropy_threshold(threshold)
    self.gen:set_entropy_threshold(threshold)
    return self
end

--- Disable speculative decoding entirely (detaches the engine).
--- @return LLM  self
function LLM:disable_speculative()
    self.gen:set_speculative(nil)
    return self
end

--- Print speculative decoding stats to stderr (acceptance rate, token counts).
--- No-op if speculation was not configured.
function LLM:spec_stats()
    if self._spec then self._spec:stats() end
end

-- ── Introspection ─────────────────────────────────────────────────────────────

--- KV usage statistics.
--- @return table  { n_past, n_ctx, fill_pct, prefix_tokens, prefix_cached,
---                  n_sink, eviction, slots_total, slots_free,
---                  n_evictions, n_tokens_evicted }
function LLM:ctx_usage()
    local stats  = self.cm:stats()
    local n_past = self.ctx:n_past()
    local n_ctx  = self.ctx:n_ctx_seq()
    stats.n_past   = n_past
    stats.n_ctx    = n_ctx
    stats.fill_pct = (n_ctx > 0) and (n_past / n_ctx * 100) or 0
    return stats
end

-- ── Generation ────────────────────────────────────────────────────────────────

--- One-shot blocking generation. Creates a temporary session internally.
---
--- @param  text  string|table  User message or message array { {role,content}, ... }.
--- @param  opts  table?
---   opts.max_tokens  number?
---   opts.grammar     table|string?
---   opts.sampler     table?
---   opts.on_token    function?   Called with each visible piece.
---   opts.on_think    function?   Called with each piece inside <think>.
--- @return Response
function LLM:complete(text, opts)
    return self.gen:chat(self:session(text), opts or {})
end

--- Blocking chat on a persistent session.
---
--- @param  session  Session
--- @param  opts     table?  Same opts as complete().
--- @return Response
function LLM:chat(session, opts)
    assert(session and session.messages,
        "[ion7.llm] chat() requires a Session — use engine:session() to create one")
    return self.gen:chat(session, opts or {})
end

--- Streaming on a persistent session. Yields visible pieces.
--- Think-block pieces are routed to opts.on_think, not yielded.
---
--- @param  session  Session
--- @param  opts     table?  Same opts as complete().
--- @return function  iterator
function LLM:stream(session, opts)
    assert(session and session.messages,
        "[ion7.llm] stream() requires a Session — use engine:session() to create one")
    return self.gen:stream(session, opts or {})
end

--- Grammar-constrained blocking generation.
--- @param  session  Session
--- @param  grammar  table|string
--- @param  opts     table?
--- @return Response
function LLM:structured(session, grammar, opts)
    opts = opts or {}; opts.grammar = grammar
    return self:chat(session, opts)
end

--- Grammar-constrained streaming.
--- @param  session  Session
--- @param  grammar  table|string
--- @param  opts     table?
--- @return function
function LLM:stream_structured(session, grammar, opts)
    opts = opts or {}; opts.grammar = grammar
    return self:stream(session, opts)
end

--- Agentic ReAct loop: generate → parse tool call → handler → continue.
--- Session is updated in place (assistant + tool turns appended automatically).
---
--- @param  session  Session
--- @param  opts     table
---   opts.parse_fn   function  fn(text) → { name, args } | nil
---   opts.handler    function  fn(name, args, session) → string
---   opts.max_turns  number?   Default: 8.
---   opts.grammar    table?    Grammar for each generation step.
---   opts.max_tokens number?
---   opts.on_token   function?
---   opts.on_think   function?
--- @return Response  last response
--- @return number    turns executed
function LLM:chat_with_tools(session, opts)
    opts = opts or {}
    assert(opts.parse_fn, "[ion7.llm] chat_with_tools requires opts.parse_fn")
    assert(opts.handler,  "[ion7.llm] chat_with_tools requires opts.handler")

    local gen_opts = {
        max_tokens = opts.max_tokens,
        grammar    = opts.grammar,
        on_token   = opts.on_token,
        on_think   = opts.on_think,
    }

    local resp
    local turns = 0

    for _ = 1, opts.max_turns or 8 do
        turns = turns + 1
        resp  = self.gen:chat(session, gen_opts)
        local call = opts.parse_fn(resp.text)
        if not call then break end
        session:add("assistant", resp.text)
        local ok, result = pcall(opts.handler, call.name, call.args, session)
        session:add("tool", ok and tostring(result) or "[tool error] " .. tostring(result))
    end

    return resp, turns
end

--- Parallel generation for N sessions via Scheduler.
--- Requires n_seq_max >= #jobs.
--- @param  jobs  table  { session, sampler?, max_tokens?, on_piece?, on_done? }[]
--- @return Scheduler
function LLM:batch(jobs)
    assert(type(jobs) == "table" and #jobs >= 2,
        "[ion7.llm] batch() requires at least 2 jobs")
    local n_seq = tonumber(self.ctx:n_seq_max())
    assert(n_seq >= #jobs, string.format(
        "[ion7.llm] n_seq_max=%d but %d jobs submitted", n_seq, #jobs))

    for i, job in ipairs(jobs) do job.session.seq_id = i - 1 end

    local sched = llm.Scheduler.new(self.ctx, self.vocab, self.cm)
    for _, job in ipairs(jobs) do
        sched:submit(job.session, {
            sampler    = job.sampler or self.gen._sampler,
            max_tokens = job.max_tokens or 2048,
            on_piece   = job.on_piece,
            on_done    = job.on_done,
            stop       = job.stop,
        })
    end
    return sched:run()
end

-- ── Module API ────────────────────────────────────────────────────────────────

--- Create a new LLM engine.
--- @param  opts  table  See LLM.new().
--- @return LLM
function llm.new(opts)
    return LLM.new(opts)
end

llm.LLM = LLM

-- Optional ion7-grammar integration
llm.Grammar = (function()
    local ok, g = pcall(require, "ion7.grammar")
    return ok and g or nil
end)()

return llm
