--- @module ion7.llm.sampler_profiles
--- SPDX-License-Identifier: MIT
--- Named sampler presets. All return a built ion7-core Sampler or CSampler.
---
--- Classic profiles (balanced, precise, creative, code, fast, thinking) use
--- Sampler.chain() and require only a vocab.
---
--- The `extended` profile uses CSampler (ion7_csampler_t) and requires both
--- model and vocab. It enables DRY, XTC, and mirostat support.
---
--- @author Ion7-Labs
--- @version 0.2.0

local function build(vocab, fn)
    local chain = require("ion7.core").Sampler.chain()
    return fn(chain):build(vocab)
end

local profiles = {}

--- @param  vocab  Vocab
--- @return Sampler
function profiles.balanced(vocab)
    return build(vocab, function(c)
        return c:top_k(40):top_p(0.95, 1):min_p(0.05, 1):temperature(0.8):dist()
    end)
end

--- @param  vocab  Vocab
--- @return Sampler
function profiles.precise(vocab)
    return build(vocab, function(c)
        return c:top_k(20):top_p(0.85, 1):min_p(0.05, 1):temperature(0.3):dist()
    end)
end

--- @param  vocab  Vocab
--- @return Sampler
function profiles.creative(vocab)
    return build(vocab, function(c)
        return c:top_k(80):top_p(0.98, 1):min_p(0.02, 1):temperature(1.1):dist()
    end)
end

--- Mild repetition penalty, low temperature.
--- @param  vocab  Vocab
--- @return Sampler
function profiles.code(vocab)
    return build(vocab, function(c)
        return c:penalties(64, 1.05, 0.0, 0.0):top_k(30):top_p(0.9, 1):temperature(0.2):dist()
    end)
end

--- Greedy - fully deterministic, maximum speed.
--- @param  vocab  Vocab
--- @return Sampler
function profiles.fast(vocab)
    return build(vocab, function(c) return c:greedy() end)
end

--- Tuned for reasoning models (Qwen3.5, DeepSeek-R1).
--- Use with opts.think = true.
--- @param  vocab  Vocab
--- @return Sampler
function profiles.thinking(vocab)
    return build(vocab, function(c)
        return c:top_k(30):top_p(0.90, 1):min_p(0.05, 1):temperature(0.6):dist()
    end)
end

--- Extended profile via CSampler (ion7_csampler_t).
--- Enables DRY repetition control, XTC diversity, and mirostat.
--- Requires the model object (not just vocab).
---
--- @param  model  Model   ion7-core Model object.
--- @param  opts   table?  Overrides (see Sampler.common() for all options).
---   Notable opts:
---     dry_mult        (default 0.8)   — DRY multiplier, 0 = disabled
---     dry_base        (default 1.75)
---     xtc_probability (default 0.1)   — 0 = disabled
---     xtc_threshold   (default 0.1)
---     mirostat        (default 0)     — 0 = off, 1 = v1, 2 = v2
---     mirostat_tau    (default 5.0)
---     temp            (default 0.8)
---     top_k           (default 40)
---     top_p           (default 0.95)
---     logit_bias      (default nil)   — { [tok_id] = delta }
--- @return CSampler
function profiles.extended(model, opts)
    local Sampler = require("ion7.core").Sampler
    opts = opts or {}
    return Sampler.common(model, {
        seed            = opts.seed,
        top_k           = opts.top_k           or 40,
        top_p           = opts.top_p           or 0.95,
        min_p           = opts.min_p           or 0.05,
        temp            = opts.temp or opts.temperature or 0.8,
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
        grammar         = opts.grammar,
        grammar_lazy    = opts.grammar_lazy,
        trigger_words   = opts.trigger_words,
        logit_bias      = opts.logit_bias,
    })
end

return profiles
