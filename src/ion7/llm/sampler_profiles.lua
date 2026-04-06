--- @module ion7.llm.sampler_profiles
--- SPDX-License-Identifier: MIT
--- Named sampler presets. All return a built ion7-core Sampler.
---
--- @author Ion7-Labs
--- @version 0.1.0

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

return profiles
