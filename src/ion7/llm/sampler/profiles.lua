--- @module ion7.llm.sampler.profiles
--- @author  ion7 / Ion7 Project Contributors
---
--- Named sampler presets. Each function returns a built
--- `ion7.core.Sampler` ready to plug into the engine. Callers that
--- want a non-standard chain build it themselves via
--- `ion7.core.Sampler.chain():...:build()`.

local Sampler = require("ion7.core").Sampler

local M = {}

local function chain() return Sampler.chain() end

--- General-purpose default. Mild top-k / top-p / min-p with a
--- moderate temperature ; a reasonable choice when you do not have
--- a specific reason to deviate.
--- @return ion7.core.Sampler
function M.balanced()
    return chain()
        :top_k(40):top_p(0.95):min_p(0.05):temperature(0.8):dist():build()
end

--- Tighter cut-offs, lower temperature. Best when you want
--- plausible, on-topic continuations without much variability.
--- @return ion7.core.Sampler
function M.precise()
    return chain()
        :top_k(20):top_p(0.85):min_p(0.05):temperature(0.3):dist():build()
end

--- Wider candidate window, higher temperature. For brainstorming,
--- creative writing, idea generation.
--- @return ion7.core.Sampler
function M.creative()
    return chain()
        :top_k(80):top_p(0.98):min_p(0.02):temperature(1.1):dist():build()
end

--- Code-tuned. Mild repetition penalty, low temperature, narrow
--- top-k. Reduces the rate of accidental loops in code completions.
--- @return ion7.core.Sampler
function M.code()
    return chain()
        :penalties(64, 1.05, 0.0, 0.0)
        :top_k(30):top_p(0.9):temperature(0.2):dist():build()
end

--- Pure greedy. Deterministic and the fastest possible step ; best
--- for grammar-constrained workloads where any randomness would
--- conflict with the constraint.
--- @return ion7.core.Sampler
function M.fast()
    return chain():greedy():build()
end

--- Tuned for reasoning models — wider top-p, slightly cooler temp,
--- still some entropy. Pair with `engine.chat({ thinking = true })`.
--- @return ion7.core.Sampler
function M.thinking()
    return chain()
        :top_k(30):top_p(0.90):min_p(0.05):temperature(0.6):dist():build()
end

return M
