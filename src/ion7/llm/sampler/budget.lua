--- @module ion7.llm.sampler.budget
--- @author  ion7 / Ion7 Project Contributors
---
--- Wraps ion7-core's `reasoning_budget` sampler step in a small
--- helper. Models that emit `<think>...</think>` blocks honour the
--- budget by closing the block early when the cap is hit ; models
--- that ignore thinking just see a no-op sampler.
---
--- The step MUST appear FIRST in the chain — it operates on logit
--- distributions before any temperature / top-k filter sees them.
--- The helpers below build complete chains so the user does not have
--- to remember the ordering rule.

local Sampler = require("ion7.core").Sampler

local M = {}

--- Build a sampler whose first step is a reasoning-budget guard.
--- The remaining steps mimic the `balanced` profile.
---
--- @param  model    ion7.core.Model|cdata  Model handle (any form).
--- @param  budget   integer                Token cap inside `<think>`.
--- @param  opts     table?
---   `top_k`        (integer, default 40)
---   `top_p`        (number,  default 0.95)
---   `min_p`        (number,  default 0.05)
---   `temperature`  (number,  default 0.8)
---   `seed`         (integer)
--- @return ion7.core.Sampler
function M.balanced_with_budget(model, budget, opts)
    opts = opts or {}
    return Sampler.chain()
        :reasoning_budget(model, budget)
        :top_k(opts.top_k or 40)
        :top_p(opts.top_p or 0.95)
        :min_p(opts.min_p or 0.05)
        :temperature(opts.temperature or 0.8)
        :dist(opts.seed):build()
end

--- Compose a budget step on top of an arbitrary builder. Handy when
--- a caller already configured a custom chain (penalties, mirostat,
--- grammar, …) and wants to add the budget in front.
---
--- @param  builder ion7.core.SamplerBuilder  Started but NOT built.
--- @param  model   ion7.core.Model|cdata
--- @param  budget  integer
--- @return ion7.core.SamplerBuilder
function M.prepend(builder, model, budget)
    -- The builder just appends to `_steps` ; we splice the budget
    -- step at the front so it runs first.
    builder._steps = builder._steps or {}
    table.insert(builder._steps, 1, {
        type     = "reasoning_budget",
        model_ptr = type(model) == "table" and model:ptr() or model,
        n_budget  = budget or 512,
    })
    return builder
end

return M
