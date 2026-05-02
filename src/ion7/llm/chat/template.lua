--- @module ion7.llm.chat.template
--- @author  ion7 / Ion7 Project Contributors
---
--- Render a message list to a prompt string through the model's
--- embedded chat template. Thin layer over `Vocab:apply_template`,
--- promoted to its own file so the rest of the code base does not
--- have to remember the `enable_thinking` magic-number convention :
---
---   - `-1` : let the template pick its own default
---   - ` 0` : disable thinking (Qwen3, DeepSeek-R1, … honour it)
---   - ` 1` : force thinking on
---
--- The `thinking` argument here takes a plain bool ; this module
--- maps it onto the int the template expects.

local Template = {}
Template.__index = Template

--- @class ion7.llm.chat.Template
--- @field _vocab ion7.core.Vocab

--- @param  vocab ion7.core.Vocab
--- @return ion7.llm.chat.Template
function Template.new(vocab)
    return setmetatable({ _vocab = vocab }, Template)
end

--- Render messages to a prompt string.
---
--- @param  messages table[]                  Canonical message array.
--- @param  opts     table?
---   `add_assistant_prefix` (boolean, default true)  Append the
---     turn-start tokens that prime the model to generate. Set to
---     false when rendering history for an external consumer.
---   `thinking` (boolean | nil, default nil)
---     `nil` → use the template's own default (`enable_thinking = -1`).
---     `true`  → `enable_thinking = 1`.
---     `false` → `enable_thinking = 0`.
--- @return string
function Template:render(messages, opts)
    opts = opts or {}
    local add_ass = opts.add_assistant_prefix
    if add_ass == nil then add_ass = true end

    local et
    if     opts.thinking == nil   then et = -1
    elseif opts.thinking == true  then et = 1
    else                                et = 0
    end

    return self._vocab:apply_template(messages, add_ass, et)
end

--- True when the embedded template recognises `enable_thinking` (and
--- therefore `opts.thinking` will have an effect).
--- @return boolean
function Template:supports_thinking()
    return self._vocab:supports_thinking()
end

--- Render + tokenise in one call, returning the cdata token array
--- and its length. Convenient when the caller only needs the tokens.
---
--- @param  messages table[]
--- @param  opts     table?  Same shape as `:render`.
--- @return cdata    `int32_t[?]` token array.
--- @return integer  Token count.
function Template:tokenize(messages, opts)
    local prompt = self:render(messages, opts)
    return self._vocab:tokenize(prompt, false, true)
end

return Template
