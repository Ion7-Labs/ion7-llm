--- @module ion7.llm.tools.spec
--- @author  ion7 / Ion7 Project Contributors
---
--- `Tool` — the description of one external action the model can
--- request. A `Tool` carries :
---
---   - `name`        unique identifier the model emits in tool_calls
---   - `description` natural-language hint for the model
---   - `schema`      JSON Schema describing the arguments
---   - `handler`     optional Lua function that executes the call
---                    locally — used by `tools.loop` to dispatch
---                    without bouncing through user code
---
--- A `ToolSet` bundles several tools together — the engine receives
--- a `ToolSet`, renders its description into the chat prompt, and
--- the response parser maps tool_calls back to entries by name.
---
--- The class is purely declarative ; nothing here talks to the model.

local json = require "ion7.vendor.json"

local Tool    = {}
local ToolSet = {}
Tool.__index    = Tool
ToolSet.__index = ToolSet

-- ── Tool ──────────────────────────────────────────────────────────────────

--- @class ion7.llm.tools.Tool
--- @field name        string
--- @field description string
--- @field schema      table       JSON Schema for the arguments object.
--- @field handler     function?   Optional `(args) -> result` dispatcher.

--- Build a tool definition.
---
--- @param  fields table
---   `name`        (string)            Required, must be a valid identifier.
---   `description` (string, default "")
---   `schema`      (table)             JSON Schema for the args object.
---   `handler`     (function?)         `(args_table) -> result_value`.
--- @return ion7.llm.tools.Tool
function Tool.new(fields)
    assert(type(fields) == "table", "[ion7.llm.tools.spec] Tool.new requires a table")
    assert(type(fields.name) == "string" and #fields.name > 0,
        "[ion7.llm.tools.spec] Tool.new requires a non-empty name")
    return setmetatable({
        name        = fields.name,
        description = fields.description or "",
        schema      = fields.schema or { type = "object", properties = {} },
        handler     = fields.handler,
    }, Tool)
end

--- Render a single tool to the `{ type = "function", function = … }`
--- shape modern chat templates expect when tool definitions are
--- injected into the system message.
--- @return table
function Tool:to_template_entry()
    return {
        type = "function",
        ["function"] = {
            name        = self.name,
            description = self.description,
            parameters  = self.schema,
        },
    }
end

--- Execute the tool locally. Raises if the tool has no handler.
--- Wraps the user's handler in `pcall` so a Lua error becomes a
--- structured error result instead of poisoning the engine loop.
---
--- @param  args table
--- @return any|table  result OR `{ error = "..." }` on handler failure.
function Tool:dispatch(args)
    assert(type(self.handler) == "function",
        "[ion7.llm.tools.spec] tool '" .. self.name .. "' has no handler")
    local ok, result = pcall(self.handler, args or {})
    if not ok then return { error = tostring(result) } end
    return result
end

-- ── ToolSet ───────────────────────────────────────────────────────────────

--- @class ion7.llm.tools.ToolSet
--- @field _by_name table<string, ion7.llm.tools.Tool>
--- @field _list    ion7.llm.tools.Tool[]

--- Build a tool set.
---
--- @param  tools ion7.llm.tools.Tool[]?  Optional initial population.
--- @return ion7.llm.tools.ToolSet
function ToolSet.new(tools)
    local s = setmetatable({ _by_name = {}, _list = {} }, ToolSet)
    if tools then for _, t in ipairs(tools) do s:add(t) end end
    return s
end

--- Add a tool to the set. Names must be unique.
--- @param  tool ion7.llm.tools.Tool
--- @return ion7.llm.tools.ToolSet self
function ToolSet:add(tool)
    assert(getmetatable(tool) == Tool,
        "[ion7.llm.tools.spec] ToolSet:add expects a Tool instance")
    assert(self._by_name[tool.name] == nil,
        "[ion7.llm.tools.spec] duplicate tool name : " .. tool.name)
    self._by_name[tool.name] = tool
    self._list[#self._list + 1] = tool
    return self
end

--- Look up a tool by name. Returns nil when unknown.
--- @param  name string
--- @return ion7.llm.tools.Tool?
function ToolSet:find(name) return self._by_name[name] end

--- Number of registered tools.
--- @return integer
function ToolSet:count() return #self._list end

--- Iterate over the tools in registration order.
--- @return function
function ToolSet:iter()
    local i = 0
    return function()
        i = i + 1
        return self._list[i]
    end
end

--- Render every tool to a JSON string usable in a system prompt or
--- chat-template `tools` parameter. Format follows the OpenAI /
--- modern chat-template convention :
---
---   [{ "type": "function", "function": { name, description, parameters } }, ...]
---
--- @return string
function ToolSet:to_json()
    local entries = {}
    for _, t in ipairs(self._list) do
        entries[#entries + 1] = t:to_template_entry()
    end
    return json.encode(entries)
end

return {
    Tool    = Tool,
    ToolSet = ToolSet,
}
