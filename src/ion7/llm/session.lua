--- @module ion7.llm.session
--- SPDX-License-Identifier: AGPL-3.0-or-later
--- Conversation state: message history, KV position, snapshot.
---
--- @author Ion7-Labs
--- @version 0.1.0

local Session = {}
Session.__index = Session

local _next_id = 0
local function next_id()
    _next_id = _next_id + 1
    return _next_id
end

--- @param  opts  table?
---   opts.system  string?  System prompt.
---   opts.seq_id  number?  KV sequence ID (assigned by ContextManager).
--- @return Session
function Session.new(opts)
    opts = opts or {}
    return setmetatable({
        id           = next_id(),
        seq_id       = opts.seq_id,
        system       = opts.system or nil,
        messages     = {},
        n_past       = 0,
        _snapshot    = nil,
        _dirty       = false,
        _msg_kv_ends = {},  -- absolute KV end pos per encoded message (ContextManager)
    }, Session)
end

--- Append a message. Sets _dirty = true.
--- @param  role     string  "user" | "assistant" | "system" | "tool"
--- @param  content  string
--- @return Session  self
function Session:add(role, content)
    assert(type(role)    == "string", "[ion7.llm.session] role must be a string")
    assert(type(content) == "string", "[ion7.llm.session] content must be a string")
    self.messages[#self.messages + 1] = { role = role, content = content }
    self._dirty = true
    return self
end

--- All messages including system (if set).
--- @return table
function Session:all_messages()
    if self.system then
        local out = { { role = "system", content = self.system } }
        for _, m in ipairs(self.messages) do out[#out + 1] = m end
        return out
    end
    return self.messages
end

--- The messages array (same reference as self.messages).
--- @return table
function Session:pending_messages()
    return self.messages
end

--- Store KV blob + position. Clears _dirty. Called by ContextManager.
function Session:_save_snapshot(blob, n_past)
    self._snapshot = blob
    self.n_past    = n_past
    self._dirty    = false
end

--- @return bool
function Session:has_snapshot()
    return self._snapshot ~= nil
end

--- @return string?
function Session:snapshot()
    return self._snapshot
end

--- Clear messages and reset KV state.
--- @return Session  self
function Session:reset()
    self.messages     = {}
    self.n_past       = 0
    self._snapshot    = nil
    self._dirty       = false
    self._msg_kv_ends = {}
    return self
end

--- Copy history into a new independent session. seq_id = nil.
--- ContextManager handles KV copy via kv_seq_cp.
--- @return Session
function Session:fork()
    local child = Session.new({ system = self.system })
    for _, m in ipairs(self.messages) do
        child.messages[#child.messages + 1] = { role = m.role, content = m.content }
    end
    child.n_past    = self.n_past
    child._snapshot = self._snapshot
    for i, v in ipairs(self._msg_kv_ends) do child._msg_kv_ends[i] = v end
    return child
end

--- Format messages as plain text. One "role: content" per line.
--- Useful for building summarization prompts in on_evict hooks.
--- @param  msgs  table?  Defaults to self.messages.
--- @return string
function Session:format(msgs)
    msgs = msgs or self.messages
    local parts = {}
    for _, m in ipairs(msgs) do
        parts[#parts + 1] = m.role .. ": " .. m.content
    end
    return table.concat(parts, "\n")
end

--- Serialize to a plain table (snapshot excluded).
--- @return table
function Session:serialize()
    return { id = self.id, system = self.system, messages = self.messages, n_past = self.n_past }
end

--- Restore from serialized table. KV must be re-prefilled (_dirty = true).
--- @param  t  table
--- @return Session
function Session.deserialize(t)
    local s = Session.new({ system = t.system })
    s.id       = t.id or next_id()
    s.messages = t.messages or {}
    s._dirty   = true
    return s
end

--- Save to JSON file. Requires dkjson or cjson.
--- @param  path  string
--- @return bool, string?
function Session:save(path)
    local ok, json = pcall(require, "dkjson")
    if not ok then ok, json = pcall(require, "cjson") end
    if not ok then return false, "no JSON lib (install dkjson)" end
    local f = io.open(path, "w")
    if not f then return false, "cannot open " .. path end
    f:write(json.encode(self:serialize()))
    f:close()
    return true
end

--- Load from JSON file. Requires dkjson or cjson.
--- @param  path  string
--- @return Session?, string?
function Session.load(path)
    local ok, json = pcall(require, "dkjson")
    if not ok then ok, json = pcall(require, "cjson") end
    if not ok then return nil, "no JSON lib" end
    local f = io.open(path, "r")
    if not f then return nil, "cannot open " .. path end
    local data = f:read("*a")
    f:close()
    local t, _, err = json.decode(data)
    if not t then return nil, err end
    return Session.deserialize(t)
end

return Session
