--- Minimal test framework for ion7-core.
--- No external dependencies.

local M = {}

M.pass      = 0
M.fail      = 0
M.n_skipped = 0
M._suite = "?"

function M.suite(name)
    M._suite = name
    io.write(string.format("\n\27[1m── %s \27[0m%s\n", name, string.rep("─", 52 - #name)))
end

function M.test(name, fn)
    local ok, err = pcall(fn)
    if ok then
        M.pass = M.pass + 1
        io.write(string.format("  \27[32m[OK]\27[0m %s\n", name))
    else
        M.fail = M.fail + 1
        io.write(string.format("  \27[31m[FAIL]\27[0m %s\n       %s\n", name, tostring(err)))
    end
end

function M.skip(name, reason)
    M.n_skipped = (M.n_skipped or 0) + 1
    io.write(string.format("  \27[33m[SKIP]\27[0m %s  (%s)\n", name, reason or ""))
end

function M.eq(a, b, msg)
    if a ~= b then
        error((msg or "expected equal") .. string.format(": %s ~= %s", tostring(a), tostring(b)), 2)
    end
end

function M.neq(a, b, msg)
    if a == b then
        error((msg or "expected not equal") .. ": " .. tostring(a), 2)
    end
end

function M.ok(v, msg)
    if not v then error(msg or "expected truthy, got " .. tostring(v), 2) end
end

function M.err(fn, pattern, msg)
    local ok, err = pcall(fn)
    if ok then error((msg or "expected error, got none"), 2) end
    if pattern and not tostring(err):find(pattern) then
        error(string.format("expected error matching '%s', got: %s", pattern, err), 2)
    end
end

function M.near(a, b, tol, msg)
    tol = tol or 1e-6
    if math.abs(a - b) > tol then
        error(string.format("%s: |%g - %g| > %g", msg or "near", a, b, tol), 2)
    end
end

--- Check type
function M.is_type(v, expected, msg)
    if type(v) ~= expected then
        error(string.format("%s: expected type %s, got %s (%s)",
            msg or "type", expected, type(v), tostring(v)), 2)
    end
end

--- Check v > threshold
function M.gt(v, threshold, msg)
    if not (v > threshold) then
        error(string.format("%s: expected > %s, got %s",
            msg or "gt", tostring(threshold), tostring(v)), 2)
    end
end

--- Check v >= threshold
function M.gte(v, threshold, msg)
    if not (v >= threshold) then
        error(string.format("%s: expected >= %s, got %s",
            msg or "gte", tostring(threshold), tostring(v)), 2)
    end
end

--- Check string contains pattern
function M.contains(s, pattern, msg)
    if type(s) ~= "string" or not s:find(pattern, 1, true) then
        error(string.format("%s: '%s' not found in '%s'",
            msg or "contains", tostring(pattern), tostring(s)), 2)
    end
end

--- Check value is in a set (table of allowed values)
function M.one_of(v, set, msg)
    for _, allowed in ipairs(set) do
        if v == allowed then return end
    end
    error(string.format("%s: %s not in {%s}",
        msg or "one_of", tostring(v), table.concat(set, ", ")), 2)
end

--- Run fn and check it does NOT error
function M.no_error(fn, msg)
    local ok, err = pcall(fn)
    if not ok then
        error(string.format("%s: unexpected error: %s", msg or "no_error", tostring(err)), 2)
    end
end

function M.summary()
    local total = M.pass + M.fail
    local skipped = M.n_skipped or 0
    io.write(string.format(
        "\n%s\n  %d/%d passed",
        string.rep("─", 60), M.pass, total))
    if M.fail > 0 then
        io.write(string.format("  \27[31m%d failed\27[0m", M.fail))
    end
    if skipped > 0 then
        io.write(string.format("  %d skipped", skipped))
    end
    io.write("\n" .. string.rep("─", 60) .. "\n")
    return M.fail == 0
end

return M
