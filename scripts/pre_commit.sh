#!/usr/bin/env bash
set -euo pipefail

# This script runs pre-commit checks: code formatting, linting, and type checking.
# If --fix is provided, it will auto-format and auto-fix lint issues. Otherwise, it just checks and reports errors.

# define some colors for output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[info]${NC}  $*"; }
success() { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()     { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }

# Check for --fix flag
FIX="${1:-}"; 
ERRORS=0

# Define a helper function to run checks and count errors
run_check() { local label="$1"; shift; info "Running: $label"; if "$@"; then success "$label passed."; else warn "$label FAILED."; ((ERRORS++)) || true; fi; }

# If --fix is provided, auto-format and auto-fix lint issues. Otherwise, just check formatting and lint.
if [[ "$FIX" == "--fix" ]]; then
  info "Auto-formatting …"; uv run ruff format .
  info "Auto-fixing lint …"; uv run ruff check . --fix
else
  run_check "ruff format" uv run ruff format --check .
  run_check "ruff lint"   uv run ruff check .
fi
run_check "ty type check" uv run ty check
echo ""

# If there were any errors, exit with failure. Otherwise, print success message.
[[ $ERRORS -eq 0 ]] && success "All pre-commit checks passed." || die "$ERRORS check(s) failed."
