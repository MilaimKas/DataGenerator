#!/usr/bin/env bash
set -euo pipefail

# Define some colors for output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[info]${NC}  $*"; }
success() { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()     { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }

ERRORS=0

# define a helper function to run checks and count errors
run_check() { local label="$1"; shift; info "Running: $label"; if "$@"; then success "$label passed."; else warn "$label FAILED."; ((ERRORS++)) || true; fi; }

# Run checks using uv to ensure the correct environment is used. If any check fails, it will be counted and reported at the end.
echo "════════════════════════════════════════"; echo " Pre-push checks"; echo "════════════════════════════════════════"
run_check "ruff format" uv run ruff format --check .
run_check "ruff lint"   uv run ruff check .
run_check "ty check"    uv run ty check
run_check "pytest"      uv run pytest --tb=short -q
echo ""

[[ $ERRORS -eq 0 ]] && success "All pre-push checks passed." || die "$ERRORS check(s) failed."
