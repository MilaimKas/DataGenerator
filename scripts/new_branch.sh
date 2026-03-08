#!/usr/bin/env bash
set -euo pipefail

# This script creates a new git branch with a standardized naming convention.
# Usage: ./scripts/new_branch.sh <type> <description>

# Define some colors for output
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[info]${NC}  $*"; }
success() { echo -e "${GREEN}[ok]${NC}    $*"; }
die()     { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# list of valid branch name types
VALID_TYPES="feat fix chore docs refactor test perf"

# Read branch type and description from command line arguments
TYPE="${1:-}"
DESC="${2:-}"

# Validate inputs
[[ -z "$TYPE" || -z "$DESC" ]] && die "Usage: $0 <type> <description>  Types: $VALID_TYPES"
echo "$VALID_TYPES" | grep -qw "$TYPE" || die "Invalid type '$TYPE'."
[[ "$DESC" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || die "Description must be lowercase hyphen-separated."

# Check for uncommitted changes
[[ -n "$(git status --porcelain)" ]] && die "Working tree is dirty. Commit or stash first."

# Create new branch from main
BRANCH="${TYPE}/${DESC}"

# Fetch latest main and create new branch
git fetch -q origin main 2>/dev/null || true

# If main exists on origin, branch from it. Otherwise, branch from local main.
if git show-ref --verify --quiet "refs/remotes/origin/main"; then
  git checkout -q -b "$BRANCH" origin/main
else
  git checkout -q main && git checkout -q -b "$BRANCH"
fi

success "Switched to new branch: $BRANCH"
