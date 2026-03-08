#!/usr/bin/env bash
set -euo pipefail

# This script builds and publishes the package to PyPI. 
# It performs checks to ensure that the version is not already published, the working tree is clean, and that the user is on the main branch. 
# It also supports a dry run mode to list the files that would be uploaded without actually publishing.

# define output  colors 
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[info]${NC}  $*"; }
success() { echo -e "${GREEN}[ok]${NC}    $*"; }
die()     { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# read command line argument for dry run option
DRY_RUN="${1:-}"

# check if main branch and clean working tree
[[ "$(git branch --show-current)" == "main" ]] || die "Must be on main to publish."
[[ -z "$(git status --porcelain)" ]] || die "Working tree is dirty."

# run pre-publish checks
bash "$(dirname "$0")/pre_push.sh"

# read version from pyproject..toml
VERSION=$(grep '^version' pyproject.toml | head -1 | sed 's/.*= *"\(.*\)"/\1/')
info "Version: $VERSION"

# check if version tag already exists
git tag | grep -q "^v${VERSION}$" && die "Tag v${VERSION} already exists. Bump version first."

# remove old builds
rm -rf dist/ && uv build

# if dry run, just list the files that would be uploaded
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "Dry run — skipping upload."; ls -lh dist/

# upload and publish to PyPI
else
  [[ -n "${PYPI_TOKEN:-}" ]] || die "PYPI_TOKEN not set."
  UV_PUBLISH_TOKEN="$PYPI_TOKEN" uv publish
  git tag -a "v${VERSION}" -m "Release v${VERSION}" && git push origin "v${VERSION}"
  success "Published v${VERSION}."
fi
