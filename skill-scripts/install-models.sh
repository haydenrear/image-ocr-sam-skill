#!/usr/bin/env bash
set -euo pipefail
PROFILE="${1:-${VISION_TOOLBELT_PROFILE:-edge}}"
CACHE="${VISION_TOOLBELT_MODEL_CACHE:-${SKILL_MANAGER_HOME:-$HOME/.skill-manager}/cache/vision-toolbelt/models}"
CATALOG="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/references/model-catalog.toml"
vision-toolbelt models install --profile "$PROFILE" --cache-dir "$CACHE" --catalog "$CATALOG"
