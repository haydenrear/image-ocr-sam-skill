#!/usr/bin/env bash
set -euo pipefail

PROFILE="${VISION_TOOLBELT_PROFILE:-edge}"
INSTALL_MODELS=0
MODEL_CACHE="${VISION_TOOLBELT_MODEL_CACHE:-}"
PYTHON_BIN="${PYTHON:-python3}"
FORCE=0

usage() {
  cat <<'EOF'
install-vision-toolbelt.sh [--profile minimal|edge|desktop|full] [--install-models] [--model-cache PATH] [--force]

Installs the uv-backed vision-toolbelt CLI into $SKILL_MANAGER_BIN_DIR and,
when requested, downloads the selected local model bundle into the model cache.

Environment:
  SKILL_MANAGER_BIN_DIR      Set by skill-manager; wrapper is written here.
  SKILL_MANAGER_HOME         Set by skill-manager; tool env and cache live here.
  SKILL_DIR                  Set by skill-manager; skill root.
  VISION_TOOLBELT_PROFILE    Overrides --profile when set by the operator.
  VISION_TOOLBELT_MODEL_CACHE Overrides model cache path.
  VISION_TOOLBELT_NO_NETWORK  Set to 1 to skip uv/bootstrap/model downloads that require network.
  VISION_TOOLBELT_INSTALL_SYSTEM_DEPS Set to 1 to try brew/apt install tesseract.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --install-models) INSTALL_MODELS=1; shift ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${SKILL_MANAGER_BIN_DIR:-}" || -z "${SKILL_DIR:-}" || -z "${SKILL_MANAGER_HOME:-}" ]]; then
  echo "This installer must be run by skill-manager so SKILL_MANAGER_BIN_DIR, SKILL_DIR, and SKILL_MANAGER_HOME are set." >&2
  exit 2
fi

case "$PROFILE" in
  minimal|edge|desktop|full) ;;
  *) echo "invalid profile '$PROFILE' (expected minimal, edge, desktop, full)" >&2; exit 2 ;;
esac

mkdir -p "$SKILL_MANAGER_BIN_DIR"
TOOL_HOME="$SKILL_MANAGER_HOME/tools/vision-toolbelt"
VENV_DIR="$TOOL_HOME/.venv"
CLI_SRC="$SKILL_DIR/cli"
MODEL_CACHE="${MODEL_CACHE:-$SKILL_MANAGER_HOME/cache/vision-toolbelt/models}"
mkdir -p "$TOOL_HOME" "$MODEL_CACHE"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi
  if [[ "${VISION_TOOLBELT_NO_NETWORK:-0}" == "1" ]]; then
    echo "uv is not on PATH and VISION_TOOLBELT_NO_NETWORK=1; cannot bootstrap uv." >&2
    exit 3
  fi
  echo "uv not found; attempting python -m pip install --user uv" >&2
  "$PYTHON_BIN" -m pip install --user uv
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
    return 0
  fi
  echo "uv installation completed but uv is still not executable on PATH or ~/.local/bin/uv." >&2
  exit 3
}

maybe_install_system_deps() {
  if [[ "${VISION_TOOLBELT_INSTALL_SYSTEM_DEPS:-0}" != "1" ]]; then
    return 0
  fi
  if command -v tesseract >/dev/null 2>&1; then
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    brew install tesseract || true
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y tesseract-ocr || true
  else
    echo "No supported package manager found for system tesseract install; continuing." >&2
  fi
}

extras_for_profile() {
  case "$PROFILE" in
    minimal) echo "" ;;
    edge) echo "[ocr,vision]" ;;
    desktop) echo "[ocr,vision,yolo,sam]" ;;
    full) echo "[ocr,vision,yolo,sam,paddle]" ;;
  esac
}

ensure_uv
maybe_install_system_deps

EXTRAS="$(extras_for_profile)"
if [[ "$FORCE" == "1" || ! -x "$VENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

# Install the local CLI package with uv. The package remains editable against the
# installed skill source, so skill-manager sync upgrades the CLI code as well.
if [[ -n "$EXTRAS" ]]; then
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" -e "$CLI_SRC$EXTRAS"
else
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" -e "$CLI_SRC"
fi

cat > "$SKILL_MANAGER_BIN_DIR/vision-toolbelt" <<EOF
#!/usr/bin/env bash
export VISION_TOOLBELT_MODEL_CACHE="${MODEL_CACHE}"
exec "${VENV_DIR}/bin/python" -m vision_toolbelt "\$@"
EOF
chmod +x "$SKILL_MANAGER_BIN_DIR/vision-toolbelt"

if [[ "$INSTALL_MODELS" == "1" ]]; then
  if [[ "${VISION_TOOLBELT_NO_NETWORK:-0}" == "1" ]]; then
    echo "VISION_TOOLBELT_NO_NETWORK=1; skipping model download for profile $PROFILE." >&2
  else
    "$SKILL_MANAGER_BIN_DIR/vision-toolbelt" models install \
      --profile "$PROFILE" \
      --cache-dir "$MODEL_CACHE" \
      --catalog "$SKILL_DIR/references/model-catalog.toml" || {
        echo "Model installation failed; CLI is installed. Re-run: vision-toolbelt models install --profile $PROFILE --cache-dir '$MODEL_CACHE'" >&2
      }
  fi
fi

echo "vision-toolbelt installed at $SKILL_MANAGER_BIN_DIR/vision-toolbelt"
echo "model cache: $MODEL_CACHE"
