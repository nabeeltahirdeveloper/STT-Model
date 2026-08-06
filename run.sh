#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Roman Urdu Auto-Captioning — task runner
#  Usage: ./run.sh <command> [args]
#         ./run.sh help
# ---------------------------------------------------------------------------
set -Eeuo pipefail

PROJECT_NAME="roman-urdu-captions"
PYTHON_VERSION="3.12"
VENV_DIR=".venv"
SRC_DIRS="src tests scripts"

# ------------------------------- colours -----------------------------------
if [[ -t 1 ]] && [[ "${NO_COLOR:-}" == "" ]] && command -v tput >/dev/null 2>&1 && [[ $(tput colors 2>/dev/null || echo 0) -ge 8 ]]; then
  C_RESET=$'\033[0m';  C_BOLD=$'\033[1m';   C_DIM=$'\033[2m'
  C_RED=$'\033[0;31m'; C_GREEN=$'\033[0;32m'; C_YELLOW=$'\033[0;33m'
  C_BLUE=$'\033[0;34m'; C_MAGENTA=$'\033[0;35m'; C_CYAN=$'\033[0;36m'
else
  C_RESET=""; C_BOLD=""; C_DIM=""
  C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_MAGENTA=""; C_CYAN=""
fi

# ------------------------------ log helpers --------------------------------
info()  { printf '%s[ INFO ]%s %s\n'  "$C_BLUE"    "$C_RESET" "$*"; }
ok()    { printf '%s[  OK  ]%s %s\n'  "$C_GREEN"   "$C_RESET" "$*"; }
warn()  { printf '%s[ WARN ]%s %s\n'  "$C_YELLOW"  "$C_RESET" "$*" >&2; }
err()   { printf '%s[ FAIL ]%s %s\n'  "$C_RED"     "$C_RESET" "$*" >&2; }
step()  { printf '%s[ STEP ]%s %s%s%s\n' "$C_MAGENTA" "$C_RESET" "$C_BOLD" "$*" "$C_RESET"; }
run()   { printf '%s[  ->  ]%s %s%s%s\n' "$C_CYAN" "$C_RESET" "$C_DIM" "$*" "$C_RESET"; "$@"; }

hr() { printf '%s%s%s\n' "$C_DIM" "$(printf '─%.0s' $(seq 1 68))" "$C_RESET"; }

banner() {
  hr
  printf '%s%s  %s%s\n' "$C_BOLD" "$C_CYAN" "$PROJECT_NAME" "$C_RESET"
  hr
}

die() { err "$*"; exit 1; }

on_error() {
  local exit_code=$?
  local line=${BASH_LINENO[0]}
  err "aborted at line ${line} (exit ${exit_code})"
  exit "$exit_code"
}
trap on_error ERR

# ------------------------------ preflight ----------------------------------
need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

activate_venv() {
  if [[ -d "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  else
    warn "no virtualenv at $VENV_DIR — run './run.sh setup' first"
  fi
}

# ------------------------------- commands ----------------------------------
cmd_setup() {
  banner
  step "Environment setup"

  need_cmd python3
  need_cmd git

  local py_ver
  py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  info "python ${py_ver} detected (target ${PYTHON_VERSION})"
  [[ "$py_ver" == "$PYTHON_VERSION" ]] || warn "python ${PYTHON_VERSION} recommended; continuing with ${py_ver}"

  if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg found"
  else
    warn "ffmpeg NOT found — required for audio extraction"
    info "  macOS:  brew install ffmpeg"
    info "  Ubuntu: sudo apt install ffmpeg"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    ok "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
      info "  $line"
    done
  else
    warn "no NVIDIA GPU — training will not be possible locally"
  fi

  step "Creating virtualenv"
  [[ -d "$VENV_DIR" ]] && info "reusing existing $VENV_DIR" || run python3 -m venv "$VENV_DIR"
  activate_venv

  step "Installing dependencies"
  run python -m pip install --upgrade pip setuptools wheel
  run python -m pip install -e ".[dev]"

  step "Creating directory tree"
  run mkdir -p data/raw data/labels data/eval data/lexicon data/work
  run mkdir -p src/{ingest,preprocess,labeling,training,inference,subtitle,api}
  run mkdir -p tests scripts notebooks configs docs

  if [[ ! -d .git ]]; then
    step "Initialising git"
    run git init -q
  fi

  if command -v pre-commit >/dev/null 2>&1; then
    run pre-commit install
  fi

  hr
  ok "setup complete"
  info "next: ${C_BOLD}./run.sh check${C_RESET}"
}

cmd_lint() {
  activate_venv
  step "Lint (ruff)"
  # shellcheck disable=SC2086
  run ruff check $SRC_DIRS
  ok "lint passed"
}

cmd_format() {
  activate_venv
  step "Format (ruff)"
  # shellcheck disable=SC2086
  run ruff format $SRC_DIRS
  # shellcheck disable=SC2086
  run ruff check --fix $SRC_DIRS
  ok "formatted"
}

cmd_typecheck() {
  activate_venv
  step "Type check (mypy)"
  run mypy src
  ok "types passed"
}

cmd_test() {
  activate_venv
  step "Tests (pytest)"
  run pytest -q --cov=src --cov-report=term-missing "$@"
  ok "tests passed"
}

cmd_check() {
  banner
  info "running all quality gates"
  cmd_lint
  cmd_typecheck
  cmd_test
  hr
  ok "${C_BOLD}all gates green${C_RESET}"
}

cmd_lexicon() {
  activate_venv
  step "Building frequency lexicon from Roman-Urdu-Parl"
  [[ -d data/raw/roman-urdu-parl ]] || die "corpus missing: data/raw/roman-urdu-parl"
  run python -m scripts.build_lexicon \
    --corpus data/raw/roman-urdu-parl \
    --top-n 5000 \
    --out docs/lexicon-frequency.tsv
  ok "lexicon written to docs/lexicon-frequency.tsv"
}

cmd_transcribe() {
  activate_venv
  [[ $# -ge 1 ]] || die "usage: ./run.sh transcribe <video-file> [--out DIR]"
  step "Transcribing: $1"
  run python -m src.api.pipeline --input "$@"
  ok "done"
}

cmd_train() {
  activate_venv
  local config="${1:-configs/phase1.yaml}"
  [[ -f "$config" ]] || die "config not found: $config"
  step "Training with $config"
  warn "verify data/eval/ is NOT in the training manifest before proceeding"
  run python -m src.training.finetune --config "$config"
  ok "training complete"
}

cmd_eval() {
  activate_venv
  step "Evaluating"
  run python -m src.eval.score \
    --pred "${1:-out/predictions.txt}" \
    --ref  "${2:-data/eval/reference.txt}" \
    --metrics cer,sn-wer,normalized-wer,english-preservation
  ok "evaluation complete"
}

cmd_serve() {
  activate_venv
  local model="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
  local port="${PORT:-8000}"
  step "Serving $model on port $port"
  run qwen-asr-serve "$model" --gpu-memory-utilization 0.8 --host 0.0.0.0 --port "$port"
}

cmd_clean() {
  step "Cleaning build artefacts"
  run find . -type d -name __pycache__ -prune -exec rm -rf {} +
  run rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
  ok "cleaned (data/ and .venv/ untouched)"
}

cmd_doctor() {
  banner
  step "Diagnostics"
  local issues=0
  _chk() {
    if eval "$2" >/dev/null 2>&1; then ok "$1"; else warn "$1 — MISSING"; issues=$((issues+1)); fi
  }
  _chk "python3"        "command -v python3"
  _chk "ffmpeg"         "command -v ffmpeg"
  _chk "git"            "command -v git"
  _chk "NVIDIA GPU"     "command -v nvidia-smi"
  _chk "virtualenv"     "test -d $VENV_DIR"
  _chk "eval set"       "test -s data/eval/reference.txt"
  _chk "canonical lexicon" "test -s data/lexicon/canonical.tsv"
  _chk "english lexicon"   "test -s data/lexicon/english.txt"
  hr
  if [[ $issues -eq 0 ]]; then ok "no issues"; else warn "$issues issue(s) found"; fi
}

cmd_help() {
  banner
  cat <<EOF
${C_BOLD}USAGE${C_RESET}
  ./run.sh <command> [args]

${C_BOLD}SETUP${C_RESET}
  ${C_GREEN}setup${C_RESET}        Create venv, install deps, scaffold directories
  ${C_GREEN}doctor${C_RESET}       Diagnose missing tools and data

${C_BOLD}QUALITY${C_RESET}
  ${C_GREEN}check${C_RESET}        Run lint + typecheck + test (use before every commit)
  ${C_GREEN}lint${C_RESET}         ruff check
  ${C_GREEN}format${C_RESET}       ruff format + autofix
  ${C_GREEN}typecheck${C_RESET}    mypy
  ${C_GREEN}test${C_RESET}         pytest with coverage

${C_BOLD}PIPELINE${C_RESET}
  ${C_GREEN}lexicon${C_RESET}      Build frequency lexicon from Roman-Urdu-Parl
  ${C_GREEN}transcribe${C_RESET}   Transcribe a video file
  ${C_GREEN}train${C_RESET}        Fine-tune  (default configs/phase1.yaml)
  ${C_GREEN}eval${C_RESET}         Score predictions (CER, SN-WER, ...)
  ${C_GREEN}serve${C_RESET}        Start vLLM inference server

${C_BOLD}MISC${C_RESET}
  ${C_GREEN}clean${C_RESET}        Remove caches and build artefacts
  ${C_GREEN}help${C_RESET}         This message

${C_BOLD}ENVIRONMENT${C_RESET}
  MODEL=<hf-id>   override serving model
  PORT=<n>        override serving port
  NO_COLOR=1      disable coloured output
EOF
}

# ------------------------------- dispatch ----------------------------------
main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    setup)       cmd_setup "$@" ;;
    doctor)      cmd_doctor "$@" ;;
    lint)        cmd_lint "$@" ;;
    format|fmt)  cmd_format "$@" ;;
    typecheck)   cmd_typecheck "$@" ;;
    test)        cmd_test "$@" ;;
    check)       cmd_check "$@" ;;
    lexicon)     cmd_lexicon "$@" ;;
    transcribe)  cmd_transcribe "$@" ;;
    train)       cmd_train "$@" ;;
    eval)        cmd_eval "$@" ;;
    serve)       cmd_serve "$@" ;;
    clean)       cmd_clean "$@" ;;
    help|-h|--help) cmd_help ;;
    *)           err "unknown command: $cmd"; echo; cmd_help; exit 1 ;;
  esac
}

main "$@"
