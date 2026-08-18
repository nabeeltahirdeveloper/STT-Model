#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Roman Urdu Auto-Captioning — task runner
#  Usage: ./run.sh <command> [args]
#         ./run.sh help
# ---------------------------------------------------------------------------
set -Eeuo pipefail

PROJECT_NAME="roman-urdu-captions"
PYTHON_VERSION="3.12"
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

# Every tool runs through `uv run`, which resolves and syncs the project
# environment on demand. There is deliberately no activation step: an activated
# shell can silently point at the wrong interpreter, and that failure is
# invisible until a version-dependent bug appears in CI and not locally.

# ------------------------------- commands ----------------------------------
cmd_setup() {
  banner
  step "Environment setup"

  need_cmd uv
  need_cmd git

  info "uv $(uv --version | awk '{print $2}')"

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

  step "Provisioning Python ${PYTHON_VERSION}"
  run uv python install "$PYTHON_VERSION"

  # Two passes. flash-attn's build imports torch to read the CUDA version, so it
  # cannot build in an isolated environment ([tool.uv] no-build-isolation-package)
  # and torch must already be installed when it builds. Pass one gets torch in
  # place; pass two builds flash-attn against it. On a machine without CUDA the
  # `cuda` extra is marker-gated to Linux and pass two is a no-op.
  step "Installing dependencies (pass 1 — everything except flash-attn)"
  run uv sync --all-extras --all-groups --no-extra cuda

  step "Installing dependencies (pass 2 — flash-attn)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    info "CUDA detected — building flash-attn without build isolation"
    info "  set MAX_JOBS=4 if this machine has under 96 GB RAM"
  else
    info "no CUDA — flash-attn is skipped by its platform marker"
  fi
  run uv sync --all-extras --all-groups

  step "Creating directory tree"
  run mkdir -p data/raw data/labels data/eval data/lexicon data/work
  run mkdir -p src/{ingest,preprocess,labeling,training,inference,subtitle,api}
  run mkdir -p tests scripts notebooks configs docs

  if [[ ! -d .git ]]; then
    step "Initialising git"
    run git init -q
  fi

  step "Installing git hooks"
  run uv run pre-commit install

  hr
  ok "setup complete"
  info "next: ${C_BOLD}./run.sh check${C_RESET}"
}

cmd_lint() {
  step "Lint (ruff)"
  # shellcheck disable=SC2086
  run uv run ruff check $SRC_DIRS
  ok "lint passed"
}

cmd_format() {
  step "Format (ruff)"
  # shellcheck disable=SC2086
  run uv run ruff format $SRC_DIRS
  # shellcheck disable=SC2086
  run uv run ruff check --fix $SRC_DIRS
  ok "formatted"
}

cmd_typecheck() {
  step "Type check (mypy)"
  run uv run mypy src
  ok "types passed"
}

cmd_test() {
  step "Tests (pytest)"
  run uv run pytest -q --cov=src --cov-report=term-missing "$@"
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
  step "Building frequency lexicon from Roman-Urdu-Parl"
  [[ -d data/raw/roman-urdu-parl ]] || die "corpus missing: data/raw/roman-urdu-parl"
  run uv run python -m scripts.build_lexicon \
    --corpus data/raw/roman-urdu-parl \
    --top-n 5000 \
    --out docs/lexicon-frequency.tsv
  ok "lexicon written to docs/lexicon-frequency.tsv"
}

cmd_transcribe() {
  [[ $# -ge 1 ]] || die "usage: ./run.sh transcribe <video-file> [--out DIR]"
  step "Transcribing: $1"
  run uv run python -m src.api.pipeline "$@"
  ok "done"
}

# Build every manifest the training run needs, in the one order that is
# correct: the dev set must exist before the subset, so the subset can exclude
# it. A dev CER measured on memorised audio reports recall, not generalisation.
cmd_data() {
  local hours="${1:-20}"
  step "Transcripts"
  run uv run python -m scripts.download_transcripts --corpus-transcripts
  step "Labels"
  run uv run python -m scripts.build_labels --split US-CS
  step "Dev set (held out of training)"
  run uv run python -m scripts.build_dev_set --clips 150
  step "Training subset: ${hours} h, excluding the dev set"
  run uv run python -m scripts.select_training_subset --hours "$hours" \
      --exclude data/labels/dev-set.jsonl
  step "Audio (one archive, not 13,470 requests)"
  run uv run python -m scripts.fetch_training_audio \
      --manifest data/labels/train-subset.jsonl \
      --archive "${AUDIO_ARCHIVE:-MubeenAmjad205/roman-urdu-captions-audio}"
  ok "data ready -- now run ./run.sh rehearse"
}

# Four minutes that exercise every line the real run executes, generation
# included. Two bugs cost full sessions because the old rehearsal only trained.
cmd_rehearse() {
  local base="${BASE:-Qwen/Qwen3-ASR-0.6B}"
  step "Rehearsal on ~20 clips with $base"
  run uv run python -m scripts.select_training_subset --hours 0.05 \
      --out data/labels/rehearsal.jsonl
  run uv run python -m scripts.fetch_training_audio \
      --manifest data/labels/rehearsal.jsonl
  run uv run python -m src.training.finetune \
      --manifest data/labels/rehearsal.jsonl --base "$base" \
      --limit 22 --accumulate 4 --probe-every 5 --out out/rehearsal
  ok "read the GB peak, p(eos) and probe output before training for real"
  warn "the rehearsal uses the SHORTEST clips -- the real run will use more memory"
}

cmd_train() {
  local base="${BASE:-Qwen/Qwen3-ASR-0.6B}"
  local out="${OUT:-out/finetuned}"
  [[ -f data/labels/train-subset.jsonl ]] || die "no manifest -- run ./run.sh data first"
  step "Full fine-tune: $base -> $out"
  warn "data/eval/ must never appear in the training manifest (constraint 5)"
  local extra=()
  [[ -n "${HF_REPO:-}" ]] && extra+=(--hf-repo "$HF_REPO" --push-every 300)
  [[ -n "${MAX_MINUTES:-}" ]] && extra+=(--max-minutes "$MAX_MINUTES")
  [[ -n "${RESUME:-}" ]] && extra+=(--resume "$RESUME")
  run uv run python -m src.training.finetune \
      --base "$base" --out "$out" \
      --dev-manifest data/labels/dev-set.jsonl --dev-every 150 \
      "${extra[@]}" "$@"
  ok "training complete -- now run ./run.sh eval"
}

# Transcribe the 269-clip benchmark, then score it. Both halves: a fresh clone
# has the references but not the audio, and scoring predictions that were never
# generated is the failure this command exists to prevent.
cmd_eval() {
  local model="${OUT:-out/finetuned}"
  local preds="out/preds"
  [[ -d "$model" || "$model" == */* ]] || die "no model at $model -- set OUT=<dir|hf-id>"

  step "Eval audio (gitignored, so a fresh clone lacks it)"
  run uv run python -m scripts.fetch_training_audio --manifest data/eval/manifest.jsonl

  step "Transcribing 269 clips with $model"
  # --no-romanize-output: a fine-tuned model already emits Roman. Sending it
  # through the Devanagari converter again rewrites correct spellings and
  # measures the converter. Omit the flag only for the stock-model baseline.
  run uv run python -m scripts.run_baseline \
      --model-id "$model" --out-dir "$preds" --no-romanize-output \
      --device "${DEVICE:-cuda}" --batch-size "${BATCH:-4}"

  step "Scoring"
  run uv run python -m src.eval.score --pred "$preds/baseline-raw.txt"
  ok "34.9% is the baseline to beat -- stock 0.6B plus our romanizer"
}

cmd_serve() {
  local model="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
  local port="${PORT:-8000}"
  step "Serving $model on port $port"
  run uv run qwen-asr-serve "$model" --gpu-memory-utilization 0.8 --host 0.0.0.0 --port "$port"
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
  # `|| status=$?` rather than `if eval ...` on purpose. Under `set -Eeuo
  # pipefail` the ERR trap fires on the failing eval, and because that line
  # redirects 2>&1 the trap's own message is swallowed -- doctor exited 1 after
  # the first missing tool, silently, which is the opposite of what a diagnostic
  # command should do.
  _chk() {
    local status=0
    ( set +eE; trap - ERR; eval "$2" ) >/dev/null 2>&1 || status=$?
    if [[ $status -eq 0 ]]; then ok "$1"; else warn "$1 — MISSING"; issues=$((issues+1)); fi
  }
  _chk "uv"             "command -v uv"
  _chk "ffmpeg"         "command -v ffmpeg"
  _chk "git"            "command -v git"
  _chk "NVIDIA GPU"     "command -v nvidia-smi"
  _chk "uv.lock"        "test -s uv.lock"
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
  ${C_GREEN}setup${C_RESET}        Provision Python, sync deps, scaffold directories
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
  ${C_GREEN}serve${C_RESET}        Start vLLM inference server

${C_BOLD}TRAINING${C_RESET}  (in this order -- see ${C_BOLD}docs/TRAINING.md${C_RESET})
  ${C_GREEN}data${C_RESET} [hours]  Transcripts, labels, dev set, subset, audio  (default 20 h)
  ${C_GREEN}rehearse${C_RESET}     4-minute dry run: train AND generate. ${C_BOLD}Always do this first${C_RESET}
  ${C_GREEN}train${C_RESET}        Full fine-tune with mid-run CER on held-out clips
  ${C_GREEN}eval${C_RESET}         Score against the 269-clip benchmark (CER, SN-WER)

${C_BOLD}MISC${C_RESET}
  ${C_GREEN}clean${C_RESET}        Remove caches and build artefacts
  ${C_GREEN}help${C_RESET}         This message

${C_BOLD}ENVIRONMENT${C_RESET}
  MODEL=<hf-id>   override serving model
  PORT=<n>        override serving port
  BASE=<hf-id>    model to fine-tune       (default Qwen/Qwen3-ASR-0.6B)
  OUT=<dir>       checkpoint directory     (default out/finetuned)
  HF_REPO=<id>    push checkpoints here during the run
  MAX_MINUTES=<n> time-box; saves and exits cleanly
  RESUME=<dir>    continue from a checkpoint
  DEVICE=<dev>    eval device: cuda | mps | cpu   (default cuda)
  BATCH=<n>       eval batch size; lower if OOM   (default 4)
  HF_TOKEN=<tok>  required: the corpus is a gated dataset
  NO_COLOR=1      disable coloured output
  MAX_JOBS=<n>    limit flash-attn build parallelism (use 4 if <96GB RAM)

${C_BOLD}NOTES${C_RESET}
  Dependencies are managed with uv. Every command runs through ${C_BOLD}uv run${C_RESET},
  which syncs the environment on demand — there is no venv to activate.
  Use ${C_BOLD}uv add${C_RESET} / ${C_BOLD}uv add --dev${C_RESET} to change dependencies, then commit uv.lock.

${C_BOLD}FIRST TIME TRAINING?${C_RESET}
  Read ${C_BOLD}docs/TRAINING.md${C_RESET}. The short version:

    export HF_TOKEN=hf_...
    ./run.sh setup && ./run.sh check
    ./run.sh data 20
    ./run.sh rehearse            ${C_DIM}# read GB peak before going further${C_RESET}
    MAX_MINUTES=180 ./run.sh train
    ./run.sh eval

  ${C_BOLD}Never${C_RESET} train on data/eval/. ${C_BOLD}Never${C_RESET} quote raw WER alone. ${C_BOLD}Always${C_RESET} rehearse first.
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
    data)        cmd_data "$@" ;;
    rehearse)    cmd_rehearse "$@" ;;
    train)       cmd_train "$@" ;;
    eval)        cmd_eval "$@" ;;
    serve)       cmd_serve "$@" ;;
    clean)       cmd_clean "$@" ;;
    help|-h|--help) cmd_help ;;
    *)           err "unknown command: $cmd"; echo; cmd_help; exit 1 ;;
  esac
}

main "$@"
