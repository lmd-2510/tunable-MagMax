#!/usr/bin/env bash
# Agreed single-seed experiment. Calls upstream training/merging unchanged.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACTION="${1:-status}"
PYTHON_BIN="${TUNABLE_PYTHON:-python}"
GPU_ID="${TUNABLE_GPU_ID:-1}"
OUT="$ROOT/outs/tunable_CIFAR100_20_A_seed5"
export MAGMAX_BASE_DIR="$ROOT"
export MAGMAX_DATA_DIR="${MAGMAX_DATA_DIR:-$(dirname "$ROOT")/MagMax/datasets}"
OPENCLIP_CACHE="${TUNABLE_OPENCLIP_CACHE:-$(dirname "$ROOT")/MagMax/checkpoints/ViT-B-16/cachedir/open_clip}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
CKPTS="$ROOT/checkpoints/ViT-B-16/sequential_finetuning/class_incremental/CIFAR100-20/ft-pattern_A-epochs-10-seed:5"
die() { echo "[STOP] $*" >&2; exit 1; }
[[ "$GPU_ID" == 0 || "$GPU_ID" == 1 ]] || die 'TUNABLE_GPU_ID must be 0 or 1.'
[[ "${TORCH_FORCE_WEIGHTS_ONLY_LOAD:-0}" != 1 ]] || die 'Unset TORCH_FORCE_WEIGHTS_ONLY_LOAD for upstream full-module checkpoints.'
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ARGS=(--model ViT-B-16 --dataset CIFAR100 --n_splits 20 --split_strategy class
      --sequential-finetuning --epochs 10 --seed 5 --taskseq_pattern A
      --batch_size 128 --lr 1e-5 --wd 0.1 --ls 0.0 --warmup_length 500
      --gpu_id 0 --data_location "$MAGMAX_DATA_DIR" --openclip-cachedir "$OPENCLIP_CACHE")
MERGE_ARGS=(--target_config target_data_config_main25 --num_target_data 500
            --num_train_data_each_task 500 --similarity_metric labels
            --results_db "$OUT/results")

checkpoint_count() {
  local count=0 idx
  for idx in {0..19}; do
    [[ ! -s "$CKPTS/finetuned_${idx}.pt" ]] || count=$((count + 1))
  done
  printf '%s\n' "$count"
}
check_training_state() {
  local count
  count="$(checkpoint_count)"
  echo "[CHECKPOINTS] $count/20 in $CKPTS"
  (( count == 0 || count == 20 )) || die 'Partial checkpoint set: upstream does not restore RNG for an exact resume. Preserve the directory before choosing a fresh run.'
  if (( count == 20 )); then
    [[ -s "$ROOT/checkpoints/ViT-B-16/zeroshot.pt" ]] || die 'Complete task set requires its original zeroshot.pt.'
  fi
}
check_environment() {
  TUNABLE_PYTHON="$PYTHON_BIN" TUNABLE_GPU_ID="$GPU_ID" bash scripts/setup_mica.sh check
  echo '[CONFIG] CIFAR100/20; ViT-B-16; pattern A; seed5; 10 epochs/task; batch128; lr1e-5; wd0.1; warmup500; FP32'
  echo '[MERGE] Label + MagMax; lambda0.5; 25 targets; 500 samples/target; 500 train samples/task for similarity'
  echo "[GPU] physical=$GPU_ID -> process cuda:0"
  echo "[DATA] $MAGMAX_DATA_DIR"
  echo "[PRETRAINED CACHE] $OPENCLIP_CACHE"
  if [[ -s "$MAGMAX_DATA_DIR/cifar-100-python/train" && -s "$MAGMAX_DATA_DIR/cifar-100-python/test" ]]; then
    echo '[DATA] CIFAR100 files present; torchvision verifies integrity at load.'
  else
    echo '[DATA] Not ready. start will prepare the dataset in the background before training.'
  fi
  df -h "$ROOT"
  free -h
}
checksum_ok() {
  [[ -s "$1" ]] && printf '%s  %s\n' "$2" "$1" | md5sum --status -c -
}
cifar_data_ready() {
  checksum_ok "$MAGMAX_DATA_DIR/cifar-100-python/train" 16019d7e3df5f24257cddd939b257f8d &&
  checksum_ok "$MAGMAX_DATA_DIR/cifar-100-python/test" f0ef6b0ae62326f3e7ffdfab6717acfc &&
  checksum_ok "$MAGMAX_DATA_DIR/cifar-100-python/meta" 7973b15100ade9c7d40fb424638fde48
}
magmax_job_active() {
  local other_out="$(dirname "$ROOT")/MagMax/outs/magmax_CIFAR100_20_seed5" pid
  # The original MagMax launcher does not use the dataset lock. Avoid writing
  # its dataset while that job is downloading/extracting into the shared root.
  [[ "$MAGMAX_DATA_DIR" -ef "$(dirname "$ROOT")/MagMax/datasets" ]] || return 1
  [[ -s "$other_out/job.pid" && ! -e "$other_out/exit_code.txt" ]] || return 1
  pid="$(<"$other_out/job.pid")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}
prepare_data() {
  local archive="$MAGMAX_DATA_DIR/cifar-100-python.tar.gz"
  local digest=eb9058c3a382ffc7106e4002c42a8d85
  local url=https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
  mkdir -p "$MAGMAX_DATA_DIR"
  command -v md5sum >/dev/null || die 'md5sum is required for dataset integrity checks.'
  exec 8> "$MAGMAX_DATA_DIR/.cifar100.download.lock"
  echo '[DATA] Acquiring dataset lock.'
  flock 8
  while ! cifar_data_ready && magmax_job_active; do
    echo '[DATA] Waiting for the existing MagMax download/extraction. Training starts automatically when ready.'
    sleep 30
  done
  if ! cifar_data_ready; then
    if command -v curl >/dev/null; then
      if ! checksum_ok "$archive" "$digest"; then
        echo '[DATA] Downloading official CIFAR100; attempting to resume the existing archive.'
        if ! curl -fL --connect-timeout 30 --retry 5 --retry-delay 5 -C - -o "$archive" "$url" || ! checksum_ok "$archive" "$digest"; then
          # Preserve the old partial/corrupt archive if Range is unsupported.
          archive="$MAGMAX_DATA_DIR/cifar-100-python.tunable-fresh.tar.gz"
          echo '[DATA] Resume unavailable or checksum invalid; downloading a fresh copy.'
          curl -fL --connect-timeout 30 --retry 5 --retry-delay 5 -o "$archive" "$url"
        fi
      fi
      checksum_ok "$archive" "$digest" || die 'Downloaded archive failed CIFAR100 MD5 verification.'
      echo '[DATA] Archive checksum passed. Extracting.'
      tar -xzf "$archive" -C "$MAGMAX_DATA_DIR"
    else
      echo '[DATA] curl unavailable; torchvision will download/extract the dataset.'
    fi
  fi
  # Final validation uses torchvision, exactly like upstream data loading.
  # This stage uses CPU only and finishes before model training is launched.
  env CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - <<'PY'
import os
from torchvision.datasets import CIFAR100
root = os.environ['MAGMAX_DATA_DIR']
train = CIFAR100(root=root, train=True, download=True)
test = CIFAR100(root=root, train=False, download=True)
assert len(train) == 50000 and len(test) == 10000
print(f'[DATA READY] Train={len(train)}, test={len(test)}; root={root}', flush=True)
PY
  flock -u 8
  exec 8>&-
}
set_stage() {
  printf '%s\n' "$1" > "$OUT/stage.txt"
  echo "[STAGE] $1"
}
record_command() {
  printf '%q ' "$@" >> "$OUT/commands.txt"
  printf '\n' >> "$OUT/commands.txt"
}
run_job() {
  mkdir -p "$OUT"
  # A start job inherits the parent's flock descriptor across nohup/exec.
  # Direct run also obtains the same lock, so it cannot overlap another job.
  if [[ "${TUNABLE_LOCK_HELD:-0}" != 1 ]]; then
    exec 9> "$OUT/launcher.lock"
    flock -n 9 || die 'A Tunable job is already active.'
  fi
  trap 'code=$?; printf "%s\n" "$code" > "$OUT/exit_code.txt"' EXIT
  check_environment
  check_training_state
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  "$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
  cp outs/mica_setup/environment.json "$OUT/runtime.json"
  git rev-parse HEAD > "$OUT/source_commit.txt"
  printf '%s\n' "$GPU_ID" > "$OUT/gpu.txt"
  printf '%s\n' "$MAGMAX_DATA_DIR" > "$OUT/data_dir.txt"
  cp configs/target_data_config_main25.json "$OUT/target_config.json"
  : > "$OUT/commands.txt"
  local began=$SECONDS
  set_stage dataset
  prepare_data 2>&1 | tee -a "$OUT/download.log"
  if [[ "${TUNABLE_STAGE:-all}" == all ]]; then
    set_stage train
    record_command "$PYTHON_BIN" -u finetune_splitted.py "${ARGS[@]}"
    "$PYTHON_BIN" -u finetune_splitted.py "${ARGS[@]}" 2>&1 | tee -a "$OUT/train.log"
  fi
  [[ "$(checkpoint_count)" == 20 ]] || die 'Training did not produce all 20 task checkpoints.'
  # Labels use the labelled meta subset. Both methods evaluate identical targets.
  set_stage label
  record_command "$PYTHON_BIN" -u merge_for_targetdata.py "${ARGS[@]}" "${MERGE_ARGS[@]}" --merge_fn masked_magmax_with_targetdata
  "$PYTHON_BIN" -u merge_for_targetdata.py "${ARGS[@]}" "${MERGE_ARGS[@]}" --merge_fn masked_magmax_with_targetdata 2>&1 | tee -a "$OUT/label.log"
  set_stage magmax
  record_command "$PYTHON_BIN" -u merge_for_targetdata.py "${ARGS[@]}" "${MERGE_ARGS[@]}" --merge_fn magmax
  "$PYTHON_BIN" -u merge_for_targetdata.py "${ARGS[@]}" "${MERGE_ARGS[@]}" --merge_fn magmax 2>&1 | tee -a "$OUT/magmax.log"
  set_stage summary
  "$PYTHON_BIN" scripts/summarize_mica.py --run-dir "$OUT"
  printf '%s\n' "$((SECONDS - began))" > "$OUT/total_seconds.txt"
  set_stage done
  echo "[DONE] $OUT/summary.csv and summary.json"
}

case "$ACTION" in
  check) check_environment; check_training_state ;;
  start|merge)
    command -v flock >/dev/null || die 'Linux flock is required.'
    mkdir -p "$OUT"
    PYTHON_BIN="$(command -v "$PYTHON_BIN")" || die 'Activate ai_env first.'
    exec 9> "$OUT/launcher.lock"
    flock -n 9 || die 'A Tunable job is already active.'
    check_environment
    check_training_state
    STAGE=all
    if [[ "$ACTION" == merge ]]; then
      [[ "$(checkpoint_count)" == 20 ]] || die 'merge requires all 20 completed checkpoints.'
      STAGE=merge
    else
      if [[ -e "$OUT/exit_code.txt" ]]; then
        if [[ "$(checkpoint_count)" == 0 && -s "$OUT/stage.txt" && "$(<"$OUT/stage.txt")" == dataset && "$(<"$OUT/exit_code.txt")" != 0 ]]; then
          echo '[DATA] Retrying a failed preparation stage; no task training has started.'
        else
          die "Previous launch exists: inspect status; use 'merge' if training completed."
        fi
      fi
    fi
    # Clear a previous merge failure only after preflight succeeds; log is appended.
    if [[ -e "$OUT/exit_code.txt" ]]; then
      mv "$OUT/exit_code.txt" "$OUT/exit_code.previous.$(date +%s).txt"
    fi
    nohup env TUNABLE_GPU_ID="$GPU_ID" TUNABLE_PYTHON="$PYTHON_BIN" TUNABLE_LOCK_HELD=1 TUNABLE_STAGE="$STAGE" \
      bash "$ROOT/scripts/run_tunable_mica.sh" run >> "$OUT/nohup.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$OUT/job.pid"
    echo "[STARTED] PID=$!; physical GPU=$GPU_ID; stage=$STAGE"
    echo "[LOG] $OUT/nohup.log"
    ;;
  run) run_job ;;
  status)
    if [[ -s "$OUT/exit_code.txt" ]]; then
      echo "[FINISHED] exit_code=$(<"$OUT/exit_code.txt") (0 means success)"
    elif [[ -s "$OUT/job.pid" ]] && kill -0 "$(<"$OUT/job.pid")" 2>/dev/null; then
      echo "[RUNNING] PID=$(<"$OUT/job.pid")"
    else
      echo '[IDLE/STOPPED] No active job confirmed.'
    fi
    echo "[CHECKPOINTS] $(checkpoint_count)/20"
    [[ ! -s "$OUT/stage.txt" ]] || echo "[STAGE] $(<"$OUT/stage.txt")"
    [[ ! -f "$OUT/nohup.log" ]] || tail -n 25 "$OUT/nohup.log"
    ;;
  follow) [[ -f "$OUT/nohup.log" ]] || die 'No log yet; run start first.'; tail -f "$OUT/nohup.log" ;;
  summary) "$PYTHON_BIN" scripts/summarize_mica.py --run-dir "$OUT" ;;
  *) echo 'Usage: bash scripts/run_tunable_mica.sh {check|start|status|follow|merge|summary}' >&2; exit 2 ;;
esac
