#!/usr/bin/env bash
# specs.sh — print the machine facts you need to plan an ML/RL project on a box.
#
#     ./specs.sh                 # human-readable report
#
# Reconnaissance, not setup: it changes nothing. Run it first on a fresh
# instance to answer "what am I working with?" — GPU model + VRAM, how many
# GPUs and how they're wired, CPU cores, RAM, free disk — and to get a few
# derived rules-of-thumb for model size, batch/parallelism, and RL env counts.
#
# Degrades gracefully: missing tools (no nvidia-smi on a CPU box, sysctl vs
# /proc on macOS) are skipped with a note rather than failing the whole run.
set -uo pipefail   # no -e: a missing optional tool must not abort the report

# --- tiny formatting helpers -------------------------------------------------
bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true)
rst=$(tput sgr0 2>/dev/null || true)
section() { printf '\n%s== %s ==%s\n' "$bold" "$1" "$rst"; }
kv()      { printf '  %-22s %s\n' "$1" "$2"; }
note()    { printf '  %s%s%s\n' "$dim" "$1" "$rst"; }
have()    { command -v "$1" >/dev/null 2>&1; }

OS="$(uname -s)"

# --- host / OS ---------------------------------------------------------------
section "Host"
kv "hostname" "$(hostname 2>/dev/null || echo '?')"
if [ -r /etc/os-release ]; then
  kv "os" "$(. /etc/os-release && echo "$PRETTY_NAME")"
elif [ "$OS" = "Darwin" ]; then
  kv "os" "macOS $(sw_vers -productVersion 2>/dev/null)"
fi
kv "kernel" "$(uname -sr)"
have uptime && kv "uptime" "$(uptime | sed 's/^ *//')"

# --- CPU ---------------------------------------------------------------------
section "CPU"
CORES_LOGICAL=""
if have lscpu; then
  model=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
  sockets=$(lscpu | sed -n 's/^Socket(s): *//p')
  cps=$(lscpu | sed -n 's/^Core(s) per socket: *//p')
  CORES_LOGICAL=$(lscpu | sed -n 's/^CPU(s): *//p')
  [ -n "$model" ]   && kv "model" "$model"
  [ -n "$sockets" ] && [ -n "$cps" ] && kv "physical cores" "$(( sockets * cps )) (${sockets} socket(s))"
  [ -n "$CORES_LOGICAL" ] && kv "logical cpus" "$CORES_LOGICAL"
elif [ "$OS" = "Darwin" ]; then
  kv "model" "$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  kv "physical cores" "$(sysctl -n hw.physicalcpu 2>/dev/null)"
  CORES_LOGICAL="$(sysctl -n hw.logicalcpu 2>/dev/null)"
  kv "logical cpus" "$CORES_LOGICAL"
fi
[ -z "$CORES_LOGICAL" ] && have nproc && CORES_LOGICAL="$(nproc)"

# --- memory ------------------------------------------------------------------
section "Memory"
RAM_GB=""
if have free; then
  read -r RAM_GB AVAIL_GB <<<"$(free -g | awk '/^Mem:/ {print $2, $7}')"
  kv "ram total" "${RAM_GB} GB"
  [ -n "${AVAIL_GB:-}" ] && kv "ram available" "${AVAIL_GB} GB"
  swap=$(free -g | awk '/^Swap:/ {print $2}')
  [ "${swap:-0}" -gt 0 ] && kv "swap" "${swap} GB"
elif [ "$OS" = "Darwin" ]; then
  RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
  kv "ram total" "${RAM_GB} GB"
fi

# --- GPU ---------------------------------------------------------------------
section "GPU"
GPU_COUNT=0
VRAM_TOTAL_GB=0
if have nvidia-smi; then
  # CUDA driver version + driver, from the header (most portable field).
  cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  [ -n "$drv" ]  && kv "driver" "$drv"
  [ -n "$cuda" ] && kv "cuda (driver)" "$cuda"

  # Per-GPU: index, name, total/free VRAM (MiB), compute capability.
  while IFS=',' read -r idx name memtot memfree cc; do
    [ -z "$idx" ] && continue
    idx=$(echo "$idx" | xargs); name=$(echo "$name" | xargs)
    memtot=$(echo "$memtot" | xargs); memfree=$(echo "$memfree" | xargs); cc=$(echo "$cc" | xargs)
    gb=$(awk "BEGIN{printf \"%.0f\", $memtot/1024}")
    freegb=$(awk "BEGIN{printf \"%.0f\", $memfree/1024}")
    kv "gpu $idx" "$name | ${gb} GB (${freegb} GB free) | sm_${cc//./}"
    GPU_COUNT=$((GPU_COUNT + 1))
    VRAM_TOTAL_GB=$(awk "BEGIN{print $VRAM_TOTAL_GB + $gb}")
  done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap \
             --format=csv,noheader,nounits 2>/dev/null)

  kv "gpu count" "$GPU_COUNT"
  kv "vram total" "${VRAM_TOTAL_GB} GB"

  # Interconnect topology only matters once there's >1 GPU (NVLink vs PCIe
  # decides whether tensor/pipeline parallel is cheap).
  if [ "$GPU_COUNT" -gt 1 ] && nvidia-smi topo -m >/dev/null 2>&1; then
    if nvidia-smi topo -m 2>/dev/null | grep -q 'NV[0-9]'; then
      kv "interconnect" "NVLink present (good for tensor/pipeline parallel)"
    else
      kv "interconnect" "PCIe only (prefer data parallel; keep model on 1 GPU)"
    fi
    note "full topology: nvidia-smi topo -m"
  fi
elif [ "$OS" = "Darwin" ]; then
  chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
  case "$chip" in
    *Apple*) kv "gpu" "Apple Silicon integrated GPU (Metal / MPS backend)";;
    *)       note "no nvidia-smi; check 'system_profiler SPDisplaysDataType'";;
  esac
else
  note "no nvidia-smi found — treating this as a CPU-only box"
fi

# --- disk --------------------------------------------------------------------
section "Disk"
if have df; then
  here="$(cd "$(dirname "$0")" && pwd)"
  read -r size avail mnt <<<"$(df -h "$here" | awk 'NR==2 {print $2, $4, $NF}')"
  kv "working disk" "${avail} free of ${size}  (mount: ${mnt})"
  # Surface a large data mount if one is configured (kept off the install disk).
  for m in /lambda/nfs/workspace /mnt/nfs /data /scratch; do
    if [ -d "$m" ] && mountpoint -q "$m" 2>/dev/null; then
      read -r s a <<<"$(df -h "$m" | awk 'NR==2 {print $2, $4}')"
      kv "data mount" "$m  — ${a} free of ${s}"
    fi
  done
fi

# --- ML/RL planning notes (derived from the numbers above) -------------------
section "ML/RL planning notes"
if [ "$GPU_COUNT" -gt 0 ] && awk "BEGIN{exit !($VRAM_TOTAL_GB>0)}"; then
  # Rough param budgets at bf16. Inference ≈ 2 B/param; full fine-tune with
  # Adam ≈ 18 B/param (weights+grads+2 optimizer states+activations headroom).
  infer_b=$(awk "BEGIN{printf \"%.1f\", $VRAM_TOTAL_GB*1e9/2/1e9}")
  train_b=$(awk "BEGIN{printf \"%.1f\", $VRAM_TOTAL_GB*1e9/18/1e9}")
  note "total VRAM ${VRAM_TOTAL_GB} GB across ${GPU_COUNT} GPU(s)"
  note "max model (bf16 inference, ~2 B/param):      ~${infer_b} B params"
  note "max model (bf16 full fine-tune w/ Adam, ~18 B/param): ~${train_b} B params"
  note "LoRA / QLoRA fits much larger models — frozen base in 4-bit cuts this ~4-8x"
  [ "$GPU_COUNT" -gt 1 ] && note "${GPU_COUNT} GPUs: DDP for data parallel; FSDP/tensor-parallel only if model > 1 GPU"
else
  note "CPU-only: expect small models / prototyping; runtimes 10-100x a single GPU"
fi
if [ -n "${CORES_LOGICAL:-}" ]; then
  note "RL: ~${CORES_LOGICAL} logical cpus → up to ~${CORES_LOGICAL} parallel envs / dataloader workers"
fi
if [ -n "${RAM_GB:-}" ] && [ "${RAM_GB:-0}" -gt 0 ]; then
  note "host RAM ${RAM_GB} GB caps CPU replay-buffer / dataset-in-memory size"
fi

printf '\n'

