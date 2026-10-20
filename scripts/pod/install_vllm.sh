#!/usr/bin/env bash
# vLLM venv for generation (copied from timf34/grader-shifts-persona-space@0ad65c4
# scripts/pod/install_easysteer.sh, minus the EasySteer overlay: this project only generates).
# Idempotent: run it on every pod start (scripts/pod/_lib.sh es_install does), since apt
# packages and the CUDA compat libraries live on the container disk, which a stop wipes.
#
# vllm 0.26.0 (the version grader-shifts and Betley et al. ran Qwen3.6-27B on) with
# transformers pinned exactly as in requirements/pod.txt (chat templates are rendered here).
#
# Wheels follow the HOST driver (nvidia-smi "CUDA Version"):
#   >= 13 : PyPI vllm==0.26.0 (CUDA 13 build, torch 2.11.0)       -> $ES_VENV_ROOT/vllm_venv_cu130
#   12.x  : the official vllm-0.26.0+cu129 release wheel with torch 2.11.0+cu129 and a CPU
#           torchcodec, plus NVIDIA's cuda-compat-12-9 libraries on drivers < 575
#                                                                  -> $ES_VENV_ROOT/vllm_venv_cu129
#           (the PyPI wheel needs libcudart.so.13 even with a cu128 torch index; recipe from
#           timf34/tess-antra-project reports/generation_runtime_repair_2026-10-08.md)
# Writes /workspace/.es_env (ES_VENV, ES_LD_LIBRARY_PATH, ES_CUDA_LINE); the compat library
# path is applied to vLLM processes only (scripts/pod/_lib.sh es_py), never to the HF venv.
#
#   bash scripts/pod/install_vllm.sh        # prints ES_INSTALL_DONE <venv>
set -euo pipefail
VLLM_PIN="${VLLM_PIN:-0.26.0}"
# same exact pin as requirements/pod.txt (chat templates are rendered in this venv)
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-5.14.1}"
COMPAT_VER="12-9_575.57.08-0ubuntu1"

DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
case "${DRV_CUDA%%.*}" in
  12) LINE=cu129 ;;
  1[3-9]|[2-9][0-9]) LINE=cu130 ;;
  *) echo "!! host driver ${DRV:-?} (CUDA ${DRV_CUDA:-?}) is too old: need CUDA >= 12.x"; exit 2 ;;
esac
ES_ROOT="${ES_VENV_ROOT:-/workspace}"
ES_VENV="${ES_VENV:-${ES_ROOT%/}/vllm_venv_$LINE}"
echo "driver ${DRV:-?} (CUDA ${DRV_CUDA:-?}) -> $LINE wheels, venv $ES_VENV"

# torchcodec (a vLLM dependency) needs the FFmpeg shared libraries just to import
if ! command -v ffmpeg >/dev/null 2>&1; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null) \
    || echo "(apt-get install ffmpeg failed; the import self-test below will tell if it matters)"
fi

[ -x "$ES_VENV/bin/python" ] || python3 -m venv "$ES_VENV"
PIP=("$ES_VENV/bin/python" -m pip)
"${PIP[@]}" install -q -U pip
COMMON=("transformers==$TRANSFORMERS_PIN" ninja pyyaml python-dotenv hf_transfer pandas pyarrow)
if [ "$LINE" = cu129 ]; then
  PYTAG=$("$ES_VENV/bin/python" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
  "${PIP[@]}" install -q \
    --extra-index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://download.pytorch.org/whl/cpu \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_PIN}/vllm-${VLLM_PIN}%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    "torch==2.11.0+cu129" "torchaudio==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchcodec==0.14.0+cpu" \
    "${COMMON[@]}"
  echo "python $PYTAG, $("$ES_VENV/bin/python" -m pip show torchcodec 2>/dev/null | grep -i '^version')"
else
  "${PIP[@]}" install -q "vllm==$VLLM_PIN" "${COMMON[@]}"
fi

ES_LD=""
if [ "$LINE" = cu129 ] && [ "${DRV%%.*}" -lt 575 ]; then
  # forward-compatibility libraries (datacenter GPUs: A100 / H100 / H200), generation only
  if [ ! -f /usr/local/cuda-12.9/compat/libcuda.so.1 ]; then
    . /etc/os-release
    DISTRO="ubuntu${VERSION_ID//./}"
    curl -fsSL --retry 3 \
      "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-compat-${COMPAT_VER}_amd64.deb" \
      -o /tmp/cuda-compat.deb
    dpkg -i /tmp/cuda-compat.deb >/dev/null
  fi
  ES_LD=/usr/local/cuda-12.9/compat
  echo "driver ${DRV} < 575: using CUDA 12.9 forward-compat libraries from $ES_LD (vLLM processes only)"
fi

# self-test with exactly the environment generation will use: venv bin on PATH (vLLM
# launches ninja by name), compat libraries, the full LLM entrypoint import, a real GPU op
export PATH="$ES_VENV/bin:$PATH"
[ -n "$ES_LD" ] && export LD_LIBRARY_PATH="$ES_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ninja --version >/dev/null
EXPECT_CUDA=$([ "$LINE" = cu129 ] && echo 12.9 || echo 13)
EXPECT_CUDA="$EXPECT_CUDA" TRANSFORMERS_PIN="$TRANSFORMERS_PIN" "$ES_VENV/bin/python" - <<'PY'
import os

import torch
import transformers
import vllm
from vllm import LLM, SamplingParams  # noqa: F401  (exercises media/dependency imports before any weights)

assert transformers.__version__ == os.environ["TRANSFORMERS_PIN"], transformers.__version__
assert str(torch.version.cuda).startswith(os.environ["EXPECT_CUDA"]), torch.version.cuda
x = torch.ones(16, device="cuda")
assert (x @ x).item() == 16
torch.cuda.synchronize()
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "transformers",
      transformers.__version__, "| GPU op ok on", torch.cuda.get_device_name(0))
PY
printf 'ES_VENV=%s\nES_LD_LIBRARY_PATH=%s\nES_CUDA_LINE=%s\n' "$ES_VENV" "$ES_LD" "$LINE" > /workspace/.es_env
echo "$ES_VENV" > /workspace/.es_venv_path
echo ES_INSTALL_DONE "$ES_VENV"
