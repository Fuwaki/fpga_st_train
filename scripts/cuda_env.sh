# Source this from the training scripts. Do not execute it directly.
#
# The training GPU is a Tesla V100 (sm_70). CUDA 13 nvrtc rejects that
# architecture; its lowest target is sm_75. CUDA 12.8 still emits sm_70 PTX,
# and driver 580 loads it. The distro toolkit at /opt/cuda is 13.4, whose PTX
# this driver also refuses, so the compiler has to be the 12.8 redistributable.

# Always this tree. A login shell often exports CUDA_PATH=/opt/cuda, and that
# 13.4 nvrtc cannot compile for the V100.
export CUDA_PATH="/data/competition/2026fpga/cuda-12.8/prefix"

if [ -n "${CUDA_NVRTC_LIB:-}" ] && [ -e "${CUDA_NVRTC_LIB}/libnvrtc.so" ]; then
  _nvrtc="${CUDA_NVRTC_LIB}"
else
  _nvrtc=""
  for _cand in \
    "${CUDA_PATH}/lib" \
    "/data/competition/2026fpga/cuda-12.8/prefix/lib"
  do
    if [ -e "${_cand}/libnvrtc.so" ]; then
      _nvrtc="${_cand}"
      break
    fi
  done
fi

if [ -n "${_nvrtc}" ]; then
  # Drop a previous copy so this directory is first. /opt/cuda/lib must not
  # stay ahead of it: that tree is libnvrtc.so.13, which rejects sm_70.
  _rest="$(printf '%s' ":${LD_LIBRARY_PATH:-}:" | sed -e "s|:${_nvrtc}:|:|g" -e 's/^://' -e 's/:$//')"
  export LD_LIBRARY_PATH="${_nvrtc}${_rest:+:${_rest}}"
fi
unset _nvrtc _cand _rest

# The CUDA build NEEDs libwebgpu_dawn.so for the ONNX teacher. ort symlinks it
# next to the binary and does not set RPATH, so the loader only finds it here.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_dawn="${_root}/target/release"
case ":${LD_LIBRARY_PATH:-}:" in
  *":${_dawn}:"*) ;;
  *) export LD_LIBRARY_PATH="${_dawn}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
unset _root _dawn

export CUDARC_CUDA_VERSION="12080"
