# Praxis Phase-0 environment: MuJoCo Playground / MJX + Brax PPO on JAX-CUDA12.
# JAX has no native-Windows CUDA wheel, so the user (Windows 10 + RTX 4090s) runs
# everything inside this container via Docker Desktop's WSL2 GPU passthrough.
#
#   docker build -t praxis:gpu .
#   docker run --rm --gpus all -v "C:/Users/Asav/source/repos/Praxis":/workspace praxis:gpu \
#       python -c "import jax; print(jax.default_backend())"   # -> gpu
#
# CUDA itself comes from the jax[cuda12] pip wheels; this base only provides the
# driver-capability env vars + EGL graphics libs for headless rendering.
FROM nvidia/cuda:12.6.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    JAX_DEFAULT_MATMUL_PRECISION=highest

# Python 3.11 (deadsnakes) + headless rendering stack (EGL primary, OSMesa fallback)
# + ffmpeg for mediapy video encoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        ffmpeg \
        libgl1-mesa-glx libgl1-mesa-dri libglew2.2 libglfw3 \
        libegl1 libgles2 libglvnd0 libopengl0 \
        libosmesa6 libosmesa6-dev \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Isolated virtualenv so pip never collides with Debian's distutils-installed
# system packages. deadsnakes' python3.11 inherits Debian's site.py patch that
# exposes /usr/lib/python3/dist-packages (e.g. blinker 1.4) on sys.path; flask
# (a brax dep) tries to upgrade that distutils-installed blinker and pip aborts
# with "uninstall-distutils-installed-package". A venv excludes dist-packages.
RUN python3.11 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Python deps. jax is PINNED to 0.9.2 (with the CUDA12 extra) in the SAME resolve so
# pip backtracks flax/optax/orbax to compatible versions. Why 0.9.2 (not latest 0.10.1):
# brax 0.14.2 (the newest brax) calls jax.device_put_replicated, which jax REMOVED in
# 0.10.0 -> brax crashes on jax>=0.10. 0.9.2 is the newest jax that still has that API
# AND imports the latest mujoco-mjx 3.9.0 / mujoco_playground. Keeps the whole modern
# MuJoCo/Brax stack; only jax is held one minor behind.
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        "jax[cuda12]==0.9.2" \
        playground \
        brax \
        mujoco \
        mujoco-mjx \
        flax optax orbax-checkpoint \
        ml_collections \
        mediapy "imageio[ffmpeg]" \
        matplotlib tensorboardX \
        pytest

WORKDIR /workspace
ENV PYTHONPATH=/workspace
CMD ["bash"]
