#!/usr/bin/env bash
set -euo pipefail
root=/workspace/nextlat-spec-scaling
cd "$root"
commit=baf09f39544d3cd812c77ed78132eedefaee5c5d
base=lmsysorg/sglang@sha256:bde16a8447b19e89056b9eea06c72be6c02801dc89d528c9ea90c53368fd74bf
mkdir -p build/compiler-cache outputs
python3 -m venv control-venv
control-venv/bin/pip install -q wandb==0.29.0
if [[ ! -d source/.git ]]; then
  git clone source-sidecar-grid.bundle source
  git -C source checkout --detach b1cdebe16959754be3c1c3b1f4f747e0982caaa0
fi
if [[ ! -d build/sglang/.git ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git build/sglang
  git -C build/sglang fetch --depth 1 origin "$commit"
  git -C build/sglang checkout --detach FETCH_HEAD
fi
[[ $(git -C build/sglang rev-parse HEAD) == "$commit" ]]
sudo -n docker pull "$base"
sudo -n docker build --build-arg BASE_REF="$base" -t nextlat-sglang:scaling-20260909 -f Dockerfile build
sudo -n docker image inspect nextlat-sglang:scaling-20260909 --format '{{.Id}}'
echo BOOTSTRAP_READY
