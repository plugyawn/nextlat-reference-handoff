"""Build-time, no-GPU execution check for the derived SM90 image."""

from __future__ import annotations

import importlib.metadata
import json

import flashinfer
import safetensors
import sglang
import torch

from .admission import admit_sglang_source
from .constants import AUDITED_SGLANG_COMMIT


def main() -> None:
    source = admit_sglang_source()
    if len(source.audited_files) != 26:
        raise RuntimeError("installed SGLang source receipt did not admit 26 files")
    arches = torch.cuda.get_arch_list()
    if "sm_90" not in arches:
        raise RuntimeError(f"PyTorch wheel has no SM90 support: {arches}")
    entries = [
        entry
        for entry in importlib.metadata.entry_points(group="sglang.srt.plugins")
        if entry.name == "nextlat"
    ]
    loaded_entry = entries[0].load() if len(entries) == 1 else None
    if (
        len(entries) != 1
        or entries[0].value != "sglang_nextlat.plugin:register"
        or loaded_entry.__module__ != "sglang_nextlat.plugin"
        or loaded_entry.__name__ != "register"
    ):
        raise RuntimeError("image does not expose exactly one NextLat plugin")
    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("sglang", "sglang-nextlat", "safetensors", "torch")
    }
    print(
        json.dumps(
            {
                "sglang_commit": AUDITED_SGLANG_COMMIT,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "flashinfer": getattr(flashinfer, "__version__", "unknown"),
                "safetensors": safetensors.__version__,
                "sglang": getattr(sglang, "__version__", package_versions["sglang"]),
                "packages": package_versions,
                "torch_arches": arches,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
