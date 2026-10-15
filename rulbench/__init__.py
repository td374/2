"""XJTU-SY rolling-bearing RUL benchmark.

A reproducible re-implementation of the SE-ResNet + BiGRU + MK-MMD pipeline for
cross-operating-condition remaining useful life (RUL) prediction.

Package layout
--------------
rulbench.config        configuration loading / merging
rulbench.utils         seeding, logging, path helpers
rulbench.data          raw loading -> denoising -> features -> labels -> scaling -> windows
rulbench.models        proposed model, baselines and ablation variants
rulbench.losses        weighted MSE and (efficient) multi-kernel MMD
rulbench.train         unified training loop and evaluation protocols
rulbench.evaluation    metrics, uncertainty quantification, post-processing
rulbench.pipelines     end-to-end runnable pipelines
"""

import os as _os

# cuBLAS picks its GEMM workspace / rank ordering when it creates its handle on
# the FIRST cuBLAS call, so this variable is only honoured if it is set before
# torch is imported anywhere. ``rulbench`` is the first import of every entry
# point, which makes this the earliest safe place. Previously the variable was
# set inside ``utils.seed_everything`` -- long after ``resolve_device`` had
# already initialised CUDA -- so it did nothing and identical configurations
# disagreed run to run (scripts/17_reproducibility_probe.py).
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.2.0"
