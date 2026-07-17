# FLA Kernel Overlay

This directory contains the OSDN Triton kernels in the same path layout used by
`flash-linear-attention`.

The files are intended to be copied into an existing FLA checkout or installed
FLA source tree:

```bash
rsync -av fla/ops/os_delta_rule/ /path/to/flash-linear-attention/fla/ops/os_delta_rule/
rsync -av fla/ops/os_gated_delta_rule/ /path/to/flash-linear-attention/fla/ops/os_gated_delta_rule/
rsync -av fla/ops/os_kda/ /path/to/flash-linear-attention/fla/ops/os_kda/
```

They keep FLA-style imports such as `from fla.ops.common...`; copy them into an
FLA environment with the usual FLA dependencies available.

Included paper kernel groups:

- `fla/ops/os_delta_rule/`: OSDN and OSDN-APF DeltaNet chunk paths, fused
  recurrent paths, phase-1 sweeps, phase-2/WY helpers, and shared scaled-dot
  helpers used by OSGDN and OSKDA.
- `fla/ops/os_gated_delta_rule/`: OSGDN chunk and fused recurrent kernels,34
  including the post-gate-regret path.
- `fla/ops/os_kda/`: OSKDA chunk, fused recurrent, gate, intra-chunk, and
  backward kernels.

This overlay contains the paper kernels and their direct dependencies. Internal
`osgm` file/function names are retained where FLA imports depend on them.
