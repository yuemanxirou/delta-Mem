# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Lazy imports: these submodules depend on the installed `fla` package.
try:
    from .chunk import chunk_kda
except ImportError:
    chunk_kda = None

try:
    from .fused_recurrent import fused_recurrent_kda
except ImportError:
    fused_recurrent_kda = None

__all__ = [
    "chunk_kda",
    "fused_recurrent_kda",
]
