# -*- coding: utf-8 -*-

# Lazy imports: these submodules depend on the installed `fla` package.
try:
    from fla.ops.os_gated_delta_rule.chunk import chunk_os_gated_delta_rule
except ImportError:
    chunk_os_gated_delta_rule = None

try:
    from fla.ops.os_gated_delta_rule.fused_recurrent import fused_recurrent_os_gated_delta_rule
except ImportError:
    fused_recurrent_os_gated_delta_rule = None

__all__ = [
    'chunk_os_gated_delta_rule',
    'fused_recurrent_os_gated_delta_rule',
]
