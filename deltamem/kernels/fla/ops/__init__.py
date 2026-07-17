# Lazy imports: these submodules depend on the installed `fla` package.
# Wrapping in try/except so that the self-contained kernels (e.g. chunk_osgm_phase)
# remain importable even when the full FLA stack is unavailable.
try:
    from .chunk_osgm import chunk_delta_rule_osgm
except ImportError:
    chunk_delta_rule_osgm = None

try:
    from .chunk_osgm_dd_decay import chunk_delta_rule_osgm_dd_decay
except ImportError:
    chunk_delta_rule_osgm_dd_decay = None

try:
    from .fused_recurrent_osgm import fused_recurrent_delta_rule_osgm
except ImportError:
    fused_recurrent_delta_rule_osgm = None

try:
    from .fused_recurrent_osgm_dd_decay import fused_recurrent_delta_rule_osgm_dd_decay
except ImportError:
    fused_recurrent_delta_rule_osgm_dd_decay = None

__all__ = [
    "chunk_delta_rule_osgm",
    "chunk_delta_rule_osgm_dd_decay",
    "fused_recurrent_delta_rule_osgm",
    "fused_recurrent_delta_rule_osgm_dd_decay",
]
