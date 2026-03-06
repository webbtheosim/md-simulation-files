"""
Patch shap's XGBTreeModelLoader to handle bracketed base_score strings
from xgboost >= 2.1 (e.g. '[8.022989E0]' instead of '8.022989E0').

Import this module before calling shap.TreeExplainer with an XGBoost model.
"""
import shap.explainers._tree as _tree

_orig_float = float

def _safe_base_score_float(val):
    if isinstance(val, str) and val.startswith("["):
        val = val.strip("[]")
    return _orig_float(val)

# Patch the two lines in XGBTreeModelLoader.__init__ that call float(base_score)
_orig_xgb_init = _tree.XGBTreeModelLoader.__init__

def _patched_xgb_init(self, xgb_model):
    import builtins
    _saved = builtins.float
    builtins.float = _safe_base_score_float
    try:
        _orig_xgb_init(self, xgb_model)
    finally:
        builtins.float = _saved

_tree.XGBTreeModelLoader.__init__ = _patched_xgb_init
