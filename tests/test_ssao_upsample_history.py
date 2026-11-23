import pytest
import graphics.ssao_upsample_history as mod

CASES = [
    ("depth_upsample", (1,)),
    ("history_valid", (1,)),
    ("history_reset", (1,)),
    ("pass_order", (1,)),
    ("sample_sequence", (1,)),
    ("history_clip", (1,)),
    ("dependency_fence", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("edge_mask", (1,)),
    ("metrics", (1,)),
    ("teleport", (1,)),
    ("denoise", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
