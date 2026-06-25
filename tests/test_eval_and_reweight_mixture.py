from scripts.eval_and_reweight_mixture import DEFAULT_TARGETS, BASE_WEIGHTS, reweight


def test_reweight_boosts_weak_task_skill() -> None:
    scores = dict(DEFAULT_TARGETS)
    scores["hellaswag"] = DEFAULT_TARGETS["hellaswag"] - 0.10
    payload = reweight(
        scores,
        targets=DEFAULT_TARGETS,
        base_weights=BASE_WEIGHTS,
        max_boost=0.12,
        floor=0.02,
        ceiling=0.32,
    )
    weights = payload["weights"]
    assert weights["hellaswag_continuation"] > BASE_WEIGHTS["hellaswag_continuation"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
