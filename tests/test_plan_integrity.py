from src.app.engine.targets import build_target_profile


def test_target_profile_probabilities_and_snap_trace_present() -> None:
    target_meta = [
        {"label": "TP1", "prob_touch": 0.45},
        {"label": "TP2", "probability": 0.25},
    ]
    debug_payload = {
        "em_limit": 3.5,
        "meta": [
            {
                "label": "TP1",
                "candidate_nodes": [
                    {"label": "ORL", "price": 99.2, "picked": True, "rr_multiple": 1.3},
                    {"label": "SESSION_LOW", "price": 98.0, "picked": False, "rr_multiple": 2.1},
                ],
            }
        ],
    }
    tp_reasons = [
        {
            "label": "TP1",
            "reason": "Nearest structure clearing RR",
            "snap_tag": "ORL",
            "candidate_nodes": debug_payload["meta"][0]["candidate_nodes"],
        }
    ]

    profile = build_target_profile(
        entry=100.0,
        stop=99.2,
        targets=[101.2, 102.8],
        target_meta=target_meta,
        debug=debug_payload,
        runner={},
        warnings=["note"],
        atr_used=1.4,
        expected_move=3.5,
        style="intraday",
        bias="short",
        key_levels_used={"session": []},
        tp_reasons=tp_reasons,
        entry_candidates=[{"level": 100.0}],
        runner_policy={"type": "trail"},
    )

    assert profile.probabilities == {"tp1": 0.45, "tp2": 0.25}
    assert profile.snap_trace == debug_payload["meta"]
    assert profile.tp_reasons == tp_reasons
    assert profile.key_levels_used == {"session": []}
