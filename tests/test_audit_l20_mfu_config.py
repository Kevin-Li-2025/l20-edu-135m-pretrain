from scripts.audit_l20_mfu_config import audit_config


def test_audit_flags_slow_8k_without_liger_or_compile() -> None:
    payload = audit_config(
        {
            "model": {"block_size": 8192, "attn_implementation": "sdpa"},
            "trainer": {
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "compile": False,
                "liger_kernel": False,
                "gradient_checkpointing": True,
            },
        }
    )

    issues = {item["issue"] for item in payload["findings"]}
    assert payload["status"] == "review"
    assert "block_size_above_4k" in issues
    assert "liger_disabled" in issues


def test_audit_accepts_fast_2k_liger_compile_shape() -> None:
    payload = audit_config(
        {
            "model": {"block_size": 2048, "attn_implementation": "sdpa"},
            "trainer": {
                "micro_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "compile": True,
                "liger_kernel": True,
                "gradient_checkpointing": False,
            },
        }
    )

    assert payload["status"] == "pass"
    assert payload["tokens_per_step"] == 32768
