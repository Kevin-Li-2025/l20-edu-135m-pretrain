from l20_pretrain.rlvr_rewards import (
    extract_gsm8k_gold,
    extract_numeric_answer,
    gsm8k_correctness_reward,
    gsm8k_reward_func,
    repetition_penalty,
)


def test_extract_gsm8k_gold_prefers_hash_answer() -> None:
    assert extract_gsm8k_gold("We compute 5 + 7. #### 12") == "12"
    assert extract_gsm8k_gold("Total is 1,234.") == "1234"


def test_extract_numeric_answer_uses_boxed_or_tail() -> None:
    assert extract_numeric_answer("some work \\boxed{42}") == "42"
    assert extract_numeric_answer("3 + 4 = 7, so the answer is 7.") == "7"
    assert extract_numeric_answer("No numbers here") is None


def test_gsm8k_correctness_reward_matches_normalized_number() -> None:
    assert gsm8k_correctness_reward("The final answer is 1,200.", "#### 1200") == 1.0
    assert gsm8k_correctness_reward("The final answer is 1199.", "#### 1200") == 0.0


def test_gsm8k_reward_func_accepts_chat_completions() -> None:
    rewards = gsm8k_reward_func(
        [[{"role": "assistant", "content": "Add the two quantities carefully. The answer is 8."}]],
        ["#### 8"],
    )
    assert rewards == [1.3]


def test_repetition_penalty_catches_degenerate_copying() -> None:
    repeated = (
        "The red chickens produce eggs every day. "
        "The red chickens produce eggs every day. "
        "The red chickens produce eggs every day. "
        "The red chickens produce eggs every day. "
        "The answer is 4."
    )
    concise = "Let r be the red chickens. Then 3r + 5(r + 2) = 42, so 8r = 32. The answer is 4."
    assert repetition_penalty(repeated) > repetition_penalty(concise)
