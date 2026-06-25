import json
from pathlib import Path

from l20_pretrain.skill_corpus import (
    clean_skill_corpus,
    infer_skill,
    lexical_quality_score,
)


def test_infer_skill_detects_targeted_domains() -> None:
    assert infer_skill("The force of gravity affects the planet. Explain the answer.") == "arc_science"
    assert infer_skill("Which option should she choose? The answer depends on who her friend helped.") == "winogrande_coreference"
    assert infer_skill("def solve(x):\n    return x + 1") == "python_edu"


def test_lexical_quality_scores_reasoning_text() -> None:
    text = (
        "A student observes that the water temperature rises because heat energy moves "
        "from the warmer surface into the cooler liquid. Therefore the answer is based "
        "on energy transfer and evidence from the experiment. "
    ) * 3
    assert lexical_quality_score(text, "arc_science") > 0.34


def test_clean_skill_corpus_filters_duplicates_and_contamination(tmp_path: Path) -> None:
    contamination = tmp_path / "benchmarks.jsonl"
    contamination.write_text(
        json.dumps(
            {
                "benchmark": "arc_challenge",
                "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    good = (
        "A teacher explains that gravity pulls objects toward the planet. The student "
        "uses evidence from a simple experiment, because the ball falls every time it "
        "is released. Therefore the answer is that the force changes the motion. "
        "Next, the class compares a feather, a stone, and a paper cup to see how air "
        "resistance changes the observation. The explanation separates the force of "
        "gravity from the effect of the surrounding air, and it asks which conclusion "
        "is best supported by the evidence. A careful answer notes that repeated trials "
        "make the pattern more reliable, while a single surprising trial is not enough "
        "to reject the original hypothesis."
    )
    duplicate = good.replace("released", "dropped")
    contaminated = (
        "A passage explains a classroom puzzle with alpha beta gamma delta epsilon "
        "zeta eta theta iota kappa lambda mu nu xi as the exact sequence students "
        "must inspect. The teacher then asks for evidence, reasoning, and a final "
        "answer about why the sequence should not be copied into the training set. "
        "The surrounding words make the document long enough to pass ordinary quality "
        "checks while still preserving the benchmark overlap. Additional discussion "
        "describes how a careful data pipeline must reject passages that copy an "
        "evaluation sequence, even when the rest of the paragraph looks educational, "
        "well written, diverse, and useful for a small science reasoning model."
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"text": good, "skill": "arc_science"}),
                json.dumps({"text": duplicate, "skill": "arc_science"}),
                json.dumps({"text": contaminated, "skill": "arc_science"}),
                json.dumps({"text": "too short"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = clean_skill_corpus(
        input_paths=[input_path],
        out_jsonl=tmp_path / "clean.jsonl",
        guard_index=tmp_path / "guard.sqlite",
        contamination_path=contamination,
        min_chars=40,
        min_quality_score=0.2,
    )

    clean_rows = [
        json.loads(line)
        for line in (tmp_path / "clean.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rejected_rows = [
        json.loads(line)
        for line in (tmp_path / "clean.rejected.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["kept_records"] == 1
    assert clean_rows[0]["skill"] == "arc_science"
    assert any(row["reason"] == "near_duplicate" for row in rejected_rows)
    assert any(row["reason"].startswith("benchmark_13gram_lcs") for row in rejected_rows)
    assert manifest["counters"]["kept"] == 1
