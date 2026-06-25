import json
from pathlib import Path

from l20_pretrain.skill_corpus import (
    BenchmarkSimilarityIndex,
    clean_skill_corpus,
    extract_answer_label,
    infer_skill,
    lexical_quality_score,
    template_signature,
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


def test_template_signature_masks_numbers_and_quotes() -> None:
    left = template_signature('Question 123: "red ball" Which option is correct?', "piqa_physical")
    right = template_signature('Question 999: "blue cup" Which option is correct?', "piqa_physical")
    assert left == right


def test_extract_answer_label_from_metadata_or_text() -> None:
    assert extract_answer_label({"answer": "b"}, "ignored") == "B"
    assert extract_answer_label({}, "Correct option: C because it fits.") == "C"


def test_benchmark_similarity_index_flags_high_overlap(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    path.write_text(
        json.dumps({"benchmark": "piqa", "text": "use a cotton swab to apply eyeshadow without a brush"})
        + "\n",
        encoding="utf-8",
    )
    index = BenchmarkSimilarityIndex(path, threshold=0.55)
    match = index.match("A person can use a cotton swab to apply eyeshadow carefully.")
    assert match is not None
    assert match[0] == "piqa"
    assert match[1] >= 0.55


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


def test_clean_skill_corpus_caps_repeated_templates(tmp_path: Path) -> None:
    rows = []
    for idx, label in enumerate(["A", "B"]):
        rows.append(
            {
                "text": (
                    f"Question {idx}: A person needs to push an object across a surface. "
                    f"Option A uses steady force. Option B uses a fragile tool. Correct option: {label}. "
                    "The answer is explained with friction, weight, surface contact, and physical reasoning "
                    "so the training example is long enough and useful for a small model. The explanation "
                    "mentions pressure, balance, grip, motion, stability, resistance, safety, material, "
                    "surface, direction, and outcome in plain language for physical commonsense training. "
                    "It also contrasts smooth wood, rough cloth, metal handles, plastic wheels, careful "
                    "posture, slow movement, and the reason a stable action is safer than a careless one."
                ),
                "skill": "piqa_physical",
            }
        )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = clean_skill_corpus(
        input_paths=[input_path],
        out_jsonl=tmp_path / "clean.jsonl",
        guard_index=tmp_path / "guard.sqlite",
        min_chars=40,
        min_quality_score=0.1,
        max_template_repeats=1,
        max_answer_label_count=None,
    )

    assert manifest["kept_records"] == 1
    assert manifest["counters"]["template_cap"] == 1


def test_clean_skill_corpus_caps_answer_labels(tmp_path: Path) -> None:
    rows = []
    texts = [
        "A cook wants to cool soup safely. Option A moves it to a shallow bowl. Option B seals it in a hot jar.",
        "A worker needs to lift a box. Option A bends knees and holds it close. Option B pulls from a weak corner.",
        "A student needs to dry a wet floor. Option A uses an absorbent towel. Option B spreads more water.",
    ]
    for text in texts:
        rows.append(
            {
                "text": (
                    f"{text} Correct option: A. The answer is explained with physical reasoning, "
                    "tool use, surface contact, safety, and a short causal explanation that is "
                    "long enough for the corpus quality filter. The note mentions pressure, balance, "
                    "motion, resistance, material, shape, temperature, direction, stability, and outcome "
                    "using simple language that a small physical commonsense model can learn from. "
                    "It contrasts smooth wood, rough cloth, metal handles, plastic wheels, careful posture, "
                    "slow movement, and the reason a stable action is safer than a careless one."
                ),
                "skill": "piqa_physical",
            }
        )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = clean_skill_corpus(
        input_paths=[input_path],
        out_jsonl=tmp_path / "clean.jsonl",
        guard_index=tmp_path / "guard.sqlite",
        min_chars=40,
        min_quality_score=0.1,
        max_template_repeats=10,
        max_answer_label_count=1,
    )

    assert manifest["kept_records"] == 1
    assert manifest["answer_label_counts"] == {"A": 1}
    assert manifest["counters"]["answer_label_cap_A"] == 2
