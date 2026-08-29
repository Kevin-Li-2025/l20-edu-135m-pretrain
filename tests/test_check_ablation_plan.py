from pathlib import Path

from scripts.check_ablation_plan import validate_plan


def test_current_ablation_plan_is_valid() -> None:
    assert validate_plan(Path("docs/project_report/ablation_plan.json")) == []


def test_ablation_plan_reports_missing_experiment_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad_plan.json"
    path.write_text(
        """
        {
          "schema_version": 1,
          "project": "test",
          "purpose": "test",
          "default_eval_suite": ["arc_challenge"],
          "experiments": [{"id": "missing_fields"}]
        }
        """,
        encoding="utf-8",
    )

    errors = validate_plan(path)

    assert errors
    assert "missing fields" in errors[0]
