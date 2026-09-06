from types import SimpleNamespace

from l20_pretrain import provenance


def test_resolve_hf_revision_returns_server_commit(monkeypatch) -> None:
    class FakeApi:
        def repo_info(self, **kwargs):
            assert kwargs == {
                "repo_id": "org/data",
                "repo_type": "dataset",
                "revision": "release",
            }
            return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    assert provenance.resolve_hf_revision(
        "org/data",
        repo_type="dataset",
        revision="release",
    ) == "a" * 40
