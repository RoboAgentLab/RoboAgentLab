"""setup.sh's Hugging Face downloads (utils/hf_get.py): the command line it is given, and a named
file the repository does not have."""

import sys
import types

import pytest

from pi_embodied_services.utils import hf_get


def test_the_command_line_is_snapshot_downloads_arguments():
    """A whole snapshot into a directory; named files as patterns; ``-`` is the cache; a dataset
    repository's type may follow the files, as setup.sh writes it."""
    rev = "3c879f39826c281e95690f02c7821c4de09afae7"
    assert hf_get.kwargs(hf_get.parse(["RLinf/x", rev, "/w/x"])) == {
        "repo_id": "RLinf/x",
        "revision": rev,
        "repo_type": "model",
        "local_dir": "/w/x",
    }
    assert hf_get.kwargs(
        hf_get.parse(["facebook/sam3", rev, "/w/sam3", "sam3.pt"])
    ) == {
        "repo_id": "facebook/sam3",
        "revision": rev,
        "repo_type": "model",
        "local_dir": "/w/sam3",
        "allow_patterns": ["sam3.pt"],
    }
    cached = hf_get.kwargs(hf_get.parse(["RLWRLD/RLDX-1-VLM", rev, "-"]))
    assert "local_dir" not in cached and "allow_patterns" not in cached
    dataset = hf_get.parse(
        ["Sylvest/LIBERO-plus", rev, "/w/z", "assets.zip", "--repo-type", "dataset"]
    )
    assert hf_get.kwargs(dataset)["repo_type"] == "dataset"
    assert hf_get.kwargs(dataset)["allow_patterns"] == ["assets.zip"]


def test_a_named_file_the_repository_lacks_fails_the_download(
    monkeypatch, tmp_path, capsys
):
    """Names are patterns to ``snapshot_download``: one that matches nothing downloads nothing and
    raises nothing, so the files are checked afterwards (the ``hf`` CLI refused such a name)."""
    calls = []

    def snapshot_download(**kw):
        calls.append(kw)
        (tmp_path / "sam3.pt").write_bytes(b"x")
        return str(tmp_path)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    hf_get.main(["facebook/sam3", "abc", str(tmp_path), "sam3.pt"])
    assert capsys.readouterr().out.strip() == str(tmp_path)
    assert calls[0]["allow_patterns"] == ["sam3.pt"]
    with pytest.raises(SystemExit, match="facebook/sam3@abc has no sam4.pt"):
        hf_get.main(["facebook/sam3", "abc", str(tmp_path), "sam3.pt", "sam4.pt"])
