"""setup.sh's Hugging Face downloads (services/hf_get.py): the command line it is given, a named
file the repository does not have, and the script run as setup.sh runs it."""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "hf_get.py"
_spec = importlib.util.spec_from_file_location("hf_get", SCRIPT)
hf_get = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf_get)


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


def test_run_as_setup_sh_runs_it_the_standard_library_is_not_shadowed(tmp_path):
    """setup.sh runs the file with a venv's python and no PYTHONPATH. A script's directory comes
    first on ``sys.path``: beside ``pi_embodied_services/utils/logging.py`` the standard
    ``logging`` that huggingface_hub's imports need was that file, and the download died on
    ``No module named 'pi_embodied_services'``. The stand-in hub imports what the real one does."""
    hub = tmp_path / "site" / "huggingface_hub"
    hub.mkdir(parents=True)
    (hub / "__init__.py").write_text(
        "import logging, json, os, typing, collections, types\n"
        "assert hasattr(logging, 'getLogger') and 'pi_embodied_services' not in (logging.__file__ or '')\n"
        "def snapshot_download(**kw):\n"
        "    os.makedirs(kw['local_dir'], exist_ok=True)\n"
        "    open(os.path.join(kw['local_dir'], 'config.json'), 'w').write(json.dumps(kw))\n"
        "    return kw['local_dir']\n"
    )
    out = tmp_path / "w"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(hub.parent)
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "RLinf/x", "abc", str(out), "config.json"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(out) and (out / "config.json").is_file()
    # No module of the script's directory shares a standard library name the hub's imports reach.
    beside = {p.stem for p in SCRIPT.parent.glob("*.py")}
    assert not beside & set(sys.stdlib_module_names), beside & set(
        sys.stdlib_module_names
    )
