"""Download a Hugging Face snapshot, or some files of it, for ``services/setup.sh`` (``hf_get``).

    hf_get.py REPO REVISION DIR [FILE...] [--repo-type TYPE]

``DIR`` is the directory the files land in (``-``: the Hugging Face cache). huggingface_hub reads
``HF_ENDPOINT`` and ``HF_TOKEN`` itself.

The Python API rather than the ``hf`` CLI: the CLI is built on typer, and in the BEHAVIOR-1K venv
(Isaac Sim pins click 8.1.7, huggingface_hub needs typer >= 0.20) every ``hf`` command exits 1 on
an uncaught ``click.exceptions.Exit: 0``, after a complete download too.

Run as a file, so the venv needs huggingface_hub and nothing of the services package. It lives
beside setup.sh, not in the package: a script's directory comes first on ``sys.path``, and
``pi_embodied_services/utils`` holds a ``logging.py`` that would stand in for the standard
library's in everything huggingface_hub imports.
"""

import argparse
import os


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("repo")
    p.add_argument("revision")
    p.add_argument("dir", help='the target directory; "-" is the Hugging Face cache')
    p.add_argument("files", nargs="*", help="only these files (default: the snapshot)")
    p.add_argument(
        "--repo-type", default="model", choices=["model", "dataset", "space"]
    )
    return p.parse_args(argv)


def kwargs(args: argparse.Namespace) -> dict:
    """``snapshot_download``'s arguments for a parsed command line."""
    out = {
        "repo_id": args.repo,
        "revision": args.revision,
        "repo_type": args.repo_type,
    }
    if args.dir != "-":
        out["local_dir"] = args.dir
    if args.files:
        out["allow_patterns"] = list(args.files)
    return out


def missing(path: str, files: list[str]) -> list[str]:
    """The named files a download left out: ``snapshot_download`` treats names as patterns, so a
    name the repository does not have downloads nothing and raises nothing."""
    return [f for f in files if not os.path.exists(os.path.join(path, f))]


def main(argv: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    args = parse(argv)
    path = snapshot_download(**kwargs(args))
    gone = missing(path, args.files)
    if gone:
        raise SystemExit(f"{args.repo}@{args.revision} has no {', '.join(gone)}")
    print(path)


if __name__ == "__main__":
    main()
