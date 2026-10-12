#!/usr/bin/env python3
"""Re-export an omp session archive (`omp-session-archive/v1`) from the live session store.

The interactive client writes an archive bundle of the form

    manifest.json
    session.jsonl              full transcript
    artifacts/*                spilled tool output + subagent transcripts
    agents.json, jobs.json
    viewer.html, README.md

That export is a client action; this tool rebuilds the same bundle from what is on disk
(``$PI_CODING_AGENT_DIR``/sessions, default ``~/.omp/agent/sessions``) so a session can be
archived from the command line.

Usage:
    omp_archive_export.py [--session <session-id|path.jsonl>] [--cwd DIR]
                          [--out FILE.zip] [--template OLD.zip]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

SESSIONS = Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".omp" / "agent")) / "sessions"


def cwd_slug(cwd: str) -> str:
    return cwd.rstrip("/").replace("/", "-") or "-"


def find_session(cwd: str, session: str | None) -> Path:
    if session:
        p = Path(session)
        if p.is_file():
            return p
        if not p.suffix:
            p = p.with_suffix(".jsonl")
        for cand in SESSIONS.rglob("*.jsonl"):
            if cand.name.startswith(session) or session in cand.name:
                return cand
        raise SystemExit(f"session not found: {session}")
    base = SESSIONS / cwd_slug(cwd)
    files = sorted(base.glob("*.jsonl"), key=lambda q: q.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"no transcripts under {base}")
    return files[0]


def line_count(p: Path) -> int:
    with open(p, encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def build_manifest(session_file: Path, artifacts: list[tuple[str, int]], cwd: str,
                   title: str, artifact_dir: Path) -> dict:
    subagents = []
    plain = []
    for name, size in artifacts:
        if name.endswith(".jsonl") and name != "session.jsonl":
            subagents.append({"key": Path(name).stem, "file": f"artifacts/{name}",
                              "entries": line_count(artifact_dir / name),
                              "bytes": size, "aborted": False})
        else:
            plain.append({"file": f"artifacts/{name}", "bytes": size})
    main_bytes = session_file.stat().st_size
    files = 1 + len(artifacts) + 5          # manifest + artifacts + bundle extras
    total = main_bytes + sum(s for _, s in artifacts)
    return {
        "format": "omp-session-archive/v1",
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "source": {"sessionFile": str(session_file), "sessionId": session_file.stem.split("_")[-1],
                   "cwd": cwd, "title": title},
        "main": {"entries": line_count(session_file), "bytes": main_bytes},
        "subagents": subagents,
        "artifacts": plain + [{"file": s["file"], "bytes": s["bytes"]} for s in subagents],
        "agents": {"count": len(subagents), "agents": [s["key"] for s in subagents]},
        "jobs": {"liveAgents": [], "completed": [], "completedJobs": []},
        "totals": {"files": files, "bytes": total},
        "omitted": ["agents.json/jobs.json: live per-agent and job state is not "
                    "observable after the fact (rebuilt as an empty skeleton)"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="session id, id prefix, or path to the .jsonl")
    ap.add_argument("--cwd", default=os.getcwd(), help="cwd the session belongs to")
    ap.add_argument("--out", help="output zip (default: ./omp-archive-<ts>_<id>.zip)")
    ap.add_argument("--template", help="previous bundle to reuse viewer.html/README.md from")
    ap.add_argument("--title", default="", help="session title recorded in the manifest")
    args = ap.parse_args()

    session_file = find_session(args.cwd, args.session)
    session_id = session_file.stem.split("_")[-1]
    artifact_dir = session_file.with_suffix("")
    out = Path(args.out or f"omp-archive-{time.strftime('%Y-%m-%dT%H-%M-%S')}_{session_id}.zip")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "artifacts").mkdir()
        names = []
        if artifact_dir.is_dir():
            for src in sorted(artifact_dir.iterdir()):
                if src.is_file():
                    shutil.copy2(src, stage / "artifacts" / src.name)
                    names.append((src.name, src.stat().st_size))
        shutil.copy2(session_file, stage / "session.jsonl")

        manifest = build_manifest(session_file, names, args.cwd, args.title, artifact_dir)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        (stage / "agents.json").write_text(json.dumps({"count": manifest["agents"]["count"],
                                                       "agents": []}, indent=1))
        (stage / "jobs.json").write_text(json.dumps(manifest["jobs"], indent=1))

        if args.template and Path(args.template).exists():
            with zipfile.ZipFile(args.template) as zf:
                for extra in ("viewer.html", "README.md"):
                    try:
                        (stage / extra).write_bytes(zf.read(extra))
                    except KeyError:
                        pass
        if not (stage / "README.md").exists():
            (stage / "README.md").write_text(
                "omp session archive\n\n"
                f"- session: {session_id}\n- cwd: {args.cwd}\n"
                f"- transcript: session.jsonl ({manifest['main']['entries']} entries)\n"
                f"- artifacts: {len(names)} files\n"
                "Re-exported with tools/omp_archive_export.py (omp-archive-export).\n")

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())
    print(f"wrote {out} ({out.stat().st_size:,} bytes, "
          f"{manifest['main']['entries']} transcript entries, {len(names)} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
