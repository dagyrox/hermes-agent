"""Offline argv contract for Apollo's native and explicit fallback entry points."""
import json
import os
from pathlib import Path
import subprocess


def test_apollo_uses_shared_native_wrapper_and_explicit_codex(tmp_path):
    capture = tmp_path / "capture"
    recorder = tmp_path / "record"
    recorder.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    recorder.chmod(0o755)
    launcher = Path(__file__).resolve().parents[2] / "scripts" / "launchers" / "apollo"
    env = {
        **os.environ,
        "CAPTURE": str(capture),
        "APOLLO_PYTHON": str(recorder),
        "APOLLO_HERMES_BIN": str(recorder),
        "APOLLO_CLAUDE_BIN": "/offline/claude with spaces",
        "APOLLO_WORKDIR": str(tmp_path),
    }
    prefix = ["-m", "hermes_cli.native_claude", "--profile", "apollo",
              "--claude-bin", "/offline/claude with spaces"]
    for command, expected in (
        (["status"], ["--status"]),
        (["login"], ["--login"]),
        (["smoke"], ["--model", "sonnet", "--query", "Reply with exactly: CLAUDE_CODE_SMOKE_OK"]),
        (["claude", "--resume", "session with spaces"],
         ["--model", "sonnet", "--name", "Apollo", "--", "--resume", "session with spaces"]),
    ):
        subprocess.run(["bash", str(launcher), *command], env=env, check=True)
        assert json.loads(capture.read_text()) == prefix + expected
    subprocess.run(["bash", str(launcher), "codex", "chat", "-q", "two words"], env=env, check=True)
    assert json.loads(capture.read_text()) == [
        "-p", "apollo", "--provider", "openai-codex", "-m", "gpt-6.1-sol",
        "chat", "-q", "two words",
    ]
