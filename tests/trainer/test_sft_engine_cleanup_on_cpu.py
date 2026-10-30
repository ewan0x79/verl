# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the SFT shell harness with synthetic checkpoints and a fake Python command."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _checkpoint_root():
    """Reject cleanup outside the marked test directory, including symlink escapes."""
    root = Path(os.environ["SFT_TEST_ROOT"])
    raw = os.environ.get("ckpts_home", "")
    if not raw or not root.is_absolute() or root == root.parent:
        raise ValueError("Missing or unsafe test root")
    checkpoint = Path(raw)
    if checkpoint != root / "checkpoints" or ".." in checkpoint.parts:
        raise ValueError("Checkpoint directory must be a direct child of the test root")
    if any(path.is_symlink() for path in [checkpoint, *checkpoint.parents]):
        raise ValueError("Symlink in checkpoint path")
    if (root / ".sft-shell-test").read_text() != "isolated cleanup test\n":
        raise ValueError("Missing test ownership marker")
    if any(path.is_symlink() for path in checkpoint.rglob("*")):
        raise ValueError("Symlink in checkpoint tree")
    return checkpoint


def _guarded_rm(arguments):
    """Execute the original rm only for children of this test's checkpoint directory."""
    checkpoint = _checkpoint_root()
    if arguments[:2] != ["-rf", "--"] or len(arguments) < 3:
        raise ValueError("Unexpected rm arguments")
    for raw in arguments[2:]:
        path = Path(raw)
        if ".." in path.parts or path == checkpoint or not path.is_relative_to(checkpoint):
            raise ValueError("Cleanup escaped the checkpoint directory")
    return subprocess.run([os.environ["SFT_TEST_RM"], *arguments], check=False).returncode


def _state(checkpoint):
    """Observe input, output and unrelated files without inspecting checkpoint contents."""
    return {
        "input": (checkpoint / "global_step_2/input.marker").exists(),
        "generated": (checkpoint / "global_step_3/output.marker").exists(),
        "notes": (checkpoint / "notes.txt").exists(),
        "hidden": (checkpoint / ".hidden-note").exists(),
        "tracker": (checkpoint / "latest_checkpointed_iteration.txt").exists(),
    }


def _mock_python(arguments):
    """Record startup effects, synthesize a save, then return the requested exit code."""
    checkpoint = _checkpoint_root()
    assert f"trainer.default_local_dir={checkpoint}" in arguments
    expected_resume = os.environ["SFT_TEST_EXPECTED_RESUME_MODE"]
    assert f"trainer.resume_mode={expected_resume}" in arguments
    (checkpoint.parent / "training-entry.json").write_text(json.dumps(_state(checkpoint)))
    output = checkpoint / "global_step_3/output.marker"
    output.parent.mkdir()
    output.write_text("synthetic output\n")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("3\n")
    (checkpoint.parent / "training-exit.json").write_text(json.dumps(_state(checkpoint)))
    return int(os.environ["SFT_TEST_EXIT_CODE"])


@pytest.mark.parametrize("mode", ["spmd", "ray"])
@pytest.mark.parametrize("exit_code", [0, 37], ids=["success", "failure"])
@pytest.mark.parametrize(
    "resume_mode",
    [None, "", "disable", "auto", "resume_path"],
    ids=["default", "resume-empty", "disable", "auto", "resume-path"],
)
@pytest.mark.parametrize(
    "keep",
    [None, "", "False", "True", "true"],
    ids=["keep-default", "keep-empty", "keep-false", "keep-true", "keep-lowercase"],
)
def test_sft_engine_checkpoint_cleanup(tmp_path, mode, exit_code, resume_mode, keep):
    """Preserve resume inputs by default and honor explicit cleanup on success or failure."""
    effective_resume = resume_mode or "disable"
    root = tmp_path.resolve()
    (root / ".sft-shell-test").write_text("isolated cleanup test\n")
    checkpoint = root / "checkpoints"
    (checkpoint / "global_step_2").mkdir(parents=True)
    (checkpoint / "global_step_2/input.marker").write_text("synthetic resume input\n")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("2\n")
    (checkpoint / "notes.txt").write_text("ordinary non-checkpoint file\n")
    (checkpoint / ".hidden-note").write_text("hidden control file\n")
    outside = root / "outside.marker"
    outside.write_text("outside checkpoint directory\n")

    bin_dir = root / "bin"
    bin_dir.mkdir()
    for name, action in [("python", "--mock-python"), ("python3", "--mock-python"), ("rm", "--guarded-rm")]:
        shim = bin_dir / name
        shim.write_text(f'#!/bin/sh\nexec "$SFT_TEST_PYTHON" "$SFT_TEST_DRIVER" {action} "$@"\n')
        shim.chmod(0o755)
    # Use an allowlist so inherited resume flags, BASH_ENV and glob settings
    # cannot redirect the test to a real checkpoint or training command.
    environment = {
        "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
        "HOME": str(root),
        "ckpts_home": str(checkpoint),
        "mode": mode,
        "MODEL_PATH": str(root / "unused-model"),
        "DATASET_DIR": str(root / "unused-data"),
        "SFT_TEST_ROOT": str(root),
        "SFT_TEST_PYTHON": sys.executable,
        "SFT_TEST_DRIVER": str(Path(__file__).resolve()),
        "SFT_TEST_RM": shutil.which("rm"),
        "SFT_TEST_EXIT_CODE": str(exit_code),
        "SFT_TEST_EXPECTED_RESUME_MODE": effective_resume,
        # Populate the override array: Bash 3.2 treats an empty array under
        # nounset as unbound. This flag does not participate in shell cleanup.
        "ASYNC_SAVE": "True",
    }
    if resume_mode is not None:
        environment["RESUME_MODE"] = resume_mode
    if keep is not None:
        environment["KEEP_CHECKPOINTS"] = keep
    script = Path(__file__).resolve().parents[2] / "tests/special_e2e/sft/run_sft_engine.sh"
    result = subprocess.run(
        [shutil.which("bash"), str(script)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    entry = json.loads((root / "training-entry.json").read_text())
    before_exit = json.loads((root / "training-exit.json").read_text())
    fresh = effective_resume == "disable"
    retain = keep == "True" or (keep in (None, "") and not fresh)
    assert entry == {"input": not fresh, "generated": False, "notes": not fresh, "hidden": True, "tracker": not fresh}
    assert before_exit == {"input": not fresh, "generated": True, "notes": not fresh, "hidden": True, "tracker": True}
    assert _state(checkpoint) == {
        "input": not fresh and retain,
        "generated": retain,
        "notes": not fresh and retain,
        "hidden": True,
        "tracker": retain,
    }
    assert checkpoint.is_dir()
    assert outside.read_text() == "outside checkpoint directory\n"


if __name__ == "__main__":
    if sys.argv[1] == "--guarded-rm":
        sys.exit(_guarded_rm(sys.argv[2:]))
    if sys.argv[1] == "--mock-python":
        sys.exit(_mock_python(sys.argv[2:]))
    raise ValueError("Unknown test helper command")
