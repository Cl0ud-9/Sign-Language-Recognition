import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, script), *args], capture_output=True, text=True,
                          timeout=120, cwd=ROOT)


@pytest.mark.parametrize("script", ["app.py", "collectdata.py"])
def test_missing_camera_is_reported(script):
    result = run(script, "--camera", "99")
    assert result.returncode == 1
    assert "Could not open camera 99" in result.stderr
    assert "--camera 0" in result.stderr


@pytest.mark.parametrize("script", ["app.py", "collectdata.py"])
def test_help(script):
    result = run(script, "--help")
    assert result.returncode == 0
    assert "--camera" in result.stdout
