import os
import shutil
import subprocess
import sys
from pathlib import Path


WORKDIR = Path("/kaggle/working")
REPO_ZIP = Path("/kaggle/input/qualitrace-source/qualitrace_source.zip")
REPO_DIR = WORKDIR / "QualiTrace"


def run(command: list[str], cwd: Path | None = None, env: dict | None = None):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def prepare_repo():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    REPO_DIR.mkdir(parents=True)
    shutil.unpack_archive(str(REPO_ZIP), str(REPO_DIR))


def main():
    prepare_repo()

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REPO_DIR)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR / "src")
    env["QUALITRACE_ENV"] = "kaggle"

    run([sys.executable, "src/train.py", "--config", "config/config.yaml"], cwd=REPO_DIR, env=env)
    run([sys.executable, "src/evaluate.py", "--config", "config/config.yaml"], cwd=REPO_DIR, env=env)


if __name__ == "__main__":
    main()
