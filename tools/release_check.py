#!/usr/bin/env python3
"""Pre-release validation script for deployment readiness."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        print(f"[FAIL] command exited with code {result.returncode}")
        return False
    print("[PASS]")
    return True


def check_required_files():
    required = [
        "run.py",
        "gunicorn.conf.py",
        "docker-compose.yml",
        "docker-compose.prod.yml",
        "requirements.txt",
        "tools/smoke_test.py",
    ]
    ok = True
    for rel in required:
        exists = (ROOT / rel).exists()
        print(f"[{'PASS' if exists else 'FAIL'}] required file: {rel}")
        ok = ok and exists
    return ok


def check_requirements_entries():
    text = (ROOT / "requirements.txt").read_text()
    required_entries = [
        "flask",
        "gunicorn",
        "pydantic",
        "pydantic-settings",
        "ultralytics",
        "boxmot",
    ]
    ok = True
    for entry in required_entries:
        present = entry in text
        print(f"[{'PASS' if present else 'FAIL'}] requirements contains '{entry}'")
        ok = ok and present
    return ok


def check_model_path_exists():
    model_path = ROOT / "models" / "bifpn_best.pt"
    exists = model_path.exists()
    print(f"[{'PASS' if exists else 'FAIL'}] model file exists: {model_path}")
    return exists


def main():
    parser = argparse.ArgumentParser(description="Run release readiness checks")
    parser.add_argument(
        "--use-current-db",
        action="store_true",
        help="Run smoke test against configured DATABASE_URL",
    )
    args = parser.parse_args()

    all_ok = True

    print("== Static checks ==")
    all_ok = check_required_files() and all_ok
    all_ok = check_requirements_entries() and all_ok
    all_ok = check_model_path_exists() and all_ok

    print("\n== Runtime checks ==")
    all_ok = run_cmd([sys.executable, "-m", "compileall", "app", "tools", "run.py"]) and all_ok

    smoke_cmd = [sys.executable, "tools/smoke_test.py", "--require-healthy"]
    if args.use_current_db:
        smoke_cmd.append("--use-current-db")
    all_ok = run_cmd(smoke_cmd) and all_ok

    if all_ok:
        print("\nRelease check passed")
        return 0

    print("\nRelease check failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
