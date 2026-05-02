#!/usr/bin/env python3
"""Minimal smoke test for deployment readiness."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.config import settings


def check_endpoint(client, path, expected_statuses):
    response = client.get(path)
    ok = response.status_code in expected_statuses
    print(f"[{ 'PASS' if ok else 'FAIL' }] GET {path} -> {response.status_code}")
    return ok, response


def main():
    parser = argparse.ArgumentParser(description="Run minimal smoke tests")
    parser.add_argument(
        "--require-healthy",
        action="store_true",
        help="Fail when /api/healthz is not 200",
    )
    parser.add_argument(
        "--use-current-db",
        action="store_true",
        help="Use configured DATABASE_URL instead of temporary sqlite for smoke test",
    )
    args = parser.parse_args()

    # Use sqlite by default so smoke checks can run without external DB dependencies.
    if not args.use_current_db:
        settings.database.url = "sqlite:///:memory:"

    app = create_app()
    app.testing = True
    client = app.test_client()

    all_ok = True

    # Basic route checks
    for path, statuses in (
        ("/upload", {200}),
        ("/api/healthz", {200, 503}),
    ):
        ok, response = check_endpoint(client, path, statuses)
        all_ok = all_ok and ok
        if path == "/api/healthz":
            payload = response.get_json(silent=True) or {}
            has_checks = isinstance(payload.get("checks"), dict)
            print(f"[{'PASS' if has_checks else 'FAIL'}] health payload has checks")
            all_ok = all_ok and has_checks
            if args.require_healthy and response.status_code != 200:
                print("[FAIL] health is degraded while --require-healthy is set")
                all_ok = False

    # Filesystem checks
    upload_exists = Path(settings.storage.upload_folder).exists()
    output_exists = Path(settings.storage.output_folder).exists()
    model_exists = Path(settings.model.model_path).exists()

    print(f"[{'PASS' if upload_exists else 'FAIL'}] upload dir: {settings.storage.upload_folder}")
    print(f"[{'PASS' if output_exists else 'FAIL'}] output dir: {settings.storage.output_folder}")
    print(f"[{'PASS' if model_exists else 'FAIL'}] model path: {settings.model.model_path}")

    all_ok = all_ok and upload_exists and output_exists and model_exists

    if all_ok:
        print("\nSmoke test passed")
        return 0

    print("\nSmoke test failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
