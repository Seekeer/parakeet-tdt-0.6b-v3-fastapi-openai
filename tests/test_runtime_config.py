import os
import subprocess
import sys

from conftest import TEST_API_KEY


def check_config(code="print(INFER_WORKERS, CHUNK_MAX_SEC, DEFAULT_MODEL)", **overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PARAKEET_")}
    env.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", f"from parakeet_service.config import *; {code}"],
        env=env,
        capture_output=True,
        text=True,
    )


def test_cpu_defaults():
    result = check_config(PARAKEET_API_KEYS=TEST_API_KEY)
    assert result.returncode == 0, result.stderr
    assert "4 75.0 parakeet-tdt-0.6b-v3" in result.stdout


def test_missing_api_keys_fail_closed():
    result = check_config()
    assert result.returncode != 0
    assert "No API keys configured" in result.stderr


def test_auth_can_be_disabled_explicitly():
    result = check_config("print(AUTH_DISABLED, len(API_KEY_DIGESTS))", PARAKEET_AUTH_DISABLED="true")
    assert result.returncode == 0, result.stderr
    assert "True 0" in result.stdout


def test_short_api_key_is_rejected():
    result = check_config(PARAKEET_API_KEYS="short")
    assert result.returncode != 0
    assert "shorter than 16" in result.stderr


def test_non_ascii_api_key_is_rejected():
    result = check_config(PARAKEET_API_KEYS="ключ-0123456789abcdef")
    assert result.returncode != 0
    assert "printable ASCII" in result.stderr


def test_keys_from_env_and_file_are_merged(tmp_path):
    keys_file = tmp_path / "keys.txt"
    # utf-8-sig writes a BOM, as Windows Notepad may.
    keys_file.write_text(
        "# client keys\n\nfile-key-0123456789abcdef\n" + TEST_API_KEY + "\n",
        encoding="utf-8-sig",
    )
    result = check_config(
        "print(len(API_KEY_DIGESTS))",
        PARAKEET_API_KEYS=f" {TEST_API_KEY} ,env-key-0123456789abcdef,",
        PARAKEET_API_KEYS_FILE=str(keys_file),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3"


def test_unreadable_keys_file_fails(tmp_path):
    result = check_config(PARAKEET_API_KEYS_FILE=str(tmp_path / "missing.txt"))
    assert result.returncode != 0
    assert "PARAKEET_API_KEYS_FILE could not be read" in result.stderr
