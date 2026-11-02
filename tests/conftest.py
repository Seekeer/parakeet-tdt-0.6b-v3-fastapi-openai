import os

# parakeet_service.config refuses to import without API keys and reads auth
# settings once at import time. Pin them before any test module imports the
# package so a developer's own PARAKEET_* variables cannot change the outcome.
TEST_API_KEY = "test-key-0123456789abcdef"
os.environ["PARAKEET_API_KEYS"] = TEST_API_KEY
os.environ.pop("PARAKEET_API_KEYS_FILE", None)
os.environ.pop("PARAKEET_AUTH_DISABLED", None)
