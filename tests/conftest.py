import os

# Must be set before any project module is imported so settings validation passes
os.environ.setdefault("OPENAI_API_KEY", "sk-test-ci-key-not-real")
