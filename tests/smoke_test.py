"""Smoke test for mjviser package."""

import sys


def test_imports() -> None:
  from mjviser import Viewer, ViserMujocoScene  # noqa: F401


if __name__ == "__main__":
  try:
    test_imports()
    print("✓ Smoke test passed!")
    sys.exit(0)
  except Exception as e:
    print(f"✗ Smoke test failed: {e}")
    sys.exit(1)
