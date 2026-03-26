"""Allow ``python -m mjviser model.xml`` to launch the active viewer."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

from mjviser import Viewer


def _resolve_path(arg: str) -> Path:
  """Resolve a model path, falling back to recursive glob search in CWD."""
  path = Path(arg)
  if path.is_file():
    return path

  # If pointing at a directory, search for XMLs inside it.
  if path.is_dir():
    matches = sorted(path.glob("**/*.xml"))
  elif "*" in arg:
    matches = sorted(Path.cwd().glob(arg))
  else:
    matches = sorted(Path.cwd().glob(f"**/*{arg}*.xml"))
  if not matches:
    print(f"No XML files matching '{arg}' found in {Path.cwd()}")
    sys.exit(1)
  if len(matches) == 1:
    print(f"Found: {matches[0]}")
    return matches[0]

  # Multiple matches — let the user pick.
  print(f"Multiple matches for '{arg}':")
  for i, m in enumerate(matches, 1):
    try:
      label = m.relative_to(Path.cwd())
    except ValueError:
      label = m
    print(f"  [{i}] {label}")
  try:
    choice = input("Select [1]: ").strip()
    idx = int(choice) - 1 if choice else 0
    return matches[idx]
  except (ValueError, IndexError, KeyboardInterrupt):
    sys.exit(1)


def main() -> None:
  if len(sys.argv) < 2:
    print("Usage: python -m mjviser <model.xml>")
    sys.exit(1)

  path = _resolve_path(sys.argv[1])
  model = mujoco.MjModel.from_xml_path(str(path))
  data = mujoco.MjData(model)
  Viewer(model, data).run()


if __name__ == "__main__":
  main()
