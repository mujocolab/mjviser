"""Allow ``python -m mjviser model.xml`` to launch the active viewer."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import mujoco
import viser

from mjviser import Viewer


def _pick(prompt: str, options: list[str]) -> int:
  """Show a numbered list and return the selected index."""
  for i, opt in enumerate(options, 1):
    print(f"  [{i}] {opt}")
  try:
    choice = input(f"{prompt} [1]: ").strip()
    return int(choice) - 1 if choice else 0
  except (ValueError, IndexError, KeyboardInterrupt):
    sys.exit(1)


def _resolve_from_robot_descriptions(name: str) -> Path | None:
  """Try to resolve a model name via the robot_descriptions package."""
  try:
    from robot_descriptions import DESCRIPTIONS  # ty: ignore[import-not-found]
  except ImportError:
    return None

  suffix = "_mj_description"
  candidates = {
    n.removesuffix(suffix): n for n, d in DESCRIPTIONS.items() if d.has_mjcf
  }

  # Exact match: "go1" or "go1_mj_description".
  desc_name = candidates.get(name) or candidates.get(name.removesuffix(suffix))
  if desc_name is None:
    # Fuzzy substring match.
    hits = [(short, full) for short, full in candidates.items() if name in short]
    if not hits:
      return None
    if len(hits) == 1:
      desc_name = hits[0][1]
    else:
      print(f"Multiple robot_descriptions matches for '{name}':")
      idx = _pick("Select", [h[0] for h in hits])
      desc_name = hits[idx][1]

  desc = importlib.import_module(f"robot_descriptions.{desc_name}")
  # Prefer scene.xml when available.
  scene = Path(desc.PACKAGE_PATH) / "scene.xml"
  return scene if scene.is_file() else Path(desc.MJCF_PATH)


def _resolve_path(arg: str) -> Path:
  """Resolve a model path via file, robot_descriptions, or glob search."""
  path = Path(arg)
  if path.is_file():
    return path

  # Try robot_descriptions before expensive glob.
  rd_path = _resolve_from_robot_descriptions(arg)
  if rd_path is not None:
    print(f"Found: {rd_path}")
    return rd_path

  # Glob search in CWD.
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

  print(f"Multiple matches for '{arg}':")
  labels = []
  for m in matches:
    try:
      labels.append(str(m.relative_to(Path.cwd())))
    except ValueError:
      labels.append(str(m))
  return matches[_pick("Select", labels)]


def main() -> None:
  parser = argparse.ArgumentParser(
    prog="mjviser",
    description=(
      "Interactive MuJoCo viewer powered by Viser.\n\n"
      "MODEL can be a file path, a robot_descriptions name (e.g. 'go1'),\n"
      "or a substring / glob pattern to search under the current directory."
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "model",
    metavar="MODEL",
    help="path, robot_descriptions name, or search pattern for a .xml model",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="port to bind the Viser server to (default: 8080)",
  )
  args = parser.parse_args()

  path = _resolve_path(args.model)
  model = mujoco.MjModel.from_xml_path(str(path))
  data = mujoco.MjData(model)
  server = viser.ViserServer(port=args.port)
  Viewer(model, data, server=server).run()


if __name__ == "__main__":
  main()
