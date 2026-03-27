"""Manual simulation loop using ViserMujocoScene directly.

uv run examples/passive_viewer.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import viser

from mjviser import ViserMujocoScene


def main() -> None:
  if len(sys.argv) > 1:
    model = mujoco.MjModel.from_xml_path(sys.argv[1])
  else:
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "humanoid.xml"))

  data = mujoco.MjData(model)

  server = viser.ViserServer()
  scene = ViserMujocoScene(server, model, num_envs=1)

  scene.create_visualization_gui()

  print("Running simulation. Press Ctrl+C to stop.")
  step_dt = model.opt.timestep
  try:
    while True:
      t0 = time.perf_counter()
      mujoco.mj_step(model, data)
      scene.update_from_mjdata(data)
      elapsed = time.perf_counter() - t0
      remaining = step_dt - elapsed
      if remaining > 0:
        time.sleep(remaining)
  except KeyboardInterrupt:
    print("\nStopped.")
    server.stop()


if __name__ == "__main__":
  main()
