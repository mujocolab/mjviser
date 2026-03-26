"""Ghost overlay: render a time-delayed copy as a semi-transparent ghost.

Demonstrates how to inject custom rendering logic into the viewer
using ``render_fn``. The ghost shows where the humanoid was N steps
ago, useful for comparing predictions against ground truth.

uv run examples/ghost_overlay.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import viser.transforms as vtf

from mjviser import Viewer
from mjviser.conversions import (
  get_body_name,
  is_fixed_body,
  merge_geoms,
)
from mjviser.scene import ViserMujocoScene

GHOST_DELAY_STEPS = 100
GHOST_RGBA = np.array([0.4, 0.5, 0.9, 0.5])


def _build_ghost_handles(
  scene: ViserMujocoScene,
  model: mujoco.MjModel,
) -> dict[int, object]:
  """Create semi-transparent batched mesh handles for non-fixed bodies."""
  body_geoms: dict[int, list[int]] = {}
  for i in range(model.ngeom):
    body_id = model.geom_bodyid[i]
    if is_fixed_body(model, body_id):
      continue
    if model.geom_rgba[i, 3] == 0:
      continue
    if model.geom_group[i] > 2:
      continue
    body_geoms.setdefault(body_id, []).append(i)

  handles = {}
  color_uint8 = (np.clip(GHOST_RGBA[:3], 0, 1) * 255).astype(np.uint8)

  for body_id, geom_ids in body_geoms.items():
    mesh = merge_geoms(model, geom_ids)
    name = get_body_name(model, body_id)
    handle = scene.server.scene.add_batched_meshes_simple(
      f"/ghost/{name}",
      mesh.vertices,
      mesh.faces,
      batched_wxyzs=np.array([[1, 0, 0, 0]], dtype=np.float32),
      batched_positions=np.array([[0, 0, 0]], dtype=np.float32),
      batched_colors=color_uint8,
      opacity=GHOST_RGBA[3],
      cast_shadow=False,
      receive_shadow=False,
    )
    handles[body_id] = handle

  return handles


def main() -> None:
  path = Path(__file__).parent / "humanoid.xml"
  model = mujoco.MjModel.from_xml_path(str(path))
  data = mujoco.MjData(model)

  # Ring buffer of past body transforms.
  history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=GHOST_DELAY_STEPS)
  ghost_handles: dict[int, object] = {}

  def step(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    # Save current poses before stepping.
    history.append((d.xpos.copy(), d.xmat.copy()))
    mujoco.mj_step(m, d)

  def render(scene: ViserMujocoScene) -> None:
    nonlocal ghost_handles

    # Normal render.
    scene.update_from_mjdata(data)

    # Create ghost handles once the scene is ready.
    if not ghost_handles:
      ghost_handles = _build_ghost_handles(scene, model)

    # Update ghost from history.
    if len(history) < GHOST_DELAY_STEPS:
      return

    xpos, xmat = history[0]
    offset = scene._scene_offset
    xquat = vtf.SO3.from_matrix(xmat.reshape(-1, 3, 3)).wxyz

    for body_id, handle in ghost_handles.items():
      pos = (xpos[body_id] + offset).astype(np.float32)
      quat = xquat[body_id].astype(np.float32)
      handle.batched_positions = pos[None]
      handle.batched_wxyzs = quat[None]

  Viewer(model, data, step_fn=step, render_fn=render).run()


if __name__ == "__main__":
  main()
