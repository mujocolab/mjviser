"""Tests for interactive drag perturbation.

These drive the handler's state the way the viser drag callbacks would
(grab point + cursor target in world coords) and check that the MuJoCo
perturbation engine is wired up correctly: force direction and magnitude,
off-center torque, critical damping, force clearing, paused repositioning,
the viser/mujoco frame offset, and model-rebuild bounds safety.
"""

import mujoco
import numpy as np
import pytest
import viser

from mjviser.interaction import PerturbationHandler

_FREE_BODY_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="ball" pos="0 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""

_BID = 1  # the free body


def _stop(server: viser.ViserServer) -> None:
  try:
    server.stop()
  except RuntimeError:
    pass


@pytest.fixture
def env():
  model = mujoco.MjModel.from_xml_string(_FREE_BODY_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  server = viser.ViserServer(port=0)
  handler = PerturbationHandler(server, model, data)
  yield handler, model, data
  _stop(server)


def _start_drag(
  handler: PerturbationHandler,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  bid: int,
  grab_world: np.ndarray,
  target_world: np.ndarray,
  scene_offset: np.ndarray | None = None,
) -> None:
  """Mimic a viser drag: cache pose, grab a point, then set the target."""
  if scene_offset is None:
    scene_offset = np.zeros(3)
  body_xpos = data.xpos[None].copy()
  body_xmat = data.xmat.reshape(model.nbody, 3, 3)[None].copy()
  handler.update_state(body_xpos, body_xmat, 0, scene_offset)
  handler._on_drag_start(bid, tuple(grab_world))
  handler._target_viser = np.array(target_world, dtype=float)


def test_no_drag_applies_no_force(env):
  handler, model, data = env
  assert handler.apply(model, data, paused=False) is False
  assert np.all(data.xfrc_applied == 0.0)


def test_force_points_toward_target(env):
  handler, model, data = env
  com = data.xipos[_BID].copy()
  _start_drag(handler, model, data, _BID, com, com + [0.3, 0.0, 0.0])
  handler.apply(model, data, paused=False)
  force = data.xfrc_applied[_BID, :3]
  # stiffness(100) * localmass(=mass 2) * displacement(0.3) = 60, +x only.
  np.testing.assert_allclose(force, [60.0, 0.0, 0.0], atol=1e-6)
  np.testing.assert_allclose(data.xfrc_applied[_BID, 3:], 0.0, atol=1e-6)


def test_drag_moves_body_toward_target(env):
  handler, model, data = env
  com = data.xipos[_BID].copy()
  target = com + [0.5, 0.0, 0.0]
  _start_drag(handler, model, data, _BID, com, target)
  for _ in range(200):
    handler.apply(model, data, paused=False)
    mujoco.mj_step(model, data)
  assert data.xpos[_BID][0] > 0.3  # pulled most of the way to +0.5


def test_offcenter_grab_produces_torque(env):
  handler, model, data = env
  grab = data.xipos[_BID] + [0.0, 0.0, 0.1]  # top face, above COM
  _start_drag(handler, model, data, _BID, grab, grab + [0.3, 0.0, 0.0])
  handler.apply(model, data, paused=False)
  torque = data.xfrc_applied[_BID, 3:]
  # +x force applied 0.1 above COM -> torque about +y.
  assert torque[1] > 1.0
  assert abs(torque[0]) < 1e-6 and abs(torque[2]) < 1e-6


def test_damping_reduces_force_when_moving_toward_target(env):
  handler, model, data = env
  com = data.xipos[_BID].copy()
  _start_drag(handler, model, data, _BID, com, com + [0.3, 0.0, 0.0])
  handler.apply(model, data, paused=False)
  rest_force = data.xfrc_applied[_BID, 0]
  # Now give the body velocity toward the target; damping should cut force.
  data.qvel[0] = 2.0
  mujoco.mj_forward(model, data)
  handler.apply(model, data, paused=False)
  assert data.xfrc_applied[_BID, 0] < rest_force


def test_drag_end_clears_applied_force(env):
  handler, model, data = env
  com = data.xipos[_BID].copy()
  _start_drag(handler, model, data, _BID, com, com + [0.3, 0.0, 0.0])
  handler.apply(model, data, paused=False)
  assert np.any(data.xfrc_applied[_BID] != 0.0)
  # End the drag and apply again: the body's force must be zeroed.
  handler._drag_body_id = None
  handler.apply(model, data, paused=False)
  assert np.all(data.xfrc_applied[_BID] == 0.0)


def test_clear_resets_state_and_force(env):
  handler, model, data = env
  com = data.xipos[_BID].copy()
  _start_drag(handler, model, data, _BID, com, com + [0.3, 0.0, 0.0])
  handler.apply(model, data, paused=False)
  handler.clear()
  assert handler.selected_body_id is None
  assert handler._drag_body_id is None
  handler.apply(model, data, paused=False)
  assert np.all(data.xfrc_applied[_BID] == 0.0)


def test_paused_repositions_free_body(env):
  handler, model, data = env
  com = data.xipos[_BID].copy()
  target = com + [0.3, 0.0, 0.0]
  _start_drag(handler, model, data, _BID, com, target)
  assert handler.apply(model, data, paused=True) is True
  np.testing.assert_allclose(data.qpos[:3], [0.3, 0.0, 0.0], atol=1e-6)
  assert np.all(data.xfrc_applied == 0.0)  # pose drag applies no force


def test_scene_offset_is_subtracted(env):
  handler, model, data = env
  com = data.xipos[_BID].copy()
  offset = np.array([10.0, 0.0, 0.0])
  # Grab and target are reported in viser frame (= mujoco + offset).
  _start_drag(
    handler, model, data, _BID, com + offset, com + offset + [0.3, 0.0, 0.0], offset
  )
  handler.apply(model, data, paused=False)
  # Despite the large offset, the force is the same as without it.
  np.testing.assert_allclose(data.xfrc_applied[_BID, :3], [60.0, 0.0, 0.0], atol=1e-6)


def _start_rotate(handler, bid, delta_quat):
  """Mimic a rotate drag: latch the body and set the target rotation."""
  handler._drag_body_id = bid
  handler._drag_rotate = True
  handler._rotate_initial_quat = None
  handler._rotate_delta_quat = np.asarray(delta_quat, dtype=float)


def test_rotate_drag_applies_torque(env):
  handler, model, data = env
  q = np.empty(4)
  mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.pi / 2)
  _start_rotate(handler, _BID, q)
  handler.apply(model, data, paused=False)
  torque = data.xfrc_applied[_BID, 3:]
  # Spring torque points along +z (toward the target yaw), no net force.
  assert torque[2] > 1.0
  np.testing.assert_allclose(data.xfrc_applied[_BID, :3], 0.0, atol=1e-6)


def test_rotate_drag_paused_reorients_free_body(env):
  handler, model, data = env
  q = np.empty(4)
  mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.pi / 2)
  _start_rotate(handler, _BID, q)
  assert handler.apply(model, data, paused=True) is True
  np.testing.assert_allclose(
    data.qpos[3:7], [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], atol=1e-6
  )


def test_stale_body_id_after_rebuild_is_safe(env):
  handler, model, data = env
  com = data.xipos[_BID].copy()
  _start_drag(handler, model, data, _BID, com, com + [0.3, 0.0, 0.0])
  # Simulate a model rebuild that dropped bodies: id now out of range.
  handler._drag_body_id = model.nbody + 5
  assert handler.apply(model, data, paused=False) is False
  assert np.all(data.xfrc_applied == 0.0)
