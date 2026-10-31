import asyncio
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import viser

from mjviser import ViserMujocoScene
from mjviser.interaction import _DRAG_BINDINGS, Perturbation

_BOX_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

# A camera 3 m in front of the box (along -y), looking at it.
_CAMERA = SimpleNamespace(
  position=(0.0, -3.0, 1.0),
  look_at=(0.0, 0.0, 1.0),
  up_direction=(0.0, 0.0, 1.0),
  fov=np.pi / 3,
  aspect=1.0,
)
_FRONT_FACE = (0.0, -0.1, 1.0)


@pytest.fixture
def setup():
  model = mujoco.MjModel.from_xml_string(_BOX_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  server = viser.ViserServer(port=0)
  yield Perturbation(server, model), data
  server.stop()


def _drag(pert, modifier, phase, screen_pos=(0.5, 0.5)):
  """Send one right-button drag event through the callback viser would call."""
  kind, action = next(
    (k, a) for b, m, k, a in _DRAG_BINDINGS if b == "right" and m == modifier
  )
  event = SimpleNamespace(
    client=SimpleNamespace(camera=_CAMERA),
    phase=phase,
    start_position=_FRONT_FACE,
    end_screen_pos=screen_pos,
  )
  asyncio.run(pert._drag_callback(kind, action)(event))


def test_click_selects_body_and_empty_space_deselects(setup):
  pert, data = setup
  pert._select(data, np.array(_CAMERA.position), np.array([0.0, 1.0, 0.0]))
  assert pert.pert.select == 1
  np.testing.assert_allclose(pert.pert.localpos, [0, -0.1, 0], atol=1e-9)

  pert._select(data, np.array([2.0, -3.0, 1.0]), np.array([0.0, 0.0, -1.0]))
  assert pert.pert.select == 0


def test_drag_pulls_body_toward_cursor_then_releases(setup):
  pert, data = setup
  _drag(pert, "cmd/ctrl", "start")
  _drag(pert, "cmd/ctrl", "update", (0.6, 0.5))  # Cursor moves right (+x).
  pert.step(data, paused=False)
  assert data.xfrc_applied[1, 0] > 0
  assert abs(data.xfrc_applied[1, 2]) < 1e-9

  _drag(pert, "cmd/ctrl", "end")
  pert.step(data, paused=False)
  assert not pert.pert.active
  np.testing.assert_array_equal(data.xfrc_applied, 0)


def test_shift_mid_drag_continues_in_horizontal_plane(setup):
  pert, data = setup
  _drag(pert, "cmd/ctrl", "start")
  _drag(pert, "cmd/ctrl", "update", (0.6, 0.5))
  _drag(pert, "cmd/ctrl", "end")
  pert.step(data, paused=False)
  target = pert.pert.refselpos.copy()

  # viser starts a new segment when Shift goes down. The drag must resume
  # (keeping its target), then move in the horizontal plane.
  _drag(pert, "cmd/ctrl+shift", "start", (0.6, 0.5))
  pert.step(data, paused=False)
  np.testing.assert_allclose(pert.pert.refselpos, target)
  _drag(pert, "cmd/ctrl+shift", "update", (0.6, 0.4))
  pert.step(data, paused=False)
  moved = pert.pert.refselpos - target
  assert abs(moved[1]) > 0
  assert abs(moved[2]) < 1e-9


def test_paused_drag_poses_body(setup):
  pert, data = setup
  _drag(pert, "cmd/ctrl", "start")
  _drag(pert, "cmd/ctrl", "update", (0.5, 0.4))  # Cursor moves up (+z).
  assert pert.step(data, paused=True)
  assert data.qpos[2] > 1.0


def test_selected_body_is_swapped_for_highlight(humanoid_model):
  server = viser.ViserServer(port=0)
  scene = ViserMujocoScene(server, humanoid_model, num_envs=2)
  pert = scene.enable_perturbation()
  data = mujoco.MjData(humanoid_model)
  mujoco.mj_forward(humanoid_model, data)

  pert.pert.select = 4
  scene.update_from_mjdata(data)
  group = next(mg for mg in scene._mesh_groups if 4 in mg.body_ids)
  assert scene._highlight is not None
  # Hidden in the active env only, so the copy never overlaps it.
  scales = np.asarray(group.handle.batched_scales).reshape(2, -1)
  assert scales[0, group.body_ids == 4] == 0 and scales[1].min() == 1

  pert.pert.select = 0
  scene.update_from_mjdata(data)
  assert scene._highlight is None
  assert np.asarray(group.handle.batched_scales).min() == 1
  server.stop()
