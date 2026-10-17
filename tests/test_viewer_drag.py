"""Tests for body drag interactions in Viewer."""

import mujoco
import numpy as np
import pytest
import viser

from mjviser import BodyDragEvent, Viewer

_DRAG_TEST_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="fixed_wall" pos="0 -1 0">
      <geom type="box" size="0.5 0.1 0.5"/>
    </body>
    <body name="box" pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size="0.5 0.5 0.5" mass="1.0"/>
    </body>
    <body name="mocap_target" mocap="true" pos="2 0 1">
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def viewer():
  server = viser.ViserServer(port=0)
  model = mujoco.MjModel.from_xml_string(_DRAG_TEST_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  v = Viewer(model, data, server=server)
  yield v
  try:
    server.stop()
  except RuntimeError:
    pass


def _make_event(
  phase: str,
  body_id: int,
  body_name: str,
  start_pos: tuple[float, float, float],
  end_pos: tuple[float, float, float],
  mocap_id: int | None = None,
  modifier: str | None = "cmd/ctrl",
) -> BodyDragEvent:
  return BodyDragEvent(
    client=None,  # type: ignore[arg-type]
    phase=phase,  # type: ignore[arg-type]
    body_id=body_id,
    body_name=body_name,
    env_idx=0,
    mocap_id=mocap_id,
    start_position=start_pos,
    end_position=end_pos,
    start_screen_pos=(0.5, 0.5),
    end_screen_pos=(0.6, 0.6),
    button="left",
    modifier=modifier,  # type: ignore[arg-type]
  )


def test_viewer_translate_drag_applies_force_and_torque(viewer: Viewer) -> None:
  box_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, "box")
  viewer.perturbation_stiffness = 100.0

  # Grab off-center at (0.5, 0.0, 1.0) on the box centered at (0, 0, 1).
  start_ev = _make_event(
    "start", box_id, "box", (0.5, 0.0, 1.0), (0.5, 0.0, 1.0), modifier="cmd/ctrl"
  )
  viewer._on_body_drag(start_ev)
  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id], np.zeros(6))

  # Drag target to (0.5, 1.0, 1.0) -> delta = (0, 1, 0), F = (0, 100, 0)
  # Lever arm r = (0.5, 0, 0), torque = r x F = (0, 0, 50)
  update_ev = _make_event(
    "update", box_id, "box", (0.5, 0.0, 1.0), (0.5, 1.0, 1.0), modifier="cmd/ctrl"
  )
  viewer._on_body_drag(update_ev)

  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id, :3], [0.0, 100.0, 0.0])
  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id, 3:], [0.0, 0.0, 50.0])
  assert viewer._perturb_line_handle is not None
  assert viewer._perturb_line_handle.visible

  # End drag resets xfrc_applied and hides visual line.
  end_ev = _make_event(
    "end", box_id, "box", (0.5, 0.0, 1.0), (0.5, 1.0, 1.0), modifier="cmd/ctrl"
  )
  viewer._on_body_drag(end_ev)
  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id], np.zeros(6))
  assert not viewer._perturb_line_handle.visible


def test_viewer_rotate_drag_applies_pin_and_torque(viewer: Viewer) -> None:
  box_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, "box")
  viewer.perturbation_stiffness = 100.0
  viewer.perturbation_torque_scale = 0.5

  start_ev = _make_event(
    "start",
    box_id,
    "box",
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    modifier="cmd/ctrl+shift",
  )
  viewer._on_body_drag(start_ev)

  update_ev = _make_event(
    "update",
    box_id,
    "box",
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 2.0),
    modifier="cmd/ctrl+shift",
  )
  viewer._on_body_drag(update_ev)

  # Unit cube (half-size 0.5, mass 1.0) has inertia 1/6; mjv_applyPerturbForce
  # applies 250 * I * 0.5 = 41.6667 N*m around +Z while pinning linear force at 0.
  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id, :3], [0.0, 0.0, 0.0])
  np.testing.assert_allclose(
    viewer.data.xfrc_applied[box_id, 3:], [0.0, 0.0, 250.0 / 6.0], rtol=1e-4
  )


def test_viewer_damping_and_reset_mid_drag(viewer: Viewer) -> None:
  box_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, "box")
  viewer.perturbation_stiffness = 100.0

  # Give the box +1 m/s velocity along Y; mjv_applyPerturbForce damps the 100 N pull to 90 N.
  viewer.data.qvel[1] = 1.0
  mujoco.mj_forward(viewer.model, viewer.data)

  viewer._on_body_drag(
    _make_event("start", box_id, "box", (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
  )
  viewer._on_body_drag(
    _make_event("update", box_id, "box", (0.0, 0.0, 1.0), (0.0, 1.0, 1.0))
  )
  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id, :3], [0.0, 90.0, 0.0])

  # Reset mid-drag clears xfrc_applied and active perturbation state.
  viewer._reset()
  assert viewer._drag_body_id is None
  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id], np.zeros(6))


def test_viewer_paused_free_body_drag_updates_qpos(viewer: Viewer) -> None:
  box_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, "box")
  viewer._paused = True

  viewer._on_body_drag(
    _make_event("start", box_id, "box", (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
  )
  viewer._on_body_drag(
    _make_event("update", box_id, "box", (0.0, 0.0, 1.0), (1.0, 2.0, 3.0))
  )
  np.testing.assert_allclose(viewer.data.qpos[:3], [1.0, 2.0, 3.0])


def test_viewer_mocap_body_drag_updates_mocap_pos_and_quat(viewer: Viewer) -> None:
  mocap_body_id = mujoco.mj_name2id(
    viewer.model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target"
  )
  mocap_id = int(viewer.model.body_mocapid[mocap_body_id])

  # Translate mocap body.
  viewer._on_body_drag(
    _make_event(
      "start",
      mocap_body_id,
      "mocap_target",
      (2.0, 0.0, 1.0),
      (2.0, 0.0, 1.0),
      mocap_id=mocap_id,
      modifier="cmd/ctrl",
    )
  )
  viewer._on_body_drag(
    _make_event(
      "update",
      mocap_body_id,
      "mocap_target",
      (2.0, 0.0, 1.0),
      (3.5, -1.0, 2.0),
      mocap_id=mocap_id,
      modifier="cmd/ctrl",
    )
  )
  np.testing.assert_allclose(viewer.data.mocap_pos[mocap_id], [3.5, -1.0, 2.0])

  viewer._on_body_drag(
    _make_event(
      "end",
      mocap_body_id,
      "mocap_target",
      (2.0, 0.0, 1.0),
      (3.5, -1.0, 2.0),
      mocap_id=mocap_id,
      modifier="cmd/ctrl",
    )
  )

  # Rotate mocap body.
  init_quat = viewer.data.mocap_quat[mocap_id].copy()
  viewer._on_body_drag(
    _make_event(
      "start",
      mocap_body_id,
      "mocap_target",
      (3.5, -1.0, 2.0),
      (3.5, -1.0, 2.0),
      mocap_id=mocap_id,
      modifier="cmd/ctrl+shift",
    )
  )
  viewer._on_body_drag(
    _make_event(
      "update",
      mocap_body_id,
      "mocap_target",
      (3.5, -1.0, 2.0),
      (3.5, -1.0, 2.5),
      mocap_id=mocap_id,
      modifier="cmd/ctrl+shift",
    )
  )
  assert not np.allclose(viewer.data.mocap_quat[mocap_id], init_quat)


def test_viewer_ignores_fixed_body_drag(viewer: Viewer) -> None:
  wall_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, "fixed_wall")
  viewer._on_body_drag(
    _make_event("start", wall_id, "fixed_wall", (0.0, -1.0, 0.0), (0.0, -1.0, 0.0))
  )
  assert viewer._drag_body_id is None


def test_viewer_custom_map_stiffness_and_paused_step(viewer: Viewer) -> None:
  box_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, "box")
  # Even if the model specifies <map stiffness="400"/>, perturbation_stiffness=100
  # still produces 100 N/m per kg of subtree mass.
  viewer.model.vis.map.stiffness = 400.0
  viewer.perturbation_stiffness = 100.0

  viewer._on_body_drag(
    _make_event("start", box_id, "box", (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
  )
  viewer._on_body_drag(
    _make_event("update", box_id, "box", (0.0, 0.0, 1.0), (0.0, 1.0, 1.0))
  )
  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id, :3], [0.0, 100.0, 0.0])

  # While paused, the perturbation force line is hidden, and single-step applies force.
  viewer._paused = True
  viewer._update_perturb_visual()
  assert viewer._perturb_line_handle is not None
  assert not viewer._perturb_line_handle.visible

  viewer.data.xfrc_applied[box_id] = 0.0
  viewer._apply_drag_perturbation(apply_force=True)
  np.testing.assert_allclose(viewer.data.xfrc_applied[box_id, :3], [0.0, 100.0, 0.0])
