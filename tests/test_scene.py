import mujoco
import numpy as np
import viser

from mjviser import ViserMujocoScene
from mjviser.conversions import is_fixed_body

# Construction--------------------------------------------------------


def test_scene_creates_mesh_handles(scene, simple_model):
  # box_body has 3 geoms (box, sphere, capsule), all non-fixed.
  assert len(scene._mesh_groups) > 0
  for mg in scene._mesh_groups:
    for body_id in mg.body_ids:
      assert not is_fixed_body(simple_model, body_id)


def test_scene_no_handles_for_fixed_bodies(scene, simple_model):
  for mg in scene._mesh_groups:
    for body_id in mg.body_ids:
      assert body_id != 0, "World body should not have batched handles"


def test_scene_fixed_bodies_frame_exists(scene):
  assert scene.fixed_bodies_frame is not None


def test_scene_tracked_body(scene, simple_model):
  assert scene._tracked_body_id is not None
  assert not is_fixed_body(simple_model, scene._tracked_body_id)


def test_scene_default_visibility(scene):
  assert scene.geom_groups_visible[:3] == [True, True, True]
  assert scene.geom_groups_visible[3:] == [False, False, False]


# Batched transform---------------------------------------------------


def test_batched_transform_no_tile(scene):
  positions = np.random.randn(1, 3, 3).astype(np.float32)
  quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (1, 3, 1))
  offset = np.zeros(3)

  pos, quat = scene._batched_transform(positions, quats, 1, 0, offset, False)
  np.testing.assert_array_equal(pos, positions[..., 1, :])


def test_batched_transform_slice_single():
  server = viser.ViserServer(port=0)
  xml = """
  <mujoco>
    <worldbody>
      <body name="a"><joint type="free"/><geom size="0.1"/></body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  scene = ViserMujocoScene(server, model, num_envs=4)

  positions = np.random.randn(4, 2, 3).astype(np.float32)
  quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (4, 2, 1))
  offset = np.zeros(3)

  pos, quat = scene._batched_transform(positions, quats, 1, 0, offset, True)
  assert pos.shape == (1, 3)
  # Should contain only the selected env's data.
  np.testing.assert_array_equal(pos[0], positions[0, 1])

  try:
    server.stop()
  except RuntimeError:
    pass


# Update from mjdata--------------------------------------------------


def test_update_from_mjdata_caches_state(scene, simple_model):
  data = mujoco.MjData(simple_model)
  mujoco.mj_forward(simple_model, data)
  scene.update_from_mjdata(data)

  assert scene._last_body_xpos is not None
  assert scene._last_body_xmat is not None
  assert scene._last_body_xpos.shape[1] == simple_model.nbody


def test_update_from_mjdata_scene_offset(scene, simple_model):
  data = mujoco.MjData(simple_model)
  # Put box_body at a known position.
  data.qpos[:3] = [1.0, 2.0, 3.0]
  mujoco.mj_forward(simple_model, data)

  scene.camera_tracking_enabled = True
  scene.update_from_mjdata(data)

  # Offset should be the negative of the tracked body position.
  assert np.linalg.norm(scene._scene_offset) > 0


# Overlay properties -------------------------------------------------


def test_show_contact_points_property(scene):
  assert scene.show_contact_points is False
  scene.show_contact_points = True
  assert scene.show_contact_points is True
  assert bool(scene._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])


def test_show_contact_forces_property(scene):
  assert scene.show_contact_forces is False
  scene.show_contact_forces = True
  assert scene.show_contact_forces is True
  assert bool(scene._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])


def test_show_inertia_property(scene):
  assert scene.show_inertia is False
  scene.show_inertia = True
  assert scene.show_inertia is True
  assert bool(scene._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA])


def test_frame_mode_property(scene):
  assert scene.frame_mode == "None"
  scene.frame_mode = "Body"
  assert scene.frame_mode == "Body"
  assert scene._mjv_option.frame == int(mujoco.mjtFrame.mjFRAME_BODY)
  scene.frame_mode = "None"
  assert scene._mjv_option.frame == int(mujoco.mjtFrame.mjFRAME_NONE)


def test_any_decor_visible(scene):
  assert scene._any_decor_visible() is False
  scene.show_contact_points = True
  assert scene._any_decor_visible() is True
  scene.show_contact_points = False
  scene.frame_mode = "Geom"
  assert scene._any_decor_visible() is True
