import mujoco
import numpy as np
import viser

from mjviser import ViserMujocoScene
from mjviser.conversions import is_fixed_body

_MESH_BODY_XML = """
<mujoco>
  <asset>
    <mesh name="cube"
          vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                  -1 -1 1   1 -1 1   1 1 1   -1 1 1"
          face="0 3 2  0 2 1  4 5 6  4 6 7
                0 1 5  0 5 4  2 3 7  2 7 6
                0 4 7  0 7 3  1 2 6  1 6 5"/>
  </asset>
  <worldbody>
    <body name="mesh_body" pos="0 0 1">
      <joint type="free"/>
      <geom type="mesh" mesh="cube"/>
    </body>
  </worldbody>
</mujoco>
"""

_FIXED_MESH_BODY_XML = """
<mujoco>
  <asset>
    <mesh name="cube"
          vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                  -1 -1 1   1 -1 1   1 1 1   -1 1 1"
          face="0 3 2  0 2 1  4 5 6  4 6 7
                0 1 5  0 5 4  2 3 7  2 7 6
                0 4 7  0 7 3  1 2 6  1 6 5"/>
  </asset>
  <worldbody>
    <body name="fixed_mesh_body" pos="0 0 1">
      <geom type="mesh" mesh="cube"/>
    </body>
  </worldbody>
</mujoco>
"""

_ACTUATED_HINGE_XML = """
<mujoco>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

_AUTO_CONNECT_XML = """
<mujoco>
  <worldbody>
    <body name="parent" pos="0 0 1.0">
      <geom type="sphere" size="0.1"/>
      <body name="child" pos="0 0 2.0">
        <geom type="sphere" size="0.08"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

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


def test_nested_fixed_body_world_position():
  # A nested fixed body whose parent has a non-zero position: the mesh handle
  # must be placed at the world position (1.5, 0, 0), not the local offset (0.5, 0, 0).
  xml = """
  <mujoco>
    <worldbody>
      <body name="parent" pos="1 0 0">
        <body name="child" pos="0.5 0 0">
          <geom type="box" size=".1 .1 .1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  server = viser.ViserServer(port=0)
  try:
    scene = ViserMujocoScene(server, model, num_envs=1)
    child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "child")
    handle = scene._fixed_geom_handles[(child_id, 0, 0)]
    np.testing.assert_allclose(handle.position, [1.5, 0.0, 0.0], atol=1e-6)
  finally:
    server.stop()


def test_scene_tracked_body(scene, simple_model):
  assert scene._tracked_body_id is not None
  assert not is_fixed_body(simple_model, scene._tracked_body_id)


def test_scene_default_visibility(scene):
  assert scene.geom_groups_visible[:3] == [True, True, True]
  assert scene.geom_groups_visible[3:] == [False, False, False]


def test_rebuild_visual_handles_recreates_baked_mesh_handles(scene, simple_model):
  old_handles = [group.handle for group in scene._mesh_groups]
  assert old_handles

  box_id = mujoco.mj_name2id(simple_model, mujoco.mjtObj.mjOBJ_GEOM, "box")
  simple_model.geom_rgba[box_id] = [0.0, 0.0, 1.0, 1.0]

  scene.rebuild_visual_handles()

  new_handles = [group.handle for group in scene._mesh_groups]
  assert new_handles
  old_handle_ids = {id(handle) for handle in old_handles}
  assert all(id(handle) not in old_handle_ids for handle in new_handles)


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


def test_show_actuators_property(scene):
  assert scene.show_actuators is False
  scene.show_actuators = True
  assert scene.show_actuators is True
  assert bool(scene._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR])


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


def test_convex_hull_handles_follow_visibility():
  server = viser.ViserServer(port=0)
  model = mujoco.MjModel.from_xml_string(_MESH_BODY_XML)
  scene = ViserMujocoScene(server, model, num_envs=1)

  assert 1 in scene._hull_body_meshes
  assert len(scene._hull_dynamic_handles) == 1

  hull_handle, body_id = scene._hull_dynamic_handles[0]
  assert body_id == 1
  assert hull_handle.visible is False
  assert scene._mesh_groups[0].handle.visible is True

  scene.show_convex_hull = True
  assert hull_handle.visible is True

  scene._hull_hide_meshes = True
  scene._sync_visibilities()
  assert scene._mesh_groups[0].handle.visible is False

  scene.show_convex_hull = False
  assert hull_handle.visible is False
  assert scene._mesh_groups[0].handle.visible is True

  try:
    server.stop()
  except RuntimeError:
    pass


def test_convex_hull_fixed_body_visibility():
  server = viser.ViserServer(port=0)
  model = mujoco.MjModel.from_xml_string(_FIXED_MESH_BODY_XML)
  scene = ViserMujocoScene(server, model, num_envs=1)

  assert 1 in scene._hull_fixed_handles
  hull_handle = scene._hull_fixed_handles[1]
  fixed_geom_handle = next(iter(scene._fixed_geom_handles.values()))

  assert hull_handle.visible is False
  assert fixed_geom_handle.visible is True

  scene.show_convex_hull = True
  scene._hull_hide_meshes = True
  scene._sync_visibilities()

  assert hull_handle.visible is True
  assert fixed_geom_handle.visible is False

  scene.show_convex_hull = False
  assert hull_handle.visible is False
  assert fixed_geom_handle.visible is True

  try:
    server.stop()
  except RuntimeError:
    pass


def test_convex_hull_dynamic_handle_updates_with_mjdata():
  server = viser.ViserServer(port=0)
  model = mujoco.MjModel.from_xml_string(_MESH_BODY_XML)
  data = mujoco.MjData(model)
  data.qpos[:3] = [0.5, -0.25, 1.5]
  mujoco.mj_forward(model, data)

  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  scene.show_convex_hull = True
  scene.update_from_mjdata(data)

  hull_handle, body_id = scene._hull_dynamic_handles[0]
  np.testing.assert_allclose(hull_handle.batched_positions[0], data.xpos[body_id])

  try:
    server.stop()
  except RuntimeError:
    pass


def test_decor_handles_reused_when_joint_and_actuator_counts_change():
  server = viser.ViserServer(port=0)
  model = mujoco.MjModel.from_xml_string(_ACTUATED_HINGE_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  scene = ViserMujocoScene(server, model, num_envs=1)

  scene._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
  scene.update_from_mjdata(data)

  handles_before = scene._decor_handles.copy()
  counts_before = {
    key: len(handle.batched_positions) for key, handle in handles_before.items()
  }
  assert handles_before

  scene._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1
  scene.update_from_mjdata(data)

  shared_keys = handles_before.keys() & scene._decor_handles.keys()
  assert shared_keys
  assert any(
    len(scene._decor_handles[key].batched_positions) != counts_before[key]
    for key in shared_keys
  )
  for key in shared_keys:
    assert scene._decor_handles[key] is handles_before[key]

  try:
    server.stop()
  except RuntimeError:
    pass


def test_request_update_uses_refresh_handler(scene):
  calls: list[str] = []
  scene.set_refresh_handler(lambda: calls.append("refresh"))

  scene.request_update()

  assert calls == ["refresh"]


def test_autoconnect_capsules_match_mujoco_connector_sizes():
  server = viser.ViserServer(port=0)
  model = mujoco.MjModel.from_xml_string(_AUTO_CONNECT_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  scene = ViserMujocoScene(server, model, num_envs=1)

  scene._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] = 1
  scene.update_from_mjdata(data)

  capsule_key = (int(mujoco.mjtGeom.mjGEOM_CAPSULE), False)
  assert capsule_key in scene._decor_handles
  assert all(handle.visible for handle in scene._fixed_geom_handles.values())

  capsule_handle = scene._decor_handles[capsule_key]
  assert len(capsule_handle.batched_positions) == 1
  assert capsule_handle.batched_scales is not None

  connector_geom = None
  for geom_idx in range(scene._mjv_scene.ngeom):
    geom = scene._mjv_scene.geoms[geom_idx]
    if int(geom.type) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
      connector_geom = geom
      break

  if connector_geom is None:
    raise AssertionError("Expected an auto-connect capsule geom in mjvScene")
  half_total = float(connector_geom.size[2])
  rendered_length = float(capsule_handle.batched_scales[0, 2])

  np.testing.assert_allclose(rendered_length, 2.0 * half_total)

  scene._autoconnect_hide_meshes = True
  scene._sync_visibilities()
  assert not any(handle.visible for handle in scene._fixed_geom_handles.values())

  try:
    server.stop()
  except RuntimeError:
    pass
