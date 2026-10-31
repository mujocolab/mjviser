import mujoco

from mjviser.scene import _VIS_FLAG_COMMANDS


def test_vis_flag_commands_toggle_flag_and_checkbox(scene):
  scene.create_visualization_gui()
  scene.register_keybindings()
  for label, _, flag in _VIS_FLAG_COMMANDS:
    before = bool(scene._mjv_option.flags[flag])
    scene._toggle_vis_flag(flag)
    assert bool(scene._mjv_option.flags[flag]) is not before, label
    cb = scene._vis_flag_checkboxes.get(flag) or scene._simple_flag_checkboxes[flag]
    assert cb.value is not before, label


def test_cycle_camera_wraps_over_mujoco_cameras(humanoid_scene):
  scene = humanoid_scene
  scene.create_visualization_gui()
  names = [
    mujoco.mj_id2name(scene.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    for i in range(scene.mj_model.ncam)
  ]
  dropdown = scene._camera_dropdown
  assert dropdown is not None

  scene._cycle_camera(1)
  assert dropdown.value == names[0]
  assert scene.camera_tracking_enabled is False
  scene._cycle_camera(-1)
  assert dropdown.value == names[-1]
  scene._cycle_camera(1)
  assert dropdown.value == names[0]
