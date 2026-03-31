"""Manages all Viser visualization handles and state for MuJoCo models."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable
from threading import RLock

import mujoco
import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj

from .conversions import (
  get_body_name,
  group_geoms_by_visual_compat,
  is_fixed_body,
  merge_geoms,
  merge_geoms_hull,
  merge_sites,
)


@dataclasses.dataclass
class _MeshGroup:
  """A group of bodies sharing identical geometry, rendered as one batched handle."""

  handle: viser.BatchedGlbHandle
  body_ids: np.ndarray  # (N,) int32 array of body IDs
  group_id: int
  mocap_ids: np.ndarray | None  # (N,) int32 mocap IDs, or None if non-mocap


# Viser visualization defaults.
_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150
_DEFAULT_ENVIRONMENT_INTENSITY = 0.8
_DEFAULT_CONTACT_POINT_COLOR = (230, 153, 51)
_DEFAULT_CONTACT_FORCE_COLOR = (255, 0, 0)

# Map from frame mode name to mjtFrame enum value.
_FRAME_MODES: dict[str, int] = {
  "None": int(mujoco.mjtFrame.mjFRAME_NONE),
  "Body": int(mujoco.mjtFrame.mjFRAME_BODY),
  "Geom": int(mujoco.mjtFrame.mjFRAME_GEOM),
  "Site": int(mujoco.mjtFrame.mjFRAME_SITE),
}

# Geom type constants (avoid repeated int() casts).
_CYLINDER = int(mjtGeom.mjGEOM_CYLINDER)
_CAPSULE = int(mjtGeom.mjGEOM_CAPSULE)
_BOX = int(mjtGeom.mjGEOM_BOX)
_ELLIPSOID = int(mjtGeom.mjGEOM_ELLIPSOID)
_SPHERE = int(mjtGeom.mjGEOM_SPHERE)
_ARROW = int(mjtGeom.mjGEOM_ARROW)
_ARROW1 = int(mjtGeom.mjGEOM_ARROW1)
_ARROW2 = int(mjtGeom.mjGEOM_ARROW2)
_ARROWS = (_ARROW, _ARROW1, _ARROW2)
_ARROW_HEAD = -1  # Synthetic geom type for arrow cone heads.

_CAT_DECOR = int(mujoco.mjtCatBit.mjCAT_DECOR)
_OBJ_TENDON = int(mujoco.mjtObj.mjOBJ_TENDON)
_OBJ_JOINT = int(mujoco.mjtObj.mjOBJ_JOINT)

# Cached unit meshes for decor rendering, keyed by mjtGeom int value.
_UNIT_MESHES: dict[int, trimesh.Trimesh] = {}


def _get_unit_mesh(geom_type: int) -> trimesh.Trimesh:
  """Return a cached unit mesh for the given mjvGeom type."""
  if geom_type not in _UNIT_MESHES:
    if geom_type == _CYLINDER:
      _UNIT_MESHES[geom_type] = trimesh.creation.cylinder(radius=1.0, height=1.0)
    elif geom_type == _CAPSULE:
      _UNIT_MESHES[geom_type] = trimesh.creation.capsule(radius=1.0, height=1.0)
    elif geom_type == _BOX:
      _UNIT_MESHES[geom_type] = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    elif geom_type in (_ELLIPSOID, _SPHERE):
      _UNIT_MESHES[geom_type] = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    elif geom_type in _ARROWS:
      # Arrow shaft: cylinder centered at origin, height along Z.
      shaft = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=12)
      shaft.apply_translation([0, 0, 0.5])
      _UNIT_MESHES[geom_type] = shaft
    elif geom_type == _ARROW_HEAD:
      _UNIT_MESHES[geom_type] = trimesh.creation.cone(
        radius=2.0, height=1.0, sections=12
      )
    else:
      _UNIT_MESHES[geom_type] = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
  return _UNIT_MESHES[geom_type]


class ViserMujocoScene:
  """Manages Viser scene handles and visualization state for MuJoCo models.

  This class handles geometry creation, batched rendering of bodies and
  sites across multiple environments, contact visualization, and GUI
  controls. Users build custom overlays on top using the ``server``
  attribute and viser's native API.
  """

  def __init__(
    self,
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    num_envs: int,
  ) -> None:
    """Create and populate a scene with geometry.

    Args:
        server: Viser server instance.
        mj_model: MuJoCo model.
        num_envs: Number of parallel environments.
    """
    self.server = server
    self.mj_model = mj_model
    self.mj_data = mujoco.MjData(mj_model)
    self.num_envs = num_envs

    # mjvScene infrastructure for decor visualization.
    self._mjv_scene = mujoco.MjvScene(mj_model, maxgeom=max(2000, mj_model.ngeom * 2))
    self._mjv_option = mujoco.MjvOption()
    self._mjv_camera = mujoco.MjvCamera()
    # Disable all vis flags; properties below enable them on demand.
    self._mjv_option.flags[:] = 0
    self._mjv_option.frame = int(mujoco.mjtFrame.mjFRAME_NONE)

    # Handles.
    self._mesh_groups: list[_MeshGroup] = []
    self.site_handles_by_group: dict[tuple[int, int], viser.BatchedGlbHandle] = {}
    self._decor_handles: dict[tuple[int, bool], viser.BatchedMeshHandle] = {}
    self._fixed_geom_handles: dict[tuple[int, int, int], viser.GlbHandle] = {}
    self._fixed_site_handles: dict[tuple[int, int], viser.GlbHandle] = {}
    self._update_lock = RLock()
    self._refresh_handler: Callable[[], None] | None = None
    self._hull_body_meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    self._hull_fixed_handles: dict[int, viser.BatchedMeshHandle] = {}
    self._hull_dynamic_handles: list[tuple[viser.BatchedMeshHandle, int]] = []
    self._hull_color: tuple[int, int, int] = (230, 230, 255)
    self._hull_opacity: float = 0.5
    self._show_convex_hull: bool = False
    self._hull_hide_meshes: bool = False

    # Visualization settings.
    self.env_idx = 0
    self.camera_tracking_enabled = True
    self.show_only_selected = False
    self.geom_groups_visible = [True, True, True, False, False, False]
    self.site_groups_visible = [True, True, True, False, False, False]
    self.meansize_override: float | None = None

    # Set default colors for decor elements (written to model.vis.rgba).
    mj_model.vis.rgba.contactpoint[:] = [
      _DEFAULT_CONTACT_POINT_COLOR[0] / 255,
      _DEFAULT_CONTACT_POINT_COLOR[1] / 255,
      _DEFAULT_CONTACT_POINT_COLOR[2] / 255,
      0.8,
    ]
    mj_model.vis.rgba.contactforce[:] = [
      _DEFAULT_CONTACT_FORCE_COLOR[0] / 255,
      _DEFAULT_CONTACT_FORCE_COLOR[1] / 255,
      _DEFAULT_CONTACT_FORCE_COLOR[2] / 255,
      0.8,
    ]
    mj_model.vis.rgba.inertia[:] = [1.0, 0.7, 0.24, 0.5]
    self.needs_update = False
    self.paused = False

    # Cached visualization state for re-rendering when settings change.
    self._tracked_body_id: int | None = None
    self._last_body_xpos: np.ndarray | None = None
    self._last_body_xmat: np.ndarray | None = None
    self._last_mocap_pos: np.ndarray | None = None
    self._last_mocap_quat: np.ndarray | None = None
    self._last_env_idx = 0
    self._last_mj_data: mujoco.MjData | None = None
    self._scene_offset = np.zeros(3)

    # Build the scene.
    server.scene.configure_environment_map(
      environment_intensity=_DEFAULT_ENVIRONMENT_INTENSITY
    )
    self.fixed_bodies_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)
    self._add_fixed_geometry()
    self._create_mesh_handles_by_group()
    self._add_fixed_sites()
    self._create_site_handles_by_group()
    self._compute_hull_body_meshes()
    self._build_hull_handles()

    for body_id in range(mj_model.nbody):
      if not is_fixed_body(mj_model, body_id):
        self._tracked_body_id = body_id
        break

  # -- Overlay visibility properties (backed by mjvOption flags) -----------

  @property
  def show_contact_points(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])

  @show_contact_points.setter
  def show_contact_points(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(value)

  @property
  def show_contact_forces(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])

  @show_contact_forces.setter
  def show_contact_forces(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(value)

  @property
  def show_tendons(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON])

  @show_tendons.setter
  def show_tendons(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = int(value)

  @property
  def show_inertia(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA])

  @show_inertia.setter
  def show_inertia(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = int(value)

  @property
  def show_actuators(self) -> bool:
    return bool(self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR])

  @show_actuators.setter
  def show_actuators(self, value: bool) -> None:
    self._mjv_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = int(value)

  @property
  def show_convex_hull(self) -> bool:
    return self._show_convex_hull

  @show_convex_hull.setter
  def show_convex_hull(self, value: bool) -> None:
    with self._update_lock:
      self._show_convex_hull = value
      for handle in self._hull_fixed_handles.values():
        handle.visible = value
      for handle, _ in self._hull_dynamic_handles:
        handle.visible = value
      self._sync_visibilities()

  def set_refresh_handler(self, handler: Callable[[], None] | None) -> None:
    """Install a callback used to refresh cached visualization state."""
    with self._update_lock:
      self._refresh_handler = handler

  def _apply_visualization_change(self, mutator: Callable[[], None]) -> None:
    """Apply a GUI-side visualization change and refresh from cached state."""
    refresh_handler: Callable[[], None] | None = None
    with self._update_lock:
      mutator()
      self.needs_update = True
      refresh_handler = self._refresh_handler
      if refresh_handler is None:
        self._refresh_visualization_locked()
        return
    refresh_handler()

  @property
  def frame_mode(self) -> str:
    current = int(self._mjv_option.frame)
    for name, val in _FRAME_MODES.items():
      if val == current:
        return name
    return "None"

  @frame_mode.setter
  def frame_mode(self, value: str) -> None:
    self._mjv_option.frame = _FRAME_MODES.get(value, 0)

  def _any_decor_visible(self) -> bool:
    """Return True if any overlay visualization is enabled."""
    if self.frame_mode != "None":
      return True
    return any(self._mjv_option.flags[i] for i in range(len(self._mjv_option.flags)))

  def _sync_visibilities(self) -> None:
    """Synchronize all handle visibilities based on current flags."""
    hidden_bodies: set[int] = set()
    if self._show_convex_hull and self._hull_hide_meshes:
      hidden_bodies = set(self._hull_body_meshes.keys())

    for mg in self._mesh_groups:
      visible = mg.group_id < 6 and self.geom_groups_visible[mg.group_id]
      if visible and any(body_id in hidden_bodies for body_id in mg.body_ids):
        visible = False
      mg.handle.visible = visible

    for (body_id, group_id, _), handle in self._fixed_geom_handles.items():
      visible = group_id < 6 and self.geom_groups_visible[group_id]
      if visible and body_id in hidden_bodies:
        visible = False
      handle.visible = visible

    for (_, group_id), handle in self.site_handles_by_group.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    for (_, group_id), handle in self._fixed_site_handles.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    if not self._any_decor_visible():
      for handle in self._decor_handles.values():
        handle.visible = False

  def create_scene_gui(
    self,
    camera_distance: float = -1.0,
    camera_azimuth: float = 120.0,
    camera_elevation: float = 20.0,
  ) -> None:
    """Add camera and environment controls into the current GUI context.

    Args:
        camera_distance: Default camera distance. If negative, derived
            from model extent.
        camera_azimuth: Default camera azimuth angle in degrees.
        camera_elevation: Default camera elevation angle in degrees.
    """
    if self.num_envs > 1:
      with self.server.gui.add_folder("Environment"):
        env_slider = self.server.gui.add_slider(
          "Select",
          min=0,
          max=self.num_envs - 1,
          step=1,
          initial_value=self.env_idx,
          hint=f"Select environment (0-{self.num_envs - 1})",
        )

        @env_slider.on_update
        def _(_) -> None:
          self._apply_visualization_change(
            lambda: setattr(self, "env_idx", int(env_slider.value))
          )

        show_only_cb = self.server.gui.add_checkbox(
          "Hide others",
          initial_value=self.show_only_selected,
          hint="Show only the selected environment.",
        )

        @show_only_cb.on_update
        def _(_) -> None:
          self._apply_visualization_change(
            lambda: setattr(self, "show_only_selected", show_only_cb.value)
          )

    _center = self.mj_model.stat.center.copy()
    _extent = self.mj_model.stat.extent
    if camera_distance <= 0:
      camera_distance = 3.0 * _extent
    _az_rad = np.deg2rad(camera_azimuth)
    _el_rad = np.deg2rad(camera_elevation)
    _camera_offset = (
      np.array(
        [
          -np.cos(_el_rad) * np.cos(_az_rad),
          -np.cos(_el_rad) * np.sin(_az_rad),
          np.sin(_el_rad),
        ]
      )
      * camera_distance
    )

    with self.server.gui.add_folder("Camera"):
      cb_camera_tracking = self.server.gui.add_checkbox(
        "Track camera",
        initial_value=self.camera_tracking_enabled,
        hint="Keep tracked body centered.",
      )

      @cb_camera_tracking.on_update
      def _(_) -> None:
        def _mutate() -> None:
          self.camera_tracking_enabled = cb_camera_tracking.value
          if self.camera_tracking_enabled:
            for client in self.server.get_clients().values():
              client.camera.position = _camera_offset
              client.camera.look_at = np.zeros(3)

        self._apply_visualization_change(_mutate)

      slider_fov = self.server.gui.add_slider(
        "FOV (\u00b0)",
        min=_DEFAULT_FOV_MIN,
        max=_DEFAULT_FOV_MAX,
        step=1,
        initial_value=_DEFAULT_FOV_DEGREES,
        hint="Vertical FOV in degrees.",
      )

      @slider_fov.on_update
      def _(_) -> None:
        for client in self.server.get_clients().values():
          client.camera.fov = np.radians(slider_fov.value)

      @self.server.on_client_connect
      def _(client: viser.ClientHandle) -> None:
        client.camera.fov = np.radians(slider_fov.value)
        client.camera.position = _camera_offset
        client.camera.look_at = np.zeros(3) if self.camera_tracking_enabled else _center

  def create_overlay_gui(self) -> None:
    """Add overlay visualization controls into the current GUI context.

    Controls are grouped by feature so that toggle, color, opacity,
    and scale live together for each overlay type.
    """
    _rgba = self.mj_model.vis.rgba
    _vis = self.mj_model.vis.scale
    _map = self.mj_model.vis.map
    _stat = self.mj_model.stat
    _opt = self._mjv_option

    def _vis_flag_cb(
      flag_idx: int,
      initial: bool = False,
      extra_on_update: Callable[[], None] | None = None,
    ) -> None:
      """Add a vis-flag checkbox at the current GUI level."""
      _opt.flags[flag_idx] = int(initial)
      cb = self.server.gui.add_checkbox("Enabled", initial_value=initial)

      @cb.on_update
      def _(event, _idx=flag_idx) -> None:
        def _mutate() -> None:
          _opt.flags[_idx] = int(event.target.value)
          if extra_on_update is not None:
            extra_on_update()

        self._apply_visualization_change(_mutate)

    def _color_controls(rgba_attr: str) -> None:
      """Add color picker + opacity slider for a model.vis.rgba field."""
      rgba_arr = getattr(_rgba, rgba_attr)
      rgb_init = (
        int(rgba_arr[0] * 255),
        int(rgba_arr[1] * 255),
        int(rgba_arr[2] * 255),
      )
      cp = self.server.gui.add_rgb("Color", initial_value=rgb_init)
      op = self.server.gui.add_slider(
        "Opacity",
        min=0.0,
        max=1.0,
        step=0.05,
        initial_value=float(rgba_arr[3]),
      )

      def _on_update(_, _a=rgba_attr, _cp=cp, _op=op) -> None:
        def _mutate() -> None:
          arr = getattr(_rgba, _a)
          r, g, b = _cp.value
          arr[:] = [r / 255, g / 255, b / 255, _op.value]

        self._apply_visualization_change(_mutate)

      cp.on_update(_on_update)
      op.on_update(_on_update)

    def _scale_slider(
      label: str,
      obj: object,
      attr: str,
      lo: float,
      hi: float,
      step: float,
    ) -> None:
      """Add a scale slider that writes to a model field."""
      lo_r, hi_r = round(lo, 6), round(hi, 6)
      step_r = round(step, 6) or round((hi - lo) / 200, 6)
      val = max(lo_r, min(hi_r, round(float(getattr(obj, attr)), 6)))
      sl = self.server.gui.add_slider(
        label,
        min=lo_r,
        max=hi_r,
        step=step_r,
        initial_value=val,
      )

      @sl.on_update
      def _(_, _obj=obj, _attr=attr, _sl=sl) -> None:
        self._apply_visualization_change(lambda: setattr(_obj, _attr, _sl.value))

    # -- Frames -----------------------------------------------------------

    frame_dropdown = self.server.gui.add_dropdown(
      "Frames", options=["None", "Body", "Geom", "Site"]
    )

    @frame_dropdown.on_update
    def _(_) -> None:
      self._apply_visualization_change(
        lambda: setattr(self, "frame_mode", frame_dropdown.value)
      )

    _scale_slider("Frame length", _vis, "framelength", 0.1, 5.0, 0.05)
    _scale_slider("Frame width", _vis, "framewidth", 0.01, 1.0, 0.01)

    # -- Contact points ---------------------------------------------------

    with self.server.gui.add_folder("Contact points"):
      _vis_flag_cb(int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT))
      _color_controls("contactpoint")
      _scale_slider("Width", _vis, "contactwidth", 0.01, 2.0, 0.01)
      _scale_slider("Height", _vis, "contactheight", 0.01, 1.0, 0.01)

    # -- Contact forces ---------------------------------------------------

    with self.server.gui.add_folder("Contact forces"):
      _vis_flag_cb(int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE))
      _color_controls("contactforce")
      _scale_slider("Width", _vis, "forcewidth", 0.01, 1.0, 0.01)
      _scale_slider("Force scale", _map, "force", 0.001, 0.1, 0.001)

      cb_split = self.server.gui.add_checkbox(
        "Split normal/friction", initial_value=False
      )

      @cb_split.on_update
      def _(event) -> None:
        self._apply_visualization_change(
          lambda: _opt.flags.__setitem__(
            mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT, int(event.target.value)
          )
        )

    # -- Inertia ----------------------------------------------------------

    with self.server.gui.add_folder("Inertia"):
      _vis_flag_cb(int(mujoco.mjtVisFlag.mjVIS_INERTIA))

      inertia_shape = self.server.gui.add_dropdown(
        "Shape",
        options=["Box", "Ellipsoid"],
        initial_value=(
          "Ellipsoid" if self.mj_model.vis.global_.ellipsoidinertia else "Box"
        ),
      )

      @inertia_shape.on_update
      def _(_) -> None:
        self._apply_visualization_change(
          lambda: setattr(
            self.mj_model.vis.global_,
            "ellipsoidinertia",
            int(inertia_shape.value == "Ellipsoid"),
          )
        )

      _color_controls("inertia")

    # -- Joints -----------------------------------------------------------

    with self.server.gui.add_folder("Joints"):
      _vis_flag_cb(int(mujoco.mjtVisFlag.mjVIS_JOINT))
      _color_controls("joint")
      _scale_slider("Length", _vis, "jointlength", 0.1, 5.0, 0.05)
      _scale_slider("Width", _vis, "jointwidth", 0.01, 1.0, 0.01)

    # -- Actuators -------------------------------------------------------

    with self.server.gui.add_folder("Actuators"):
      _vis_flag_cb(int(mujoco.mjtVisFlag.mjVIS_ACTUATOR))
      _color_controls("actuator")
      _scale_slider("Length", _vis, "actuatorlength", 0.1, 5.0, 0.05)
      _scale_slider("Width", _vis, "actuatorwidth", 0.01, 1.0, 0.01)

    # -- Convex hull ------------------------------------------------------

    with self.server.gui.add_folder("Convex hull"):
      hull_enabled = self.server.gui.add_checkbox("Enabled", initial_value=False)
      hull_hide_meshes = self.server.gui.add_checkbox(
        "Hide meshes", initial_value=False
      )
      hull_color = self.server.gui.add_rgb("Color", initial_value=self._hull_color)
      hull_opacity = self.server.gui.add_slider(
        "Opacity",
        min=0.0,
        max=1.0,
        step=0.05,
        initial_value=self._hull_opacity,
      )

      @hull_enabled.on_update
      def _(_) -> None:
        self._apply_visualization_change(
          lambda: setattr(self, "show_convex_hull", hull_enabled.value)
        )

      @hull_hide_meshes.on_update
      def _(_) -> None:
        def _mutate() -> None:
          self._hull_hide_meshes = hull_hide_meshes.value
          self._sync_visibilities()

        self._apply_visualization_change(_mutate)

      def _rebuild_hulls(_=None) -> None:
        def _mutate() -> None:
          self._hull_color = hull_color.value
          self._hull_opacity = hull_opacity.value
          self._clear_hull_handles()
          self._build_hull_handles()
          self.show_convex_hull = hull_enabled.value

        self._apply_visualization_change(_mutate)

      hull_color.on_update(_rebuild_hulls)
      hull_opacity.on_update(_rebuild_hulls)

    # -- Simple toggles ---------------------------------------------------

    _tendon_flag = int(mujoco.mjtVisFlag.mjVIS_TENDON)
    _opt.flags[_tendon_flag] = int(self.mj_model.ntendon > 0)
    _simple_flags: list[tuple[str, int, bool]] = [
      ("Tendons", _tendon_flag, self.mj_model.ntendon > 0),
      ("COM", int(mujoco.mjtVisFlag.mjVIS_COM), False),
      ("Constraints", int(mujoco.mjtVisFlag.mjVIS_CONSTRAINT), False),
      ("Auto-connect", int(mujoco.mjtVisFlag.mjVIS_AUTOCONNECT), False),
    ]

    for label, flag_idx, initial in _simple_flags:
      cb = self.server.gui.add_checkbox(label, initial_value=initial)

      @cb.on_update
      def _(event, _idx=flag_idx) -> None:
        self._apply_visualization_change(
          lambda: _opt.flags.__setitem__(_idx, int(event.target.value))
        )

    # -- Global scale -----------------------------------------------------

    _scale_slider(
      "Global scale",
      _stat,
      "meansize",
      _stat.meansize * 0.1,
      _stat.meansize * 5.0,
      _stat.meansize * 0.01,
    )

  def create_groups_gui(self) -> None:
    """Add geom and site group visibility checkboxes into the
    current GUI context."""
    with self.server.gui.add_folder("Geoms"):
      for i in range(6):
        cb = self.server.gui.add_checkbox(
          f"G{i}",
          initial_value=self.geom_groups_visible[i],
          hint=f"Show/hide geometry in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          def _mutate() -> None:
            self.geom_groups_visible[group_idx] = event.target.value
            self._sync_visibilities()

          self._apply_visualization_change(_mutate)

    with self.server.gui.add_folder("Sites"):
      for i in range(6):
        cb = self.server.gui.add_checkbox(
          f"S{i}",
          initial_value=self.site_groups_visible[i],
          hint=f"Show/hide sites in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          def _mutate() -> None:
            self.site_groups_visible[group_idx] = event.target.value
            self._sync_visibilities()

          self._apply_visualization_change(_mutate)

    # mjvOption group arrays for decor elements.
    _opt_groups: list[tuple[str, str]] = [
      ("Joints", "jointgroup"),
      ("Tendons", "tendongroup"),
      ("Actuators", "actuatorgroup"),
    ]
    for folder_name, attr in _opt_groups:
      group_arr = getattr(self._mjv_option, attr)
      with self.server.gui.add_folder(folder_name):
        for i in range(6):
          cb = self.server.gui.add_checkbox(
            f"{folder_name[0]}{i}",
            initial_value=bool(group_arr[i]),
          )

          @cb.on_update
          def _(event, _arr=group_arr, _idx=i) -> None:
            self._apply_visualization_change(
              lambda: _arr.__setitem__(_idx, int(event.target.value))
            )

  def create_visualization_gui(
    self,
    camera_distance: float = -1.0,
    camera_azimuth: float = 120.0,
    camera_elevation: float = 20.0,
  ) -> viser.GuiTabGroupHandle:
    """Add scene, overlay, and group controls in a tabbed layout.

    Returns the tab group so callers can append additional tabs.

    Args:
        camera_distance: Default camera distance. If negative, derived
            from model extent.
        camera_azimuth: Default camera azimuth angle in degrees.
        camera_elevation: Default camera elevation angle in degrees.
    """
    tabs = self.server.gui.add_tab_group()
    with tabs.add_tab("Scene", icon=viser.Icon.VIDEO):
      self.create_scene_gui(
        camera_distance=camera_distance,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
      )
    with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
      self.create_overlay_gui()
    with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
      self.create_groups_gui()
    return tabs

  def update_from_arrays(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray | None = None,
    mocap_quat: np.ndarray | None = None,
    env_idx: int | None = None,
    qpos: np.ndarray | None = None,
    qvel: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
  ) -> None:
    """Update scene from batched numpy arrays.

    Args:
        body_xpos: Body positions, shape ``(num_envs, nbody, 3)``.
        body_xmat: Body rotation matrices, shape
            ``(num_envs, nbody, 3, 3)``.
        mocap_pos: Mocap body positions, shape
            ``(num_envs, nmocap, 3)``.
        mocap_quat: Mocap body quaternions (wxyz), shape
            ``(num_envs, nmocap, 4)``.
        env_idx: Environment index to visualize. If None, uses
            ``self.env_idx``.
        qpos: Joint positions. Required for contact and tendon
            visualization in the selected environment.
        qvel: Joint velocities. Required for contact and tendon
            visualization in the selected environment.
        ctrl: Controls for the selected environment.
    """
    if env_idx is None:
      env_idx = self.env_idx

    if mocap_pos is None:
      nworld = body_xpos.shape[0]
      mocap_pos = np.zeros((nworld, max(self.mj_model.nmocap, 0), 3))
    if mocap_quat is None:
      nworld = body_xpos.shape[0]
      mocap_quat = np.zeros((nworld, max(self.mj_model.nmocap, 0), 4))

    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = body_xpos[env_idx, self._tracked_body_id, :].copy()
      scene_offset = -tracked_pos

    mj_data: mujoco.MjData | None = None
    if self._any_decor_visible() and qpos is not None and qvel is not None:
      self.mj_data.qpos[:] = qpos[env_idx]
      self.mj_data.qvel[:] = qvel[env_idx]
      if ctrl is not None and self.mj_model.nu > 0:
        self.mj_data.ctrl[:] = ctrl[env_idx]
      if self.mj_model.nmocap > 0:
        self.mj_data.mocap_pos[:] = mocap_pos[env_idx]
        self.mj_data.mocap_quat[:] = mocap_quat[env_idx]
      mujoco.mj_forward(self.mj_model, self.mj_data)
      mj_data = self.mj_data

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      mj_data,
    )

  def update_from_mjdata(self, mj_data: mujoco.MjData) -> None:
    """Update scene from single-environment MuJoCo data.

    Args:
        mj_data: Single environment MuJoCo data.
    """
    body_xpos = mj_data.xpos[None, ...]
    body_xmat = mj_data.xmat.reshape(-1, 3, 3)[None, ...]
    if self.mj_model.nmocap > 0:
      mocap_pos = mj_data.mocap_pos[None, ...]
      mocap_quat = mj_data.mocap_quat[None, ...]
    else:
      mocap_pos = np.zeros((1, 0, 3))
      mocap_quat = np.zeros((1, 0, 4))
    env_idx = 0
    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = mj_data.xpos[self._tracked_body_id, :].copy()
      scene_offset = -tracked_pos

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      mj_data,
    )

  def _update_visualization(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    mj_data: mujoco.MjData | None = None,
  ) -> None:
    """Shared visualization update logic."""
    with self._update_lock:
      self._update_visualization_locked(
        body_xpos,
        body_xmat,
        mocap_pos,
        mocap_quat,
        env_idx,
        scene_offset,
        mj_data,
      )

  def _update_visualization_locked(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    mj_data: mujoco.MjData | None = None,
  ) -> None:
    """Shared visualization update logic.

    The caller must hold ``self._update_lock``.
    """
    self._last_body_xpos = body_xpos
    self._last_body_xmat = body_xmat
    self._last_mocap_pos = mocap_pos
    self._last_mocap_quat = mocap_quat
    self._last_env_idx = env_idx
    self._scene_offset = scene_offset
    if mj_data is not None:
      self._last_mj_data = mj_data

    self.fixed_bodies_frame.position = scene_offset
    slice_single = self.show_only_selected and self.num_envs > 1
    with self.server.atomic():
      body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz
      for mg in self._mesh_groups:
        if not mg.handle.visible:
          continue
        if mg.mocap_ids is not None:
          pos, quat = self._batched_transform_group(
            mocap_pos, mocap_quat, mg.mocap_ids, env_idx, scene_offset, slice_single
          )
        else:
          pos, quat = self._batched_transform_group(
            body_xpos, body_xquat, mg.body_ids, env_idx, scene_offset, slice_single
          )
        mg.handle.batched_positions = pos
        mg.handle.batched_wxyzs = quat

      for (body_id, _), handle in self.site_handles_by_group.items():
        if not handle.visible:
          continue
        pos, quat = self._batched_transform(
          body_xpos, body_xquat, body_id, env_idx, scene_offset, slice_single
        )
        handle.batched_positions = pos
        handle.batched_wxyzs = quat

      if self._show_convex_hull:
        for handle, body_id in self._hull_dynamic_handles:
          if not handle.visible:
            continue
          pos, quat = self._batched_transform(
            body_xpos, body_xquat, body_id, env_idx, scene_offset, slice_single
          )
          handle.batched_positions = pos
          handle.batched_wxyzs = quat

      if self._any_decor_visible() and mj_data is not None:
        self._update_decor_from_mjvscene(mj_data, scene_offset)
      elif not self._any_decor_visible():
        self._hide_all_decor()

      self.server.flush()

  def _batched_transform(
    self,
    positions: np.ndarray,
    quats: np.ndarray,
    idx: int,
    env_idx: int,
    scene_offset: np.ndarray,
    slice_single: bool,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (batched_positions, batched_wxyzs) for a single body."""
    if slice_single:
      pos = positions[env_idx : env_idx + 1, idx] + scene_offset
      quat = quats[env_idx : env_idx + 1, idx]
    else:
      pos = positions[..., idx, :] + scene_offset
      quat = quats[..., idx, :]
    return pos, quat

  def _batched_transform_group(
    self,
    positions: np.ndarray,
    quats: np.ndarray,
    ids: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    slice_single: bool,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (batched_positions, batched_wxyzs) for a group of bodies.

    Gathers transforms for all body IDs in ``ids`` and flattens
    the result to ``(num_envs * len(ids), 3/4)``.
    """
    if slice_single:
      # Show only selected env.
      pos = positions[env_idx : env_idx + 1, ids].reshape(-1, 3) + scene_offset
      quat = quats[env_idx : env_idx + 1, ids].reshape(-1, 4)
    else:
      # positions[:, ids, :] → (num_envs, N, 3) → (num_envs * N, 3)
      pos = (positions[:, ids, :] + scene_offset).reshape(-1, 3)
      quat = quats[:, ids, :].reshape(-1, 4)
    return pos, quat

  def request_update(self) -> None:
    """Request a visualization update and trigger immediate re-render
    from cache."""
    self._apply_visualization_change(lambda: None)

  def refresh_visualization(self) -> None:
    """Re-render the scene using cached visualization data."""
    with self._update_lock:
      self._refresh_visualization_locked()

  def _refresh_visualization_locked(self) -> None:
    """Re-render the scene using cached visualization data.

    The caller must hold ``self._update_lock``.
    """
    if (
      self._last_body_xpos is None
      or self._last_body_xmat is None
      or self._last_mocap_pos is None
      or self._last_mocap_quat is None
    ):
      return

    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = self._last_body_xpos[
        self._last_env_idx, self._tracked_body_id, :
      ].copy()
      scene_offset = -tracked_pos

    self._update_visualization_locked(
      self._last_body_xpos,
      self._last_body_xmat,
      self._last_mocap_pos,
      self._last_mocap_quat,
      self._last_env_idx,
      scene_offset,
      self._last_mj_data,
    )
    self.needs_update = self._any_decor_visible()

  def _add_fixed_geometry(self) -> None:
    """Add fixed world geometry to the scene."""
    # Group fixed geoms by (body_id, group_id).
    body_group_geoms: dict[tuple[int, int], list[int]] = {}
    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      if not is_fixed_body(self.mj_model, body_id):
        continue
      if self.mj_model.geom_type[i] == mjtGeom.mjGEOM_PLANE:
        body_name = get_body_name(self.mj_model, body_id)
        geom_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, i)
        self.server.scene.add_grid(
          f"/fixed_bodies/{body_name}/{geom_name}",
          infinite_grid=True,
          fade_distance=50.0,
          shadow_opacity=0.2,
          plane_opacity=0.4,
          position=self.mj_model.geom_pos[i],
          wxyz=self.mj_model.geom_quat[i],
        )
        continue
      # Skip fully transparent geoms.
      if self.mj_model.geom_rgba[i, 3] == 0:
        continue
      group_id = int(self.mj_model.geom_group[i])
      body_group_geoms.setdefault((body_id, group_id), []).append(i)

    for (body_id, group_id), geom_ids in body_group_geoms.items():
      body_name = get_body_name(self.mj_model, body_id)
      visible = group_id < 6 and self.geom_groups_visible[group_id]
      subgroups = group_geoms_by_visual_compat(self.mj_model, geom_ids)
      for sub_idx, sub_geom_ids in enumerate(subgroups):
        suffix = f"/sub{sub_idx}" if len(subgroups) > 1 else ""
        handle = self.server.scene.add_mesh_trimesh(
          f"/fixed_bodies/{body_name}/group{group_id}{suffix}",
          merge_geoms(self.mj_model, sub_geom_ids),
          cast_shadow=False,
          receive_shadow=0.2,
          position=self.mj_model.body(body_id).pos,
          wxyz=self.mj_model.body(body_id).quat,
          visible=visible,
        )
        self._fixed_geom_handles[(body_id, group_id, sub_idx)] = handle

  @staticmethod
  def _geom_subgroup_fingerprint(
    mj_model: mujoco.MjModel, geom_ids: list[int], is_mocap: bool
  ) -> tuple[object, ...]:
    """Compute a hashable fingerprint for a set of geoms in one body.

    Bodies whose subgroups share a fingerprint have identical local
    geometry and can be rendered with a single batched handle.
    """
    parts: list[tuple[object, ...]] = []
    for gid in geom_ids:
      parts.append(
        (
          int(mj_model.geom_type[gid]),
          int(mj_model.geom_dataid[gid]),
          int(mj_model.geom_matid[gid]),
          tuple(mj_model.geom_size[gid].round(6).tolist()),
          tuple(mj_model.geom_rgba[gid].round(4).tolist()),
          tuple(mj_model.geom_pos[gid].round(6).tolist()),
          tuple(mj_model.geom_quat[gid].round(6).tolist()),
        )
      )
    return (is_mocap, tuple(sorted(parts)))

  def _create_mesh_handles_by_group(self) -> None:
    """Create mesh handles, deduplicating identical geometries across bodies."""
    # Step 1: Group geoms by (body_id, group_id).
    body_group_geoms: dict[tuple[int, int], list[int]] = {}
    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      if is_fixed_body(self.mj_model, body_id):
        continue
      if self.mj_model.geom_rgba[i, 3] == 0:
        continue
      geom_group = self.mj_model.geom_group[i]
      body_group_geoms.setdefault((body_id, geom_group), []).append(i)

    # Group bodies sharing identical geometry by fingerprint.
    fp_info: dict[tuple[object, ...], tuple[list[int], list[int], bool, int, int]] = {}
    for (body_id, group_id), geom_indices in body_group_geoms.items():
      subgroups = group_geoms_by_visual_compat(self.mj_model, geom_indices)
      is_mocap = bool(self.mj_model.body_mocapid[body_id] >= 0)
      for sub_idx, sub_geom_ids in enumerate(subgroups):
        fp = self._geom_subgroup_fingerprint(self.mj_model, sub_geom_ids, is_mocap)
        key = (fp, group_id, sub_idx)
        if key in fp_info:
          fp_info[key][0].append(body_id)
        else:
          fp_info[key] = ([body_id], sub_geom_ids, is_mocap, group_id, sub_idx)

    # Create one batched handle per unique geometry.
    with self.server.atomic():
      for body_ids, geom_ids, is_mocap, group_id, sub_idx in fp_info.values():
        mesh = merge_geoms(self.mj_model, geom_ids)
        lod_ratio = 1000.0 / mesh.vertices.shape[0]

        batch_count = len(body_ids) * self.num_envs
        body_name = get_body_name(self.mj_model, body_ids[0])
        suffix = f"/sub{sub_idx}" if sub_idx > 0 else ""
        visible = group_id < 6 and self.geom_groups_visible[group_id]

        handle = self.server.scene.add_batched_meshes_trimesh(
          f"/bodies/{body_name}/group{group_id}{suffix}",
          mesh,
          batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (batch_count, 1)),
          batched_positions=np.zeros((batch_count, 3)),
          lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
          visible=visible,
        )

        body_ids_arr = np.array(body_ids, dtype=np.int32)
        mocap_ids: np.ndarray | None = None
        if is_mocap:
          mocap_ids = np.array(
            [self.mj_model.body_mocapid[b] for b in body_ids], dtype=np.int32
          )
        self._mesh_groups.append(_MeshGroup(handle, body_ids_arr, group_id, mocap_ids))

  def _add_fixed_sites(self) -> None:
    """Add fixed site geometry to the scene as static nodes."""
    body_group_sites: dict[tuple[int, int], list[int]] = {}
    for site_id in range(self.mj_model.nsite):
      body_id = self.mj_model.site_bodyid[site_id]
      if not is_fixed_body(self.mj_model, body_id):
        continue
      group_id = int(self.mj_model.site_group[site_id])
      body_group_sites.setdefault((body_id, group_id), []).append(site_id)

    for (body_id, group_id), site_ids in body_group_sites.items():
      body_name = get_body_name(self.mj_model, body_id)
      mesh = merge_sites(self.mj_model, site_ids)
      visible = group_id < 6 and self.site_groups_visible[group_id]
      handle = self.server.scene.add_mesh_trimesh(
        f"/fixed_bodies/{body_name}/sites_group{group_id}",
        mesh,
        cast_shadow=False,
        receive_shadow=0.2,
        position=self.mj_model.body(body_id).pos,
        wxyz=self.mj_model.body(body_id).quat,
        visible=visible,
      )
      self._fixed_site_handles[(body_id, group_id)] = handle

  def _create_site_handles_by_group(self) -> None:
    """Create site handles for each site group."""
    body_group_sites: dict[tuple[int, int], list[int]] = {}
    for site_id in range(self.mj_model.nsite):
      body_id = self.mj_model.site_bodyid[site_id]
      if is_fixed_body(self.mj_model, body_id):
        continue
      group_id = int(self.mj_model.site_group[site_id])
      body_group_sites.setdefault((body_id, group_id), []).append(site_id)

    with self.server.atomic():
      for (body_id, group_id), site_ids in body_group_sites.items():
        body_name = get_body_name(self.mj_model, body_id)
        mesh = merge_sites(self.mj_model, site_ids)
        visible = group_id < 6 and self.site_groups_visible[group_id]
        handle = self.server.scene.add_batched_meshes_trimesh(
          f"/bodies/{body_name}/sites_group{group_id}",
          mesh,
          batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          lod="off",
          visible=visible,
        )
        self.site_handles_by_group[(body_id, group_id)] = handle

  def _compute_hull_body_meshes(self) -> None:
    """Precompute one merged convex hull mesh per body that has mesh geoms."""
    body_geoms: dict[int, list[int]] = {}
    for geom_id in range(self.mj_model.ngeom):
      if int(self.mj_model.geom_type[geom_id]) != int(mjtGeom.mjGEOM_MESH):
        continue
      if int(self.mj_model.geom_dataid[geom_id]) < 0:
        continue
      body_id = int(self.mj_model.geom_bodyid[geom_id])
      body_geoms.setdefault(body_id, []).append(geom_id)

    for body_id, geom_ids in body_geoms.items():
      hull_mesh = merge_geoms_hull(self.mj_model, geom_ids)
      if hull_mesh is None:
        continue
      self._hull_body_meshes[body_id] = (
        hull_mesh.vertices.astype(np.float32),
        hull_mesh.faces.astype(np.int32),
      )

  def _build_hull_handles(self) -> None:
    """Build fixed and dynamic handles for precomputed body hulls."""
    color = np.array(self._hull_color, dtype=np.uint8)
    opacity = np.float32(self._hull_opacity)
    fixed_opacities = None if opacity >= 1.0 else np.array([opacity], dtype=np.float32)
    dynamic_opacities = (
      None if opacity >= 1.0 else np.full(self.num_envs, opacity, dtype=np.float32)
    )

    for body_id, (vertices, faces) in self._hull_body_meshes.items():
      if is_fixed_body(self.mj_model, body_id):
        body = self.mj_model.body(body_id)
        handle = self.server.scene.add_batched_meshes_simple(
          f"/fixed_bodies/hull/{body_id}",
          vertices,
          faces,
          batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
          batched_positions=np.zeros((1, 3), dtype=np.float32),
          batched_colors=color[None],
          batched_opacities=fixed_opacities,
          position=body.pos,
          wxyz=body.quat,
          visible=self._show_convex_hull,
          cast_shadow=False,
          receive_shadow=False,
          lod="off",
        )
        self._hull_fixed_handles[body_id] = handle
        continue

      handle = self.server.scene.add_batched_meshes_simple(
        f"/hull/{body_id}",
        vertices,
        faces,
        batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (self.num_envs, 1)).astype(
          np.float32
        ),
        batched_positions=np.zeros((self.num_envs, 3), dtype=np.float32),
        batched_colors=np.tile(color, (self.num_envs, 1)),
        batched_opacities=dynamic_opacities,
        visible=self._show_convex_hull,
        cast_shadow=False,
        receive_shadow=False,
        lod="off",
      )
      self._hull_dynamic_handles.append((handle, body_id))

  def _clear_hull_handles(self) -> None:
    """Remove convex hull handles and clear their caches."""
    for handle in self._hull_fixed_handles.values():
      handle.remove()
    self._hull_fixed_handles.clear()

    for handle, _ in self._hull_dynamic_handles:
      handle.remove()
    self._hull_dynamic_handles.clear()

  def _clear_decor_handles(self) -> None:
    """Remove all decor handles and clear the cache."""
    handles = list(self._decor_handles.values())
    self._decor_handles.clear()
    for handle in handles:
      handle.remove()

  def _hide_all_decor(self) -> None:
    """Hide all decor handles without removing them."""
    for handle in self._decor_handles.values():
      handle.visible = False

  def _update_decor_from_mjvscene(
    self, mj_data: mujoco.MjData, scene_offset: np.ndarray
  ) -> None:
    """Update overlay visualization using mjvScene.

    Calls ``mjv_updateScene`` to generate decorative geoms (contacts,
    tendons, frames, inertia) and renders them as batched viser meshes.
    """
    mujoco.mjv_updateScene(
      self.mj_model,
      mj_data,
      self._mjv_option,
      None,
      self._mjv_camera,
      int(mujoco.mjtCatBit.mjCAT_ALL),
      self._mjv_scene,
    )

    # Group geoms by (type, is_tendon). Tendons are keyed separately so
    # they get cylinder meshes (non-uniform scaling distorts capsule caps)
    # while other capsules (e.g. auto-connect) use real capsule meshes.
    geoms_by_key: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for i in range(self._mjv_scene.ngeom):
      g = self._mjv_scene.geoms[i]
      is_tendon = int(g.objtype) == _OBJ_TENDON
      if int(g.category) == _CAT_DECOR or is_tendon:
        geoms_by_key[(int(g.type), is_tendon)].append(i)

    active_keys: set[tuple[int, bool]] = set()

    # Collect arrow heads across all arrow types.
    all_head_positions: list[np.ndarray] = []
    all_head_orientations: list[np.ndarray] = []
    all_head_scales: list[np.ndarray] = []
    all_head_colors: list[np.ndarray] = []
    all_head_opacities: list[float] = []

    for (geom_type, is_tendon), indices in geoms_by_key.items():
      n = len(indices)
      active_keys.add((geom_type, is_tendon))

      positions = np.empty((n, 3), dtype=np.float32)
      orientations = np.empty((n, 4), dtype=np.float32)
      scales = np.empty((n, 3), dtype=np.float32)
      colors = np.empty((n, 3), dtype=np.uint8)
      opacities = np.empty(n, dtype=np.float32)

      for j, gi in enumerate(indices):
        g = self._mjv_scene.geoms[gi]
        pos = np.array(g.pos) + scene_offset
        mat = np.array(g.mat).reshape(3, 3)
        quat = vtf.SO3.from_matrix(mat).wxyz
        size = np.array(g.size)
        rgba = np.array(g.rgba)

        # MuJoCo ignores user alpha for joint geoms; override it.
        if int(g.objtype) == _OBJ_JOINT:
          rgba[3] = self.mj_model.vis.rgba.joint[3]

        positions[j] = pos
        orientations[j] = quat
        colors[j] = (np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8)
        opacities[j] = float(rgba[3])

        # Map mjvGeom size to unit-mesh scale.
        if geom_type in (_CYLINDER, _CAPSULE):
          # Zero-length capsule = sphere (COM dot in auto-connect).
          height = max(size[2] * 2, size[0])
          scales[j] = [size[0], size[0], height]
        elif geom_type in _ARROWS:
          # size = [shaft_radius, head_radius, total_length]
          # 80% shaft, 20% head (matches mjlab convention).
          total_len = size[2]
          shaft_len = total_len * 0.8
          head_len = total_len * 0.2
          w = size[0]
          scales[j] = [w, w, shaft_len]
          # Head: cone at the tip of the shaft.
          tip_offset = mat[:, 2] * shaft_len
          all_head_positions.append(pos + tip_offset.astype(np.float32))
          all_head_orientations.append(quat)
          all_head_scales.append(np.array([w, w, head_len], dtype=np.float32))
          all_head_colors.append(colors[j].copy())
          all_head_opacities.append(opacities[j])
        else:
          scales[j] = size

      # Tendons use cylinders; other capsules use real capsule meshes.
      mesh_type = _CYLINDER if is_tendon else geom_type
      key = (geom_type, is_tendon)
      batched_opacities = None if np.all(opacities == 1.0) else opacities

      handle = self._decor_handles.get(key)
      if handle is not None:
        handle.batched_positions = positions
        handle.batched_wxyzs = orientations
        handle.batched_scales = scales
        handle.batched_colors = colors
        handle.batched_opacities = batched_opacities
        handle.visible = True
      else:
        unit = _get_unit_mesh(mesh_type)
        handle = self.server.scene.add_batched_meshes_simple(
          f"/decor/{geom_type}_{is_tendon}",
          unit.vertices,
          unit.faces,
          batched_wxyzs=orientations,
          batched_positions=positions,
          batched_scales=scales,
          batched_colors=colors,
          batched_opacities=batched_opacities,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
        self._decor_handles[key] = handle

    # Create arrow heads as a single combined handle.
    if all_head_positions:
      head_key = (_ARROW_HEAD, False)
      active_keys.add(head_key)
      h_pos = np.array(all_head_positions)
      h_ori = np.array(all_head_orientations)
      h_scl = np.array(all_head_scales)
      h_col = np.array(all_head_colors)
      h_opa = np.array(all_head_opacities, dtype=np.float32)
      h_batched_opacities = None if np.all(h_opa == 1.0) else h_opa

      h_handle = self._decor_handles.get(head_key)
      if h_handle is not None:
        h_handle.batched_positions = h_pos
        h_handle.batched_wxyzs = h_ori
        h_handle.batched_scales = h_scl
        h_handle.batched_colors = h_col
        h_handle.batched_opacities = h_batched_opacities
        h_handle.visible = True
      else:
        unit = _get_unit_mesh(_ARROW_HEAD)
        h_handle = self.server.scene.add_batched_meshes_simple(
          "/decor/arrow_heads",
          unit.vertices,
          unit.faces,
          batched_wxyzs=h_ori,
          batched_positions=h_pos,
          batched_scales=h_scl,
          batched_colors=h_col,
          batched_opacities=h_batched_opacities,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
        self._decor_handles[head_key] = h_handle

    # Hide handles for types not present this frame.
    for key, handle in self._decor_handles.items():
      if key not in active_keys:
        handle.visible = False
