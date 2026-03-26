"""Manages all Viser visualization handles and state for MuJoCo models."""

from __future__ import annotations

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
  merge_sites,
  rotation_matrix_from_vectors,
)

# Viser visualization defaults.
_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150
_DEFAULT_ENVIRONMENT_INTENSITY = 0.8
_DEFAULT_CONTACT_POINT_COLOR = (230, 153, 51)
_DEFAULT_CONTACT_FORCE_COLOR = (255, 0, 0)


class _Contact:
  """Contact data from MuJoCo."""

  __slots__ = ("pos", "frame", "force", "included")

  def __init__(
    self,
    pos: np.ndarray,
    frame: np.ndarray,
    force: np.ndarray,
    included: bool,
  ) -> None:
    self.pos = pos
    self.frame = frame
    self.force = force
    self.included = included


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

    # Handles.
    self.mesh_handles_by_group: dict[tuple[int, int, int], viser.BatchedGlbHandle] = {}
    self.site_handles_by_group: dict[tuple[int, int], viser.BatchedGlbHandle] = {}
    self.contact_point_handle: viser.BatchedMeshHandle | None = None
    self.contact_force_shaft_handle: viser.BatchedMeshHandle | None = None
    self.contact_force_head_handle: viser.BatchedMeshHandle | None = None
    self._fixed_site_handles: dict[tuple[int, int], viser.GlbHandle] = {}

    # Visualization settings.
    self.env_idx = 0
    self.camera_tracking_enabled = True
    self.show_only_selected = False
    self.geom_groups_visible = [True, True, True, False, False, False]
    self.site_groups_visible = [True, True, True, False, False, False]
    self.show_contact_points = False
    self.show_contact_forces = False
    self.contact_point_color: tuple[int, int, int] = _DEFAULT_CONTACT_POINT_COLOR
    self.contact_force_color: tuple[int, int, int] = _DEFAULT_CONTACT_FORCE_COLOR
    self.meansize_override: float | None = None
    self.needs_update = False
    self.paused = False

    # Cached visualization state for re-rendering when settings change.
    self._tracked_body_id: int | None = None
    self._last_body_xpos: np.ndarray | None = None
    self._last_body_xmat: np.ndarray | None = None
    self._last_mocap_pos: np.ndarray | None = None
    self._last_mocap_quat: np.ndarray | None = None
    self._last_env_idx = 0
    self._last_contacts: list[_Contact] | None = None
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

    for body_id in range(mj_model.nbody):
      if not is_fixed_body(mj_model, body_id):
        self._tracked_body_id = body_id
        break

  def _sync_visibilities(self) -> None:
    """Synchronize all handle visibilities based on current flags."""
    for (
      _body_id,
      group_id,
      _sub_idx,
    ), handle in self.mesh_handles_by_group.items():
      handle.visible = group_id < 6 and self.geom_groups_visible[group_id]

    for (_body_id, group_id), handle in self.site_handles_by_group.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    for (_body_id, group_id), handle in self._fixed_site_handles.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    if self.contact_point_handle is not None and not self.show_contact_points:
      self.contact_point_handle.visible = False

    if not self.show_contact_forces:
      if self.contact_force_shaft_handle is not None:
        self.contact_force_shaft_handle.visible = False
      if self.contact_force_head_handle is not None:
        self.contact_force_head_handle.visible = False

  def create_visualization_gui(
    self,
    camera_distance: float = -1.0,
    camera_azimuth: float = 120.0,
    camera_elevation: float = 20.0,
  ) -> None:
    """Add camera, environment, and contact controls into the current
    GUI context. Call this inside a tab or folder.

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
          self.env_idx = int(env_slider.value)
          self._request_update()

        show_only_cb = self.server.gui.add_checkbox(
          "Hide others",
          initial_value=self.show_only_selected,
          hint="Show only the selected environment.",
        )

        @show_only_cb.on_update
        def _(_) -> None:
          self.show_only_selected = show_only_cb.value
          self._request_update()

    _center = self.mj_model.stat.center.copy()
    _extent = self.mj_model.stat.extent
    if camera_distance <= 0:
      camera_distance = 1.5 * _extent
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
        self.camera_tracking_enabled = cb_camera_tracking.value
        if self.camera_tracking_enabled:
          for client in self.server.get_clients().values():
            client.camera.position = _camera_offset
            client.camera.look_at = np.zeros(3)
        self._request_update()

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
        if self.camera_tracking_enabled:
          client.camera.position = _camera_offset
          client.camera.look_at = _center

    with self.server.gui.add_folder("Contacts"):
      cb_contact_points = self.server.gui.add_checkbox(
        "Points",
        initial_value=False,
      )
      contact_point_color = self.server.gui.add_rgb(
        "Points Color", initial_value=self.contact_point_color
      )
      cb_contact_forces = self.server.gui.add_checkbox(
        "Forces",
        initial_value=False,
      )
      contact_force_color = self.server.gui.add_rgb(
        "Forces Color", initial_value=self.contact_force_color
      )
      meansize_input = self.server.gui.add_number(
        "Scale",
        step=self.mj_model.stat.meansize * 0.01,
        initial_value=self.mj_model.stat.meansize,
      )

      @cb_contact_points.on_update
      def _(_) -> None:
        self.show_contact_points = cb_contact_points.value
        self._sync_visibilities()
        self._request_update()

      @contact_point_color.on_update
      def _(_) -> None:
        self.contact_point_color = contact_point_color.value
        if self.contact_point_handle is not None:
          self.contact_point_handle.remove()
          self.contact_point_handle = None
        self._request_update()

      @cb_contact_forces.on_update
      def _(_) -> None:
        self.show_contact_forces = cb_contact_forces.value
        self._sync_visibilities()
        self._request_update()

      @contact_force_color.on_update
      def _(_) -> None:
        self.contact_force_color = contact_force_color.value
        if self.contact_force_shaft_handle is not None:
          self.contact_force_shaft_handle.remove()
          self.contact_force_shaft_handle = None
        if self.contact_force_head_handle is not None:
          self.contact_force_head_handle.remove()
          self.contact_force_head_handle = None
        self._request_update()

      @meansize_input.on_update
      def _(_) -> None:
        self.meansize_override = meansize_input.value
        self._request_update()

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
          self.geom_groups_visible[group_idx] = event.target.value
          self._sync_visibilities()
          self._request_update()

    with self.server.gui.add_folder("Sites"):
      for i in range(6):
        cb = self.server.gui.add_checkbox(
          f"S{i}",
          initial_value=self.site_groups_visible[i],
          hint=f"Show/hide sites in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          self.site_groups_visible[group_idx] = event.target.value
          self._sync_visibilities()
          self._request_update()

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
        qpos: Joint positions for contact computation.
        qvel: Joint velocities for contact computation.
        ctrl: Controls for contact computation.
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

    contacts = None
    if self.show_contact_points or self.show_contact_forces:
      if qpos is not None and qvel is not None:
        self.mj_data.qpos[:] = qpos[env_idx]
        self.mj_data.qvel[:] = qvel[env_idx]
        if ctrl is not None and self.mj_model.nu > 0:
          self.mj_data.ctrl[:] = ctrl[env_idx]
        if self.mj_model.nmocap > 0:
          self.mj_data.mocap_pos[:] = mocap_pos[env_idx]
          self.mj_data.mocap_quat[:] = mocap_quat[env_idx]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        contacts = self._extract_contacts_from_mjdata(self.mj_data)

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      contacts,
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

    contacts = self._extract_contacts_from_mjdata(mj_data)

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      contacts,
    )

  def _update_visualization(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    contacts: list[_Contact] | None,
  ) -> None:
    """Shared visualization update logic."""
    self._last_body_xpos = body_xpos
    self._last_body_xmat = body_xmat
    self._last_mocap_pos = mocap_pos
    self._last_mocap_quat = mocap_quat
    self._last_env_idx = env_idx
    self._scene_offset = scene_offset
    if contacts is not None:
      self._last_contacts = contacts

    self.fixed_bodies_frame.position = scene_offset
    tile = self.show_only_selected and self.num_envs > 1
    with self.server.atomic():
      body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz
      for (body_id, _, _), handle in self.mesh_handles_by_group.items():
        if not handle.visible:
          continue
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id >= 0:
          pos, quat = self._batched_transform(
            mocap_pos, mocap_quat, mocap_id, env_idx, scene_offset, tile
          )
        else:
          pos, quat = self._batched_transform(
            body_xpos, body_xquat, body_id, env_idx, scene_offset, tile
          )
        handle.batched_positions = pos
        handle.batched_wxyzs = quat

      for (body_id, _), handle in self.site_handles_by_group.items():
        if not handle.visible:
          continue
        pos, quat = self._batched_transform(
          body_xpos, body_xquat, body_id, env_idx, scene_offset, tile
        )
        handle.batched_positions = pos
        handle.batched_wxyzs = quat

      if contacts is not None:
        self._update_contact_visualization(contacts, scene_offset)

      self.server.flush()

  def _batched_transform(
    self,
    positions: np.ndarray,
    quats: np.ndarray,
    idx: int,
    env_idx: int,
    scene_offset: np.ndarray,
    tile: bool,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (batched_positions, batched_wxyzs) for a single body."""
    if tile:
      pos = np.tile((positions[env_idx, idx] + scene_offset)[None], (self.num_envs, 1))
      quat = np.tile(quats[env_idx, idx][None], (self.num_envs, 1))
    else:
      pos = positions[..., idx, :] + scene_offset
      quat = quats[..., idx, :]
    return pos, quat

  def _request_update(self) -> None:
    """Request a visualization update and trigger immediate re-render
    from cache."""
    self.needs_update = True
    self.refresh_visualization()

  def refresh_visualization(self) -> None:
    """Re-render the scene using cached visualization data."""
    if (
      self._last_body_xpos is None
      or self._last_body_xmat is None
      or self._last_mocap_pos is None
      or self._last_mocap_quat is None
    ):
      return

    contacts = (
      self._last_contacts
      if (self.show_contact_points or self.show_contact_forces)
      else None
    )

    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = self._last_body_xpos[
        self._last_env_idx, self._tracked_body_id, :
      ].copy()
      scene_offset = -tracked_pos

    self._update_visualization(
      self._last_body_xpos,
      self._last_body_xmat,
      self._last_mocap_pos,
      self._last_mocap_quat,
      self._last_env_idx,
      scene_offset,
      contacts,
    )
    self.needs_update = self.show_contact_points or self.show_contact_forces

  def _add_fixed_geometry(self) -> None:
    """Add fixed world geometry to the scene."""
    body_geoms: dict[int, list[int]] = {}
    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      body_geoms.setdefault(body_id, []).append(i)

    for body_id, geom_ids in body_geoms.items():
      if not is_fixed_body(self.mj_model, body_id):
        continue

      body_name = get_body_name(self.mj_model, body_id)
      nonplane_geom_ids: list[int] = []
      for geom_id in geom_ids:
        geom_type = self.mj_model.geom_type[geom_id]
        if geom_type == mjtGeom.mjGEOM_PLANE:
          geom_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, geom_id)
          self.server.scene.add_grid(
            f"/fixed_bodies/{body_name}/{geom_name}",
            infinite_grid=True,
            fade_distance=50.0,
            shadow_opacity=0.2,
            plane_opacity=0.4,
            position=self.mj_model.geom_pos[geom_id],
            wxyz=self.mj_model.geom_quat[geom_id],
          )
        else:
          nonplane_geom_ids.append(geom_id)

      if nonplane_geom_ids:
        subgroups = group_geoms_by_visual_compat(self.mj_model, nonplane_geom_ids)
        for sub_idx, sub_geom_ids in enumerate(subgroups):
          suffix = f"/sub{sub_idx}" if len(subgroups) > 1 else ""
          self.server.scene.add_mesh_trimesh(
            f"/fixed_bodies/{body_name}{suffix}",
            merge_geoms(self.mj_model, sub_geom_ids),
            cast_shadow=False,
            receive_shadow=0.2,
            position=self.mj_model.body(body_id).pos,
            wxyz=self.mj_model.body(body_id).quat,
            visible=True,
          )

  def _create_mesh_handles_by_group(self) -> None:
    """Create mesh handles for each geom group separately."""
    body_group_geoms: dict[tuple[int, int], list[int]] = {}

    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      if is_fixed_body(self.mj_model, body_id):
        continue
      # Skip fully transparent geoms (alpha == 0).
      if self.mj_model.geom_rgba[i, 3] == 0:
        continue
      geom_group = self.mj_model.geom_group[i]
      body_group_geoms.setdefault((body_id, geom_group), []).append(i)

    with self.server.atomic():
      for (body_id, group_id), geom_indices in body_group_geoms.items():
        body_name = get_body_name(self.mj_model, body_id)
        subgroups = group_geoms_by_visual_compat(self.mj_model, geom_indices)
        visible = group_id < 6 and self.geom_groups_visible[group_id]

        for sub_idx, sub_geom_ids in enumerate(subgroups):
          mesh = merge_geoms(self.mj_model, sub_geom_ids)
          lod_ratio = 1000.0 / mesh.vertices.shape[0]
          suffix = f"/sub{sub_idx}" if len(subgroups) > 1 else ""
          handle = self.server.scene.add_batched_meshes_trimesh(
            f"/bodies/{body_name}/group{group_id}{suffix}",
            mesh,
            batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
              self.num_envs, axis=0
            ),
            batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
              self.num_envs, axis=0
            ),
            lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
            visible=visible,
          )
          self.mesh_handles_by_group[(body_id, group_id, sub_idx)] = handle

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

  def _extract_contacts_from_mjdata(self, mj_data: mujoco.MjData) -> list[_Contact]:
    """Extract contact data from given MuJoCo data."""

    def make_contact(i: int) -> _Contact:
      con, force = mj_data.contact[i], np.zeros(6)
      mujoco.mj_contactForce(self.mj_model, mj_data, i, force)
      return _Contact(
        pos=con.pos.copy(),
        frame=con.frame.copy().reshape(3, 3),
        force=force[:3].copy(),
        included=con.efc_address >= 0,
      )

    return [make_contact(i) for i in range(mj_data.ncon)]

  def _update_contact_visualization(
    self, contacts: list[_Contact], scene_offset: np.ndarray
  ) -> None:
    """Update contact point and force visualization."""
    meansize = self.meansize_override or self.mj_model.stat.meansize
    z_up = np.array([0, 0, 1])

    # Build arrays directly instead of intermediate dataclasses.
    point_positions: list[np.ndarray] = []
    point_orientations: list[np.ndarray] = []
    point_scales: list[np.ndarray] = []

    force_shaft_pos: list[np.ndarray] = []
    force_shaft_ori: list[np.ndarray] = []
    force_shaft_scl: list[np.ndarray] = []
    force_head_pos: list[np.ndarray] = []
    force_head_ori: list[np.ndarray] = []
    force_head_scl: list[np.ndarray] = []

    cw = self.mj_model.vis.scale.contactwidth * meansize
    ch = self.mj_model.vis.scale.contactheight * meansize
    fw = self.mj_model.vis.scale.forcewidth * meansize
    force_scale = (
      self.mj_model.vis.map.force / self.mj_model.stat.meanmass
      if self.mj_model.stat.meanmass > 0
      else 1.0
    )

    for contact in contacts:
      if not contact.included:
        continue

      pos = contact.pos + scene_offset

      if self.show_contact_points:
        point_positions.append(pos)
        point_orientations.append(
          vtf.SO3.from_matrix(
            rotation_matrix_from_vectors(z_up, contact.frame[0, :])
          ).wxyz
        )
        point_scales.append(np.array([cw, cw, ch]))

      if self.show_contact_forces:
        force_world = contact.frame.T @ contact.force
        force_mag = float(np.linalg.norm(force_world))
        if force_mag > 1e-6:
          force_dir = force_world / force_mag
          arrow_len = force_mag * force_scale
          quat = vtf.SO3.from_matrix(rotation_matrix_from_vectors(z_up, force_dir)).wxyz
          force_shaft_pos.append(pos)
          force_shaft_ori.append(quat)
          force_shaft_scl.append(np.array([fw, fw, arrow_len]))
          force_head_pos.append(pos + force_dir * arrow_len)
          force_head_ori.append(quat)
          force_head_scl.append(np.array([fw, fw, fw]))

    self._apply_batched_contacts(
      point_positions,
      point_orientations,
      point_scales,
      force_shaft_pos,
      force_shaft_ori,
      force_shaft_scl,
      force_head_pos,
      force_head_ori,
      force_head_scl,
    )

  def _apply_batched_contacts(
    self,
    point_pos: list[np.ndarray],
    point_ori: list[np.ndarray],
    point_scl: list[np.ndarray],
    shaft_pos: list[np.ndarray],
    shaft_ori: list[np.ndarray],
    shaft_scl: list[np.ndarray],
    head_pos: list[np.ndarray],
    head_ori: list[np.ndarray],
    head_scl: list[np.ndarray],
  ) -> None:
    """Push contact arrays to viser batched handles."""
    if point_pos:
      positions = np.array(point_pos, dtype=np.float32)
      orientations = np.array(point_ori, dtype=np.float32)
      scales = np.array(point_scl, dtype=np.float32)
      if self.contact_point_handle is None:
        mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
        self.contact_point_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/points",
          mesh.vertices,
          mesh.faces,
          batched_wxyzs=orientations,
          batched_positions=positions,
          batched_scales=scales,
          batched_colors=np.array(self.contact_point_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
      self.contact_point_handle.batched_positions = positions
      self.contact_point_handle.batched_wxyzs = orientations
      self.contact_point_handle.batched_scales = scales
      self.contact_point_handle.visible = True
    elif self.contact_point_handle is not None:
      self.contact_point_handle.visible = False

    if shaft_pos:
      s_pos = np.array(shaft_pos, dtype=np.float32)
      s_ori = np.array(shaft_ori, dtype=np.float32)
      s_scl = np.array(shaft_scl, dtype=np.float32)
      h_pos = np.array(head_pos, dtype=np.float32)
      h_ori = np.array(head_ori, dtype=np.float32)
      h_scl = np.array(head_scl, dtype=np.float32)
      if (
        self.contact_force_shaft_handle is None
        or self.contact_force_head_handle is None
      ):
        if self.contact_force_shaft_handle is not None:
          self.contact_force_shaft_handle.remove()
        if self.contact_force_head_handle is not None:
          self.contact_force_head_handle.remove()
        shaft_mesh = trimesh.creation.cylinder(radius=0.4, height=1.0)
        shaft_mesh.apply_translation([0, 0, 0.5])
        self.contact_force_shaft_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/forces/shaft",
          shaft_mesh.vertices,
          shaft_mesh.faces,
          batched_wxyzs=s_ori,
          batched_positions=s_pos,
          batched_scales=s_scl,
          batched_colors=np.array(self.contact_force_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
        head_mesh = trimesh.creation.cone(radius=1.0, height=1.0, sections=8)
        self.contact_force_head_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/forces/head",
          head_mesh.vertices,
          head_mesh.faces,
          batched_wxyzs=h_ori,
          batched_positions=h_pos,
          batched_scales=h_scl,
          batched_colors=np.array(self.contact_force_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
      self.contact_force_shaft_handle.batched_positions = s_pos
      self.contact_force_shaft_handle.batched_wxyzs = s_ori
      self.contact_force_shaft_handle.batched_scales = s_scl
      self.contact_force_shaft_handle.visible = True
      self.contact_force_head_handle.batched_positions = h_pos
      self.contact_force_head_handle.batched_wxyzs = h_ori
      self.contact_force_head_handle.batched_scales = h_scl
      self.contact_force_head_handle.visible = True
    elif (
      self.contact_force_shaft_handle is not None
      and self.contact_force_head_handle is not None
    ):
      self.contact_force_shaft_handle.visible = (
        self.contact_force_head_handle.visible
      ) = False
