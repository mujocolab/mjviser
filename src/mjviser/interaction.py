"""Interactive perturbation: click to select, cmd/ctrl+drag to apply a force.

Backed by MuJoCo's own perturbation engine (``mjvPerturb`` +
``mjv_applyPerturbForce`` / ``mjv_applyPerturbPose``), so the spring force,
critical damping, and pose math match MuJoCo's simulate and studio exactly --
including velocity damping computed analytically from ``mj_objectVelocity``.

The only web-specific part is the input. MuJoCo's native apps accumulate the
reference point from relative mouse deltas scaled by a depth factor, because
they only have 2D cursor motion. viser drag events instead report absolute
world-space coordinates (the grab point on the body, and the cursor projected
onto a camera-aligned plane), so we set the reference selection point directly.

cmd/ctrl + left-drag applies a force at the grab point (off-center grabs also
rotate the body through the moment arm). cmd/ctrl + right-drag rotates the body
toward an orientation built from the screen drag. While paused, both drags move
free-joint / mocap bodies kinematically instead of applying forces.
"""

from __future__ import annotations

from threading import Lock

import mujoco
import numpy as np
import viser

_CONNECTOR_NAME = "/perturb/connector"
_CONNECTOR_COLOR = (255, 90, 90)
_GHOST_NAME = "/perturb/ghost"
_GHOST_COLOR = (255, 200, 60)


class PerturbationHandler:
  """Manages body selection and drag-to-perturb on the viewer's MjData."""

  def __init__(
    self,
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
  ) -> None:
    self._server = server
    self._model = mj_model
    self._data = mj_data
    self._lock = Lock()

    self.selected_body_id: int | None = None

    # MuJoCo's perturbation state. We populate select / localpos / localmass /
    # refselpos / refpos / refquat ourselves and let MuJoCo do the physics.
    self._pert = mujoco.MjvPerturb()
    self._pert.active = 0

    self._drag_body_id: int | None = None
    self._grab_local: np.ndarray | None = None  # grab point in body-local coords
    self._target_viser: np.ndarray | None = None  # cursor target, viser frame

    # Rotate drag (cmd + right-drag): a target orientation built from the
    # camera-relative screen drag, applied as a spring torque toward it.
    self._drag_rotate: bool = False
    self._rotate_delta_quat: np.ndarray | None = None  # screen-driven rotation
    self._rotate_initial_quat: np.ndarray | None = None  # body orientation at start
    self._rotate_anchor: np.ndarray | None = None  # drag-start screen position
    # Ghost-box (MuJoCo-style orientation indicator) state, captured at start.
    self._rotate_aabb_center: np.ndarray | None = None  # body-frame box center
    self._rotate_aabb_half: np.ndarray | None = None  # body-frame box half-extents
    self._rotate_world_quat: np.ndarray | None = None  # body world orientation

    # Body pose for the active env, cached each render frame. Used to convert
    # the drag-start grab point into body-local coords and to draw the
    # connector; the force/pose physics reads the live MjData passed to apply().
    self._scene_offset = np.zeros(3)
    self._body_xpos: np.ndarray | None = None
    self._body_xmat: np.ndarray | None = None

    # Body whose xfrc_applied we last wrote, so we can zero exactly that
    # entry when the drag ends without disturbing other bodies' applied forces.
    self._applied_body_id: int | None = None

    self._info_text: viser.GuiTextHandle | None = None
    self._connector: viser.LineSegmentsHandle | None = None
    self._ghost: viser.BoxHandle | None = None

  def setup_gui(self) -> None:
    """Add a selection-info display to the GUI."""
    with self._server.gui.add_folder("Selection"):
      self._info_text = self._server.gui.add_text(
        "Body", initial_value="(none)", disabled=True
      )

  def clear(self) -> None:
    """Drop selection and any in-flight drag.

    Call on reset or after the model is rebuilt: a cached body id can
    otherwise outlive the body it pointed to (out-of-range index, or a
    silently retargeted body) and the GUI label would keep a stale name.
    The dragged body's ``xfrc_applied`` is zeroed by the next ``apply``.
    """
    with self._lock:
      self.selected_body_id = None
      self._drag_body_id = None
      self._grab_local = None
      self._target_viser = None
      self._drag_rotate = False
      self._rotate_delta_quat = None
      self._rotate_initial_quat = None
      self._rotate_anchor = None
      self._rotate_aabb_center = None
      self._rotate_aabb_half = None
      self._rotate_world_quat = None
      self._pert.active = 0
    self._hide_connector()
    self._hide_ghost()
    if self._info_text is not None:
      self._info_text.value = "(none)"

  def register_drag_handlers(
    self,
    handle: viser.BatchedGlbHandle,
    body_ids: np.ndarray,
  ) -> None:
    """Attach click (select) and cmd/ctrl+drag (perturb) handlers."""
    # Skip groups whose only body is the world body (id 0): perturbing it is
    # meaningless, and registering any handler makes the mesh interactive in
    # viser, flipping the cursor to "pointer" across the whole ground plane.
    if not bool(np.any(body_ids != 0)):
      return
    n_bodies = len(body_ids)

    def _body_id(idx: int | None) -> int | None:
      if idx is None:
        return None
      return int(body_ids[idx % n_bodies])

    @handle.on_click
    def _(event: viser.SceneNodePointerEvent) -> None:  # type: ignore[type-arg]
      bid = _body_id(event.instance_index)
      if bid is None or bid == 0:
        return
      with self._lock:
        self.selected_body_id = bid
      self._set_info(bid)

    # cmd/ctrl + left-drag moves the body in the camera-facing plane. Depth
    # (toward/away from the camera) isn't a separate gesture: viser freezes
    # the drag modifier at drag-start and has no scroll-during-drag, so there
    # is no fluid way to switch planes mid-drag. Orbit the view and drag again
    # to reach a different depth.
    @handle.on_drag("left", modifier="cmd/ctrl")
    def _(event: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      if event.phase == "start":
        self._on_drag_start(_body_id(event.instance_index), event.start_position)
      elif event.phase == "update":
        with self._lock:
          if self._drag_body_id is not None and not self._drag_rotate:
            self._target_viser = np.array(event.end_position)
      else:  # "end"
        self._end_drag()

    # cmd/ctrl + right-drag rotates the body: the screen drag builds a target
    # orientation (yaw from horizontal motion, pitch from vertical), and the
    # body is sprung toward it -- MuJoCo's rotate perturbation.
    @handle.on_drag("right", modifier="cmd/ctrl")
    def _(event: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      if event.phase == "start":
        self._on_rotate_start(_body_id(event.instance_index), event.start_screen_pos)
      elif event.phase == "update":
        dq = self._rotation_from_drag(event)
        with self._lock:
          if self._drag_body_id is not None and self._drag_rotate:
            self._rotate_delta_quat = dq
      else:  # "end"
        self._end_drag()

  def _end_drag(self) -> None:
    with self._lock:
      self._drag_body_id = None
      self._grab_local = None
      self._target_viser = None
      self._drag_rotate = False
      self._rotate_delta_quat = None
      self._rotate_initial_quat = None
      self._rotate_anchor = None
      self._rotate_aabb_center = None
      self._rotate_aabb_half = None
      self._rotate_world_quat = None
      self._pert.active = 0
    self._hide_connector()
    self._hide_ghost()

  def _on_drag_start(
    self, bid: int | None, start_position: tuple[float, float, float]
  ) -> None:
    if bid is None or bid == 0:
      return
    with self._lock:
      if (
        self._body_xpos is None
        or self._body_xmat is None
        or bid >= self._body_xpos.shape[0]
      ):
        return
      # viser reports world coords in the (camera-tracking) viser frame;
      # subtract the scene offset to get MuJoCo-frame coords.
      grab_world = np.array(start_position) - self._scene_offset
      xpos = self._body_xpos[bid]
      xmat = self._body_xmat[bid]
      self._grab_local = xmat.T @ (grab_world - xpos)
      self._drag_body_id = bid
      self._drag_rotate = False
      self._target_viser = np.array(start_position)
      self.selected_body_id = bid
    self._set_info(bid)

  def _on_rotate_start(
    self, bid: int | None, start_screen_pos: tuple[float, float]
  ) -> None:
    if bid is None or bid == 0:
      return
    with self._lock:
      self._drag_body_id = bid
      self._drag_rotate = True
      self._rotate_anchor = np.array(start_screen_pos, dtype=float)
      self._rotate_delta_quat = np.array([1.0, 0.0, 0.0, 0.0])
      self._rotate_initial_quat = None  # captured on first apply
      self._rotate_aabb_center, self._rotate_aabb_half = self._body_aabb(bid)
      if self._body_xmat is not None and bid < self._body_xmat.shape[0]:
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self._body_xmat[bid].reshape(9))
        self._rotate_world_quat = quat
      self.selected_body_id = bid
    self._set_info(bid)

  def _body_aabb(self, bid: int) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (center, half-extent) of a body's geoms,
    in the body frame, from each geom's bounding sphere."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    found = False
    for g in range(self._model.ngeom):
      if int(self._model.geom_bodyid[g]) != bid:
        continue
      pos = np.asarray(self._model.geom_pos[g], dtype=float)
      r = float(self._model.geom_rbound[g])
      if r <= 0.0:
        r = float(np.max(self._model.geom_size[g])) or 0.05
      lo = np.minimum(lo, pos - r)
      hi = np.maximum(hi, pos + r)
      found = True
    if not found:
      return np.zeros(3), np.full(3, 0.05)
    return (lo + hi) / 2.0, (hi - lo) / 2.0

  def _rotation_from_drag(
    self,
    event: viser.SceneNodeDragEvent,  # type: ignore[type-arg]
  ) -> np.ndarray:
    """Build a delta rotation quaternion from the camera-relative screen drag."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    if self._rotate_anchor is None:
      return identity
    delta = np.array(event.end_screen_pos, dtype=float) - self._rotate_anchor
    cam = event.client.camera
    pos = np.asarray(cam.position, dtype=float)
    forward = np.asarray(cam.look_at, dtype=float) - pos
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, np.asarray(cam.up_direction, dtype=float))
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    # Horizontal drag yaws about the camera up axis; vertical drag (OpenCV y is
    # down) pitches about the camera right axis. A full-screen drag ~= 180 deg.
    rotvec = up * (delta[0] * np.pi) + right * (delta[1] * np.pi)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
      return identity
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, rotvec / angle, angle)
    return quat

  def update_state(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
  ) -> None:
    """Cache body pose and scene offset from the latest render frame.

    Args:
      body_xpos: Shape ``(num_envs, nbody, 3)``.
      body_xmat: Shape ``(num_envs, nbody, 3, 3)``.
      env_idx: Active environment index.
      scene_offset: Current camera-tracking offset (viser - mujoco).
    """
    self._body_xpos = body_xpos[env_idx]
    self._body_xmat = body_xmat[env_idx]
    self._scene_offset = scene_offset
    self._update_connector()
    self._update_rotate_ghost()

  def apply(self, model: mujoco.MjModel, data: mujoco.MjData, paused: bool) -> bool:
    """Apply the active drag to ``data`` using MuJoCo's perturb engine.

    When running, writes a critically-damped spring force to
    ``xfrc_applied`` (consumed by the next ``mj_step``). When paused,
    repositions free-joint / mocap bodies so the grab point follows the
    cursor. Returns True if it moved the body's pose (paused drag), so the
    caller can trigger a re-render.
    """
    with self._lock:
      bid = self._drag_body_id
      if bid is None or bid <= 0 or bid >= model.nbody:
        self._clear_applied(data)
        return False
      self._pert.select = bid

      if self._drag_rotate:
        return self._apply_rotate(model, data, paused, bid)

      if self._grab_local is None or self._target_viser is None:
        self._clear_applied(data)
        return False

      xmat = data.xmat[bid].reshape(3, 3)
      selpos = data.xpos[bid] + xmat @ self._grab_local
      target = self._target_viser - self._scene_offset

      self._pert.localpos = self._grab_local

      if paused:
        # Reposition: translate the body so the grab point reaches the
        # target, keeping orientation. mjv_applyPerturbPose moves free
        # joints / mocap bodies and is a no-op for anything else.
        self._clear_applied(data)
        xiquat = np.empty(4)
        mujoco.mju_mulQuat(xiquat, data.xquat[bid], model.body_iquat[bid])
        self._pert.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
        self._pert.refpos = data.xipos[bid] + (target - selpos)
        self._pert.refquat = xiquat
        mujoco.mjv_applyPerturbPose(model, data, self._pert, 1)
        # mj_forward (not just mj_kinematics) so collisions, contacts, and
        # constraint forces refresh for the dragged pose -- matches MuJoCo
        # simulate, which runs mj_forward every frame while paused.
        mujoco.mj_forward(model, data)
        return True

      # Running: spring force toward the cursor. localmass is MuJoCo's
      # precomputed effective translational mass for the body.
      invweight = float(model.body_invweight0[bid, 0])
      self._pert.localmass = 1.0 / invweight if invweight > 0 else 1.0
      self._pert.refselpos = target
      self._pert.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
      mujoco.mjv_applyPerturbForce(model, data, self._pert)
      self._applied_body_id = bid
      return False

  def _apply_rotate(
    self, model: mujoco.MjModel, data: mujoco.MjData, paused: bool, bid: int
  ) -> bool:
    """Spring the body toward the drag-built target orientation.

    The caller holds ``self._lock``.
    """
    if self._rotate_delta_quat is None:
      self._clear_applied(data)
      return False
    # Capture the body's orientation once, at the first apply of this rotate.
    if self._rotate_initial_quat is None:
      initial = np.empty(4)
      mujoco.mju_mulQuat(initial, data.xquat[bid], model.body_iquat[bid])
      self._rotate_initial_quat = initial
    refquat = np.empty(4)
    mujoco.mju_mulQuat(refquat, self._rotate_delta_quat, self._rotate_initial_quat)
    mujoco.mju_normalize4(refquat)
    self._pert.refpos = data.xipos[bid].copy()
    self._pert.refquat = refquat
    self._pert.active = int(mujoco.mjtPertBit.mjPERT_ROTATE)
    if paused:
      self._clear_applied(data)
      mujoco.mjv_applyPerturbPose(model, data, self._pert, 1)
      mujoco.mj_forward(model, data)
      return True
    mujoco.mjv_applyPerturbForce(model, data, self._pert)
    self._applied_body_id = bid
    return False

  def _clear_applied(self, data: mujoco.MjData) -> None:
    """Zero the xfrc_applied entry we last wrote, if any."""
    bid = self._applied_body_id
    if bid is not None and bid < data.xfrc_applied.shape[0]:
      data.xfrc_applied[bid] = 0.0
    self._applied_body_id = None

  def _set_info(self, bid: int) -> None:
    if self._info_text is not None:
      name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, bid)
      self._info_text.value = name or f"body_{bid}"

  def _update_connector(self) -> None:
    """Draw or refresh the grab-point-to-cursor connector line."""
    selpos = target = None
    with self._lock:
      bid = self._drag_body_id
      if (
        bid is not None
        and self._grab_local is not None
        and self._target_viser is not None
        and self._body_xpos is not None
        and self._body_xmat is not None
        and bid < self._body_xpos.shape[0]
      ):
        xmat = self._body_xmat[bid]
        selpos = self._body_xpos[bid] + xmat @ self._grab_local + self._scene_offset
        target = self._target_viser.copy()
    if selpos is None or target is None:
      self._hide_connector()
      return
    points = np.array([[selpos, target]], dtype=np.float32)
    self._connector = self._server.scene.add_line_segments(
      _CONNECTOR_NAME, points=points, colors=_CONNECTOR_COLOR, line_width=3.0
    )

  def _hide_connector(self) -> None:
    if self._connector is not None:
      self._connector.remove()
      self._connector = None

  def _update_rotate_ghost(self) -> None:
    """Draw a wireframe box at the body's target orientation while rotating."""
    center = half = quat = None
    with self._lock:
      bid = self._drag_body_id
      if (
        self._drag_rotate
        and bid is not None
        and self._rotate_delta_quat is not None
        and self._rotate_world_quat is not None
        and self._rotate_aabb_center is not None
        and self._rotate_aabb_half is not None
        and self._body_xpos is not None
        and bid < self._body_xpos.shape[0]
      ):
        # Target world orientation = screen rotation applied to the body's
        # orientation at drag start.
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, self._rotate_delta_quat, self._rotate_world_quat)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, quat)
        center = (
          self._body_xpos[bid]
          + rot.reshape(3, 3) @ self._rotate_aabb_center
          + self._scene_offset
        )
        half = self._rotate_aabb_half.copy()
    if center is None or half is None or quat is None:
      self._hide_ghost()
      return
    self._ghost = self._server.scene.add_box(
      _GHOST_NAME,
      color=_GHOST_COLOR,
      dimensions=2.0 * half,
      wireframe=True,
      wxyz=quat,
      position=center,
    )

  def _hide_ghost(self) -> None:
    if self._ghost is not None:
      self._ghost.remove()
      self._ghost = None
