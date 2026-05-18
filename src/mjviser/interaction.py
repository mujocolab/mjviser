"""Interactive perturbation: click-to-select and drag-to-apply-force."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock

import mujoco
import numpy as np
import viser


@dataclass
class PerturbationState:
  """Current perturbation force to be applied by the simulation loop."""

  body_id: int
  force: np.ndarray
  torque: np.ndarray
  point: np.ndarray


class PerturbationHandler:
  """Manages body selection and drag-to-apply-force perturbation.

  Uses MuJoCo's critically damped spring model: the force is proportional
  to stiffness * localmass, with critical damping from velocity feedback.
  """

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
    self._drag_body_id: int | None = None
    self._drag_grab_local: np.ndarray | None = None
    self._drag_target: np.ndarray | None = None
    self._drag_mode: str | None = None

    self._scene_offset = np.zeros(3)
    self._body_xpos: np.ndarray | None = None
    self._body_xmat: np.ndarray | None = None

    self._info_text: viser.GuiTextHandle | None = None

    # Grab-point velocity, computed once per render frame in
    # ``update_state`` and reused across all physics substeps that
    # frame. Computing it inside ``get_perturbation`` (which runs once
    # per substep) was both wasteful and noisy: the cached body pose
    # only refreshes between render frames, so substeps 2..N within
    # a frame would see zero motion divided by a sub-millisecond
    # wall-clock dt -- damping collapsed to zero for most substeps and
    # spiked on the first one. Render-frame cadence matches the rate
    # at which the grab point actually moves on screen.
    self._grab_vel: np.ndarray = np.zeros(3)
    self._prev_grab_world: np.ndarray | None = None
    self._prev_time: float | None = None

  def setup_gui(self) -> None:
    """Add selection info display to the GUI."""
    with self._server.gui.add_folder("Selection"):
      self._info_text = self._server.gui.add_text(
        "Body", initial_value="(none)", disabled=True
      )

  def clear(self) -> None:
    """Drop selection and any in-flight drag. Call on reset or after
    the model is rebuilt -- otherwise a cached body id can outlive the
    body it pointed to (out-of-range index, silently retargets a
    different body) and the GUI label keeps showing a stale name."""
    with self._lock:
      self.selected_body_id = None
      self._drag_body_id = None
      self._drag_grab_local = None
      self._drag_target = None
      self._drag_mode = None
      self._prev_grab_world = None
      self._prev_time = None
    if self._info_text is not None:
      self._info_text.value = "(none)"

  def register_drag_handlers(
    self,
    handle: viser.BatchedGlbHandle,
    body_ids: np.ndarray,
  ) -> None:
    """Attach drag and click handlers to a batched mesh handle."""
    # Skip groups whose only body is the world body (id 0): perturbing
    # it is meaningless, and registering any handler makes the mesh
    # "interactive" in viser -- which flips the cursor to "pointer"
    # whenever the user hovers it. The world body holds the ground
    # plane, which spans the visible canvas, so the pointer cursor
    # would show across the entire scene.
    if not bool(np.any(body_ids != 0)):
      return
    n_bodies = len(body_ids)

    def _body_id_from_event(
      event: viser.SceneNodeDragEvent,  # type: ignore[type-arg]
    ) -> int | None:
      idx = event.instance_index
      if idx is None:
        return None
      return int(body_ids[idx % n_bodies])

    @handle.on_click
    def _(event: viser.SceneNodePointerEvent) -> None:  # type: ignore[type-arg]
      idx = event.instance_index
      if idx is None:
        return
      bid = int(body_ids[idx % n_bodies])
      if bid == 0:
        return
      with self._lock:
        self.selected_body_id = bid
      name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, bid)
      if self._info_text is not None:
        self._info_text.value = name or f"body_{bid}"

    def _make_drag_handler(mode: str):
      async def _handler(event: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
        if event.phase == "start":
          bid = _body_id_from_event(event)
          if bid is None or bid == 0:
            return
          # Target and grab position are tracked in viser-world frame
          # (no scene_offset subtraction) so that camera tracking,
          # which shifts scene_offset every frame, doesn't drift the
          # target between throttled mouse-move events. Body-local
          # offset still needs mujoco-frame coords for the conversion.
          target_viser = np.array(event.start_position)
          grab_world_mj = target_viser - self._scene_offset
          grab_local = self._world_to_body_local(bid, grab_world_mj)
          with self._lock:
            self._drag_body_id = bid
            self._drag_grab_local = grab_local
            self._drag_target = target_viser
            self._drag_mode = mode
            self.selected_body_id = bid
            self._prev_grab_world = target_viser.copy()
            self._prev_time = time.perf_counter()
          name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, bid)
          if self._info_text is not None:
            self._info_text.value = name or f"body_{bid}"
        elif event.phase == "update":
          with self._lock:
            if self._drag_body_id is not None:
              self._drag_target = np.array(event.end_position)
        else:  # "end"
          with self._lock:
            self._drag_body_id = None
            self._drag_grab_local = None
            self._drag_target = None
            self._drag_mode = None
            self._prev_grab_world = None
            self._prev_time = None

      return _handler

    handle.on_drag("left", modifier="cmd/ctrl")(_make_drag_handler("translate"))
    handle.on_drag("left", modifier="cmd/ctrl+shift")(_make_drag_handler("rotate"))

  def update_state(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
  ) -> None:
    """Update cached body state from the latest simulation frame.

    Args:
      body_xpos: Shape ``(num_envs, nbody, 3)``.
      body_xmat: Shape ``(num_envs, nbody, 3, 3)``.
      env_idx: Active environment index.
      scene_offset: Current camera tracking offset.
    """
    xpos_env = body_xpos[env_idx]
    xmat_env = body_xmat[env_idx]
    self._body_xpos = xpos_env
    self._body_xmat = xmat_env
    self._scene_offset = scene_offset

    # Refresh the grab-point velocity once per render frame against
    # the new pose. Substeps within the next frame all consume this
    # cached value -- see ``_grab_vel`` field comment.
    bid = self._drag_body_id
    if bid is None or self._drag_grab_local is None or bid >= xpos_env.shape[0]:
      self._grab_vel = np.zeros(3)
      self._prev_grab_world = None
      self._prev_time = None
      return
    # Velocity is measured in viser-world frame -- the same frame the
    # cursor (and therefore _drag_target) lives in -- so damping
    # actually opposes motion *relative to the target*. With camera
    # tracking on, the dragged body sits near the viser origin and
    # mujoco-frame velocity would diverge from screen-frame velocity.
    grab_world = xpos_env[bid] + xmat_env[bid] @ self._drag_grab_local + scene_offset
    now = time.perf_counter()
    if self._prev_grab_world is not None and self._prev_time is not None:
      dt = now - self._prev_time
      if dt > 1e-6:
        self._grab_vel = (grab_world - self._prev_grab_world) / dt
    self._prev_grab_world = grab_world.copy()
    self._prev_time = now

  def get_perturbation(self) -> PerturbationState | None:
    """Return the current perturbation force using MuJoCo's spring model."""
    with self._lock:
      if (
        self._drag_body_id is None
        or self._drag_grab_local is None
        or self._drag_target is None
        or self._body_xpos is None
        or self._body_xmat is None
      ):
        return None

      bid = self._drag_body_id
      # Bounds-check against the *current* model: if the model was
      # rebuilt mid-drag, the cached body id may now reference a
      # different body or be out of range entirely. Drop the drag
      # silently rather than indexing past the end.
      if bid >= self._body_xpos.shape[0] or bid >= self._model.nbody:
        self._drag_body_id = None
        self._drag_grab_local = None
        self._drag_target = None
        self._drag_mode = None
        self._prev_grab_world = None
        self._prev_time = None
        return None
      xpos = self._body_xpos[bid]
      xmat = self._body_xmat[bid]

      grab_world_mj = xpos + xmat @ self._drag_grab_local
      # Displacement is computed in viser-world frame (cursor frame).
      # Force/torque vectors are unchanged by the pure-translation
      # offset, so we apply them in mujoco frame as-is; only the
      # application point passed to mj_applyFT needs mujoco coords.
      grab_world = grab_world_mj + self._scene_offset
      diff = grab_world - self._drag_target

      # Compute localmass from body_invweight0 (translational inverse weight).
      invweight_trn = float(self._model.body_invweight0[bid, 0])
      localmass = 1.0 if invweight_trn == 0 else 1.0 / max(invweight_trn, 1e-10)

      # Velocity is refreshed per render frame in ``update_state``;
      # we just read the cached value here.
      vel = self._grab_vel

      if self._drag_mode == "translate":
        stiffness = float(self._model.vis.map.stiffness)
        # Spring force: -k * m * displacement.
        force = -stiffness * localmass * diff
        # Critical damping: -2*sqrt(k)*m * velocity.
        force -= np.sqrt(stiffness) * localmass * vel

        # Torque from moment arm (mujoco-frame coords).
        moment_arm = grab_world_mj - xpos
        torque = np.cross(moment_arm, force)
      else:
        stiffnessrot = float(self._model.vis.map.stiffnessrot)
        # Rotational inertia from body_invweight0.
        invweight_rot = float(self._model.body_invweight0[bid, 1])
        inertia = 1.0 if invweight_rot == 0 else 1.0 / max(invweight_rot, 1e-10)
        # Torque proportional to displacement (simplified from quat error).
        torque = -stiffnessrot * inertia * diff
        torque -= np.sqrt(stiffnessrot) * inertia * vel
        # Smaller translational force in rotate mode.
        stiffness = float(self._model.vis.map.stiffness)
        force = -stiffness * localmass * diff * 0.1

      return PerturbationState(
        body_id=bid,
        force=force,
        torque=torque,
        point=grab_world_mj,
      )

  def _world_to_body_local(self, body_id: int, world_pos: np.ndarray) -> np.ndarray:
    """Convert a world position to body-local coordinates."""
    if self._body_xpos is None or self._body_xmat is None:
      return world_pos
    xpos = self._body_xpos[body_id]
    xmat = self._body_xmat[body_id]
    return xmat.T @ (world_pos - xpos)
