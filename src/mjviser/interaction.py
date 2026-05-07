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

    self._prev_grab_world: np.ndarray | None = None
    self._prev_time: float | None = None

  def setup_gui(self) -> None:
    """Add selection info display to the GUI."""
    with self._server.gui.add_folder("Selection"):
      self._info_text = self._server.gui.add_text(
        "Body", initial_value="(none)", disabled=True
      )

  def register_click_handler(self) -> None:
    """Register scene-level click handler for body selection via ray cast."""

    @self._server.scene.on_click()
    def _(event: viser.SceneClickEvent) -> None:
      origin = np.array(event.ray_origin) - self._scene_offset
      direction = np.array(event.ray_direction)
      geomid = np.zeros(1, dtype=np.int32)
      dist = mujoco.mj_ray(
        self._model,
        self._data,
        origin,
        direction,
        None,
        1,
        -1,
        geomid,
      )
      if dist < 0:
        with self._lock:
          self.selected_body_id = None
        if self._info_text is not None:
          self._info_text.value = "(none)"
        return

      body_id = int(self._model.geom_bodyid[geomid[0]])
      with self._lock:
        self.selected_body_id = body_id
      name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
      if self._info_text is not None:
        self._info_text.value = name or f"body_{body_id}"

  def register_drag_handlers(
    self,
    handle: viser.BatchedGlbHandle,
    body_ids: np.ndarray,
  ) -> None:
    """Attach drag and click handlers to a batched mesh handle."""
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

    @handle.on_drag_start("left", modifier="cmd/ctrl")
    async def _(event: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      bid = _body_id_from_event(event)
      if bid is None or bid == 0:
        return
      grab_world = np.array(event.start_position) - self._scene_offset
      grab_local = self._world_to_body_local(bid, grab_world)
      with self._lock:
        self._drag_body_id = bid
        self._drag_grab_local = grab_local
        self._drag_target = grab_world
        self._drag_mode = "translate"
        self.selected_body_id = bid
        self._prev_grab_world = grab_world.copy()
        self._prev_time = time.perf_counter()
      name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, bid)
      if self._info_text is not None:
        self._info_text.value = name or f"body_{bid}"

    @handle.on_drag_update("left", modifier="cmd/ctrl")
    async def _(event: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      with self._lock:
        if self._drag_body_id is not None:
          self._drag_target = np.array(event.end_position) - self._scene_offset

    @handle.on_drag_end("left", modifier="cmd/ctrl")
    async def _(_: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      with self._lock:
        self._drag_body_id = None
        self._drag_grab_local = None
        self._drag_target = None
        self._drag_mode = None
        self._prev_grab_world = None
        self._prev_time = None

    @handle.on_drag_start("left", modifier="cmd/ctrl+shift")
    async def _(event: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      bid = _body_id_from_event(event)
      if bid is None or bid == 0:
        return
      grab_world = np.array(event.start_position) - self._scene_offset
      grab_local = self._world_to_body_local(bid, grab_world)
      with self._lock:
        self._drag_body_id = bid
        self._drag_grab_local = grab_local
        self._drag_target = grab_world
        self._drag_mode = "rotate"
        self.selected_body_id = bid
        self._prev_grab_world = grab_world.copy()
        self._prev_time = time.perf_counter()
      name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, bid)
      if self._info_text is not None:
        self._info_text.value = name or f"body_{bid}"

    @handle.on_drag_update("left", modifier="cmd/ctrl+shift")
    async def _(event: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      with self._lock:
        if self._drag_body_id is not None:
          self._drag_target = np.array(event.end_position) - self._scene_offset

    @handle.on_drag_end("left", modifier="cmd/ctrl+shift")
    async def _(_: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      with self._lock:
        self._drag_body_id = None
        self._drag_grab_local = None
        self._drag_target = None
        self._drag_mode = None
        self._prev_grab_world = None
        self._prev_time = None

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
    self._body_xpos = body_xpos[env_idx]
    self._body_xmat = body_xmat[env_idx]
    self._scene_offset = scene_offset

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
      xpos = self._body_xpos[bid]
      xmat = self._body_xmat[bid]

      grab_world = xpos + xmat @ self._drag_grab_local
      diff = grab_world - self._drag_target

      # Compute localmass from body_invweight0 (translational inverse weight).
      invweight_trn = float(self._model.body_invweight0[bid, 0])
      localmass = 1.0 if invweight_trn == 0 else 1.0 / max(invweight_trn, 1e-10)

      # Estimate velocity of the grab point via finite difference.
      now = time.perf_counter()
      vel = np.zeros(3)
      if self._prev_grab_world is not None and self._prev_time is not None:
        dt = now - self._prev_time
        if dt > 1e-6:
          vel = (grab_world - self._prev_grab_world) / dt
      self._prev_grab_world = grab_world.copy()
      self._prev_time = now

      if self._drag_mode == "translate":
        stiffness = float(self._model.vis.map.stiffness)
        # Spring force: -k * m * displacement.
        force = -stiffness * localmass * diff
        # Critical damping: -2*sqrt(k)*m * velocity.
        force -= np.sqrt(stiffness) * localmass * vel

        # Torque from moment arm.
        moment_arm = grab_world - xpos
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
        point=grab_world,
      )

  def _world_to_body_local(self, body_id: int, world_pos: np.ndarray) -> np.ndarray:
    """Convert a world position to body-local coordinates."""
    if self._body_xpos is None or self._body_xmat is None:
      return world_pos
    xpos = self._body_xpos[body_id]
    xmat = self._body_xmat[body_id]
    return xmat.T @ (world_pos - xpos)
