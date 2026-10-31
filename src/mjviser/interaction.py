"""MuJoCo-style body selection and perturbation, driven by viser input.

This behaves like MuJoCo's simulate. MuJoCo's own ``mjv_*`` functions set
the reference pose, move it with the mouse, and turn it into forces or poses;
this module only feeds them viser's pointer events.

  Click, or Cmd/Ctrl + click   select a body (click empty space to deselect)
  Cmd/Ctrl + right-drag        move it in the vertical plane facing the camera
  Cmd/Ctrl + Shift + right     move it in the horizontal plane
  Cmd/Ctrl + left-drag         rotate it
  Cmd/Ctrl + Shift + left      rotate it about the other axes

A drag acts on the selected body, wherever it starts; with nothing selected,
it grabs the body under the cursor. Shift can be pressed or released
mid-drag. While running, the body is pulled by a critically damped spring.
While paused, it is posed directly: mocap and free bodies move, and grabbing
any body of a floating base moves the base.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock
from typing import Literal, NamedTuple

import mujoco
import numpy as np
import viser

_TRANSLATE = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
_ROTATE = int(mujoco.mjtPertBit.mjPERT_ROTATE)

# (button, modifier, perturbation, mouse action), matching simulate.
_DragBinding = tuple[
  Literal["left", "right"], Literal["cmd/ctrl", "cmd/ctrl+shift"], int, int
]
_DRAG_BINDINGS: tuple[_DragBinding, ...] = (
  ("right", "cmd/ctrl", _TRANSLATE, int(mujoco.mjtMouse.mjMOUSE_MOVE_V)),
  ("right", "cmd/ctrl+shift", _TRANSLATE, int(mujoco.mjtMouse.mjMOUSE_MOVE_H)),
  ("left", "cmd/ctrl", _ROTATE, int(mujoco.mjtMouse.mjMOUSE_ROTATE_V)),
  ("left", "cmd/ctrl+shift", _ROTATE, int(mujoco.mjtMouse.mjMOUSE_ROTATE_H)),
)

# viser splits a drag into segments when the held modifier changes: the old
# segment ends and the new one starts about 100 ms later. A start this soon
# after an end of the same kind continues the perturbation instead of
# resetting it, so toggling Shift mid-drag is seamless.
_RESUME_SECONDS = 0.3

_Update = Callable[[mujoco.MjData], None]


class _View(NamedTuple):
  """A client's camera, in viser world coordinates."""

  pos: np.ndarray
  forward: np.ndarray
  up: np.ndarray
  fovy: float

  @staticmethod
  def of(client: viser.ClientHandle) -> _View:
    cam = client.camera
    pos = np.asarray(cam.position, dtype=float)
    forward = np.asarray(cam.look_at, dtype=float) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, cam.up_direction)
    up = np.cross(right / np.linalg.norm(right), forward)
    return _View(pos, forward, up, cam.fov)


class Perturbation:
  """Selects bodies and perturbs them with the mouse.

  Pointer callbacks only queue work. ``step`` runs it on the thread that owns
  the ``MjData``, then applies the perturbation, like simulate does once per
  physics step.
  """

  def __init__(self, server: viser.ViserServer, model: mujoco.MjModel) -> None:
    self.pert = mujoco.MjvPerturb()
    # Kept current by the scene: the viser-minus-MuJoCo world offset (from
    # camera tracking), and which geom groups are visible for picking.
    self.offset = np.zeros(3)
    self.geomgroup = np.ones(6, dtype=np.uint8)

    self._server = server
    self._model = model
    self._scn = mujoco.MjvScene(model, maxgeom=1)  # Only its camera is used.
    self._lock = Lock()
    self._pending: list[_Update] = []
    self._screen_pos = np.zeros(2)
    self._last_end = (-np.inf, 0)  # (time, perturbation kind)
    self._forced_body = 0  # Body whose xfrc_applied we last wrote.
    self._label: viser.GuiTextHandle | None = None

    def on_click(event: viser.SceneClickEvent) -> None:
      origin = np.asarray(event.ray_origin, dtype=float)
      direction = np.asarray(event.ray_direction, dtype=float)
      self._defer(lambda data: self._select(data, origin, direction))

    # A plain click only counts if the pointer barely moves, since viser lets
    # the camera orbit from the first pixel. Cmd/Ctrl locks the camera on a
    # body (it matches the drag bindings), so Cmd/Ctrl+click is reliable.
    server.scene.on_click()(on_click)
    server.scene.on_click(modifier="cmd/ctrl")(on_click)

  def setup_gui(self) -> None:
    """Show the selected body's name in the current GUI context."""
    self._label = self._server.gui.add_text(
      "Selected", initial_value="(none)", disabled=True
    )

  def attach(self, handle: viser.BatchedGlbHandle | viser.GlbHandle) -> None:
    """Make a body mesh draggable."""
    for button, modifier, kind, action in _DRAG_BINDINGS:
      handle.on_drag(button, modifier=modifier)(self._drag_callback(kind, action))

  def clear(self) -> None:
    """Drop the selection, e.g. after a reset or a model rebuild."""
    with self._lock:
      self._pending.clear()
    self.pert.active = 0
    self.pert.select = 0
    self._forced_body = 0
    self._show_label(0)

  def step(self, data: mujoco.MjData, paused: bool) -> bool:
    """Run queued input, then apply the perturbation to ``data``.

    Call before every physics step while running, and every frame while
    paused. Returns True if anything changed that needs a redraw.
    """
    with self._lock:
      pending, self._pending = self._pending, []
    for update in pending:
      update(data)

    # Clear the force we applied last step. (simulate zeroes all of
    # xfrc_applied; we only touch our own entry.)
    if self._forced_body:
      data.xfrc_applied[self._forced_body] = 0
      self._forced_body = 0
    if not self.pert.active:
      return bool(pending)

    mujoco.mjv_applyPerturbPose(self._model, data, self.pert, int(paused))
    if paused:
      mujoco.mj_forward(self._model, data)
    else:
      mujoco.mjv_applyPerturbForce(self._model, data, self.pert)
      self._forced_body = self.pert.select
    return True

  # Input. These run on viser's event loop and only queue work.

  def _defer(self, update: _Update) -> None:
    with self._lock:
      self._pending.append(update)

  def _drag_callback(self, kind: int, action: int):
    # async, so viser delivers start/update/end in order.
    async def callback(event: viser.SceneNodeDragEvent) -> None:
      view = _View.of(event.client)
      screen_pos = np.asarray(event.end_screen_pos, dtype=float)
      if event.phase == "start":
        grab = np.asarray(event.start_position, dtype=float)
        now = time.monotonic()
        self._defer(lambda data: self._start(data, kind, view, grab, now))
      elif event.phase == "update":
        # Motion relative to the viewport height, as simulate passes it.
        # (simulate negates a y-up delta; viser's screen y already points
        # down, so it needs no sign flip.)
        dx, dy = screen_pos - self._screen_pos
        reldx, reldy = dx * event.client.camera.aspect, dy
        self._defer(lambda data: self._move(data, action, view, reldx, reldy))
      else:
        now = time.monotonic()
        self._defer(lambda _: self._end(kind, now))
      self._screen_pos = screen_pos

    return callback

  # Work. These run in step(), on the thread that owns the MjData.

  def _select(self, data: mujoco.MjData, origin, direction) -> None:
    self.pert.active = 0
    self._set_selection(data, *self._pick(data, origin, direction))

  def _start(self, data: mujoco.MjData, kind: int, view: _View, grab, now) -> None:
    end_time, end_kind = self._last_end
    if self.pert.select > 0 and kind == end_kind and now - end_time < _RESUME_SECONDS:
      self.pert.active = kind
      return
    # Like simulate, a drag acts on the selected body wherever it starts. With
    # nothing selected, it grabs the body under the cursor.
    if self.pert.select <= 0:
      body, point = self._pick(data, view.pos, grab - view.pos)
      if body <= 0:
        return
      self._set_selection(data, body, point)
    self._set_camera(view)
    mujoco.mjv_initPerturb(self._model, data, self._scn, self.pert)
    self.pert.active = kind

  def _move(self, data: mujoco.MjData, action: int, view: _View, reldx, reldy) -> None:
    if self.pert.active:
      self._set_camera(view)
      mujoco.mjv_movePerturb(
        self._model, data, action, reldx, reldy, self._scn, self.pert
      )

  def _end(self, kind: int, now: float) -> None:
    self.pert.active = 0
    self._last_end = (now, kind)

  def _pick(self, data: mujoco.MjData, origin, direction) -> tuple[int, np.ndarray]:
    """Cast a ray from viser world coordinates, like mjv_select.

    Returns the body hit and the hit point in MuJoCo world coordinates. A miss
    returns body 0 (the world).
    """
    origin = origin - self.offset
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(
      self._model, data, origin, direction, self.geomgroup, 1, -1, geomid
    )
    if dist < 0:
      return 0, origin
    return int(self._model.geom_bodyid[geomid[0]]), origin + dist * direction

  def _set_selection(self, data: mujoco.MjData, body: int, point: np.ndarray) -> None:
    self.pert.select = body
    if body > 0:
      xmat = data.xmat[body].reshape(3, 3)
      self.pert.localpos = xmat.T @ (point - data.xpos[body])
    self._show_label(body)

  def _set_camera(self, view: _View) -> None:
    """Point the mjvScene camera (both eyes) where the client is looking."""
    half_height = np.tan(view.fovy / 2)
    for i in range(2):
      cam = self._scn.camera[i]
      cam.pos = view.pos - self.offset
      cam.forward = view.forward
      cam.up = view.up
      cam.frustum_near = 1.0
      cam.frustum_top = half_height
      cam.frustum_bottom = -half_height

  def _show_label(self, body: int) -> None:
    if self._label is None:
      return
    name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body)
    self._label.value = (name or f"body {body}") if body > 0 else "(none)"
