"""Record a simulation and scrub through it with a timeline.

Records the humanoid for a few seconds, then lets you scrub
through the recording with a timeline slider, play/pause, speed
control, and looping. Contact points and forces work at any
frame. Shows how to build a custom viewer on top of
``ViserMujocoScene`` with your own GUI and loop.

uv run examples/motion_playback.py
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
import viser

from mjviser import ViserMujocoScene

RECORD_SECONDS = 5.0


def _record(
  model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Simulate and record joint state. Returns (qpos, qvel, dt)."""
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  dt = model.opt.timestep
  n_steps = int(RECORD_SECONDS / dt)
  qpos_traj = np.zeros((n_steps, model.nq))
  qvel_traj = np.zeros((n_steps, model.nv))

  for i in range(n_steps):
    qpos_traj[i] = data.qpos
    qvel_traj[i] = data.qvel
    mujoco.mj_step(model, data)

  return qpos_traj, qvel_traj, dt


def main() -> None:
  path = Path(__file__).parent / "humanoid.xml"
  model = mujoco.MjModel.from_xml_path(str(path))

  print(f"Recording {RECORD_SECONDS}s of simulation...")
  qpos_traj, qvel_traj, dt = _record(model)
  n_frames = len(qpos_traj)
  duration = n_frames * dt
  print(f"Recorded {n_frames} frames ({duration:.1f}s).")

  server = viser.ViserServer()
  scene = ViserMujocoScene(server, model, num_envs=1)

  # Scratch MjData for replaying any frame (recomputes contacts).
  replay_data = mujoco.MjData(model)

  # State.
  frame_idx = [0]
  playing = [True]
  speed = [1.0]
  looping = [True]
  accumulator = [0.0]
  needs_render = [True]

  # GUI: standard scene controls plus a Playback tab.
  tabs = scene.create_visualization_gui()
  with tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY):
    timeline = server.gui.add_slider(
      "Frame",
      min=0,
      max=n_frames - 1,
      step=1,
      initial_value=0,
    )
    time_label = server.gui.add_html("")

    play_btn = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)

    @play_btn.on_click
    def _(_) -> None:
      playing[0] = not playing[0]
      play_btn.label = "Pause" if playing[0] else "Play"
      play_btn.icon = viser.Icon.PLAYER_PAUSE if playing[0] else viser.Icon.PLAYER_PLAY

    speed_btns = server.gui.add_button_group(
      "Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"]
    )

    @speed_btns.on_click
    def _(event) -> None:
      speed[0] = float(event.target.value.replace("x", ""))

    loop_cb = server.gui.add_checkbox("Loop", initial_value=True)

    @loop_cb.on_update
    def _(_) -> None:
      looping[0] = loop_cb.value

    @timeline.on_update
    def _(_) -> None:
      frame_idx[0] = int(timeline.value)
      needs_render[0] = True

  def render_frame(idx: int) -> None:
    replay_data.qpos[:] = qpos_traj[idx]
    replay_data.qvel[:] = qvel_traj[idx]
    mujoco.mj_forward(model, replay_data)
    scene.update_from_mjdata(replay_data)
    t = idx * dt
    time_label.content = (
      f'<span style="font-size:0.85em">'
      f"{t:.2f}s / {duration:.2f}s (frame {idx}/{n_frames - 1})"
      f"</span>"
    )

  render_frame(0)

  last_time = time.perf_counter()
  try:
    while True:
      now = time.perf_counter()
      wall_dt = now - last_time
      last_time = now

      if playing[0]:
        accumulator[0] += wall_dt * speed[0]
        frames_to_advance = int(accumulator[0] / dt)
        if frames_to_advance > 0:
          accumulator[0] -= frames_to_advance * dt
          new_idx = frame_idx[0] + frames_to_advance
          if new_idx >= n_frames:
            if looping[0]:
              new_idx = new_idx % n_frames
            else:
              new_idx = n_frames - 1
              playing[0] = False
          frame_idx[0] = new_idx
          timeline.value = new_idx
          render_frame(new_idx)
      elif needs_render[0]:
        render_frame(frame_idx[0])
        needs_render[0] = False

      time.sleep(1.0 / 60.0)
  except KeyboardInterrupt:
    print("\nStopped.")
    server.stop()


if __name__ == "__main__":
  main()
