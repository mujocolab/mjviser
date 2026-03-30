# mjviser

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjviser/ci.yml?branch=main)](https://github.com/mujocolab/mjviser/actions/workflows/ci.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/mujocolab/mjviser)](https://github.com/mujocolab/mjviser/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mjviser)](https://pypi.org/project/mjviser/)

A web-based MuJoCo viewer built on [Viser](https://github.com/viser-project/viser).

## Quick start

Run it directly with `uvx` (nothing to install):

```bash
uvx mjviser path/to/model.xml
```

Or `pip install mjviser` and run `mjviser path/to/model.xml`.

mjviser also does fuzzy path matching against the current directory:

```bash
mjviser humanoid        # finds **/humanoid*.xml
mjviser shadow_hand     # finds **/shadow_hand*.xml
```

If [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py) is available, you can load any of its 57 MuJoCo models by name:

```bash
uvx --with robot_descriptions mjviser go1
```

> [!NOTE]
> `uvx` defaults to the system Python and may not respect `requires-python` constraints ([astral-sh/uv#8206](https://github.com/astral-sh/uv/issues/8206)). If your system Python is 3.14+, where MuJoCo can't build yet, pass `-p 3.13` explicitly: `uvx -p 3.13 mjviser path/to/model.xml`.

## Python API

```python
import mujoco
from mjviser import Viewer

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
Viewer(model, data).run()
```

Open the printed URL in your browser. You get most of what the native MuJoCo viewer offers: simulation controls, joint and actuator sliders, contact and force visualization, camera tracking, keyframes, and more.

## Extension points

`Viewer` accepts three optional callbacks:

- **`step_fn(model, data)`**: called each simulation step. Defaults to `mujoco.mj_step`.
- **`render_fn(scene)`**: called each render frame. Defaults to `scene.update_from_mjdata(data)`.
- **`reset_fn(model, data)`**: called on reset.

For full control, use `ViserMujocoScene` directly. The `server` is a standard [Viser](https://viser.studio) server, so you can add GUI elements, scene overlays, or anything else Viser supports.

```python
server = viser.ViserServer()
scene = ViserMujocoScene(server, model, num_envs=1)
scene.create_visualization_gui()

with server.gui.add_folder("My Controls"):
    slider = server.gui.add_slider("Force", min=0, max=100, initial_value=0)

while True:
    mujoco.mj_step(model, data)
    scene.update_from_mjdata(data)
```

## Examples

- `active_viewer.py`: simplest usage with playback controls
- `active_viewer_with_controller.py`: custom `step_fn` with random torques
- `passive_viewer.py`: manual simulation loop with `ViserMujocoScene`
- `multi_env.py`: 4 humanoids in parallel via mujoco-warp
- `ghost_overlay.py`: custom `render_fn` that overlays a time-delayed ghost
- `motion_playback.py`: recorded trajectory with timeline scrubber, speed control, and contact replay

## Acknowledgments

Thanks to [Matija Kecman](https://github.com/okmatija) for early feedback and suggestions.

## Limitations

- **No mouse interaction**: clicking/dragging bodies and keyboard callbacks require upstream Viser support.
