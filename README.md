# mjviser

Web-based MuJoCo viewer powered by [Viser](https://github.com/viser-project/viser).

## Quick start

View any MuJoCo model instantly, no install needed:

```bash
uvx mjviser path/to/model.xml
```

Or install and use:

```bash
pip install mjviser
mjviser path/to/model.xml
```

Fuzzy path matching finds models in the current directory tree:

```bash
mjviser humanoid        # finds **/humanoid*.xml
mjviser shadow_hand     # finds **/shadow_hand*.xml
```

When there's a single match it's used automatically. Multiple matches show a numbered list to pick from.

## Python API

```python
import mujoco
from mjviser import Viewer

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
Viewer(model, data).run()
```

Open the printed URL in your browser. Most of the MuJoCo native viewer's functionality is available: simulation controls, joint and actuator sliders, contact and force visualization, camera tracking, keyframes, and more.

## Extension points

The `Viewer` accepts three callbacks for injecting custom logic:

- **`step_fn(model, data)`**: Called each simulation step. Use this to apply a controller, external forces, or any per-step logic. Defaults to `mujoco.mj_step`.

- **`render_fn(scene)`**: Called each render frame. Use this to push custom state to the scene, add ghost overlays, or render debug geometry. Defaults to `scene.update_from_mjdata(data)`.

- **`reset_fn(model, data)`**: Called on reset. Use this to restore custom simulation state.

For full control over the loop, use `ViserMujocoScene` directly. The `server` is a standard [viser](https://viser.studio) server, so you can add custom GUI elements, scene overlays, or anything else viser supports.

```python
server = viser.ViserServer()
scene = ViserMujocoScene(server, model, num_envs=1)
scene.create_visualization_gui()

# Add your own GUI.
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

## Limitations

- **No mouse interaction**: clicking/dragging bodies and keyboard callbacks are not yet supported; this requires upstream support in viser.
- **Performance with many bodies**: models with lots of independently-moving bodies can be slower than the native MuJoCo viewer due to per-body websocket message overhead.
- **Cubemap textures**: approximated via per-vertex colors rather than true cubemap rendering.
