# Development Workflow

**Always use `uv run`, not python**.

```sh
make format     # Format and lint (ruff)
make type       # Type check (pyright)
make check      # make format && make type
make test       # Run tests
```

Always run `make check` before committing.

# Architecture

mjviser has two layers:

- **`ViserMujocoScene`** (scene.py): Rendering foundation. Creates geometry from an `MjModel`, updates positions from `MjData` or numpy arrays. Handles camera tracking, contacts, geom/site groups. No simulation logic, no debug visualization.

- **`Viewer`** (viewer.py): Active viewer built on top of the scene. Owns the simulation loop with realtime pacing, playback controls (pause/step/reset/speed), keyframes, joint/actuator sliders, physics flags, inertia visualization, coordinate frames. Accepts `step_fn`, `render_fn`, `reset_fn` callbacks for customization.

`conversions.py` contains MuJoCo-to-trimesh conversion utilities (mesh extraction, textures, cube map projection, primitives, heightfields).

The `mjviser` CLI entry point (`__main__.py`) is a thin wrapper that loads a model and calls `Viewer(model, data).run()`.

# Relationship to mjlab

mjlab depends on mjviser. mjlab's `MjlabViserScene` subclasses `ViserMujocoScene` to add debug visualization (arrows, ghosts, spheres) and the mjwarp adapter. Debug visualization is an mjlab concept and does not belong in mjviser.

When changing `ViserMujocoScene`'s public API or `create_visualization_gui`/`create_groups_gui` signatures, check that mjlab's `MjlabViserScene` subclass still works.

# Tests

- Use functions and fixtures, no classes.
- Inline XML models for fixtures when possible (no external file dependencies).
- Scene tests can construct a `ViserMujocoScene` with `viser.ViserServer(port=0)` without a browser.

# Style

- Line length limit is 88 columns.
- Avoid local imports unless strictly necessary.
- PR body should be plain, concise prose. No section headers, checklists, or structured templates.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them at 88 columns. Let each sentence flow on one line.
