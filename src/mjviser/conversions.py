"""Convert MuJoCo mesh data to trimesh format with texture support."""

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj
from PIL import Image

# ------------------------------------------------------------------
# Color / texture helpers
# ------------------------------------------------------------------


def _get_texture_id(mj_model: mujoco.MjModel, matid: int) -> int:
  """Return the RGB or RGBA texture ID for a material, or -1."""
  texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
  if texid < 0:
    texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
  return texid


def _get_texture_normalmap_id(mj_model: mujoco.MjModel, matid: int) -> int:
  """Returns the normalmap texture ID for a material, or -1."""
  return int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_NORMAL)])


def _has_alpha(image: Image.Image) -> bool:
  """Return True if the image has an alpha channel with any transparent pixel."""
  if image.mode != "RGBA":
    return False
  return bool(np.asarray(image.getchannel("A")).min() < 255)


def _is_cubemap_texture(mj_model: mujoco.MjModel, texid: int) -> bool:
  """Return True if texid is a cube map stored as 6 stacked square faces."""
  if int(mj_model.tex_type[texid]) != int(mujoco.mjtTexture.mjTEXTURE_CUBE):
    return False
  w = int(mj_model.tex_width[texid])
  h = int(mj_model.tex_height[texid])
  nc = int(mj_model.tex_nchannel[texid])
  return h == w * 6 and nc in (1, 3, 4)


def _is_2d_texture_supported(mj_model: mujoco.MjModel, texid: int) -> bool:
  """Return True if texid is a 2D texture with a channel count we can extract."""
  return int(mj_model.tex_nchannel[texid]) in (1, 3, 4)


# MuJoCo stores cube faces in the order right, left, up, down, front, back and
# uploads them to GL_TEXTURE_CUBE_MAP_POSITIVE_X + i. For a geom (regular) cube
# texture MuJoCo samples with texcoords = the geom-local position (x, y, z), so
# in the geom frame the faces sit on +X, -X, +Y, -Y, +Z, -Z respectively.
_CUBEMAP_AXES: np.ndarray = np.array(
  [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
  dtype=np.float64,
)


def _extract_texture_image(
  mj_model: mujoco.MjModel, texid: int, flip: bool = True
) -> Image.Image | None:
  """Extract a 2D texture as a PIL Image, or None for unsupported types.

  Color textures use MuJoCo's bottom-left origin while GLTF expects top-left,
  so they are flipped vertically by default. Normal maps are stored without
  that flip (they are typically authored with `vflip="false"` while color maps
  use `vflip="true"`), so pass flip=False to keep them aligned with the albedo
  texture and the shared UVs.
  """
  w = mj_model.tex_width[texid]
  h = mj_model.tex_height[texid]
  nc = mj_model.tex_nchannel[texid]
  adr = mj_model.tex_adr[texid]
  data = mj_model.tex_data[adr : adr + w * h * nc]

  if nc == 1:
    arr = data.reshape(h, w)
  elif nc in (3, 4):
    arr = data.reshape(h, w, nc)
  else:
    return None

  if flip:
    arr = np.flipud(arr)
  mode = "L" if nc == 1 else None
  return Image.fromarray(arr.astype(np.uint8), mode=mode)


def _resolve_flat_rgba(mj_model: mujoco.MjModel, geom_idx: int) -> np.ndarray:
  """Resolve the flat RGBA color for a geom as uint8.

  Priority: material rgba > geom rgba.
  """
  matid = mj_model.geom_matid[geom_idx]
  if matid >= 0 and matid < mj_model.nmat:
    rgba = mj_model.mat_rgba[matid]
  else:
    rgba = mj_model.geom_rgba[geom_idx]
  return (np.clip(rgba, 0, 1) * 255).astype(np.uint8)


def _apply_flat_color(mesh: trimesh.Trimesh, rgba_uint8: np.ndarray) -> None:
  """Apply a uniform RGBA color to all vertices."""
  mesh.visual = trimesh.visual.ColorVisuals(
    vertex_colors=np.tile(rgba_uint8, (len(mesh.vertices), 1))
  )


def _cubemap_vertex_colors(
  mj_model: mujoco.MjModel,
  matid: int,
  vertices: np.ndarray,
  faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  """Project a cube map texture onto mesh triangles by normal direction.

  For each triangle, finds the cube map face most aligned with its
  normal and assigns that face's average color. If the best face is
  empty, falls back to the nearest non-empty face (handles chamfered
  edges).

  Returns (new_vertices, new_faces, vertex_colors) with duplicated
  vertices so each triangle can have its own color, or None if the
  material has no cube map texture.
  """
  texid = _get_texture_id(mj_model, matid)
  if texid < 0 or not _is_cubemap_texture(mj_model, texid):
    return None

  w = mj_model.tex_width[texid]
  h = mj_model.tex_height[texid]
  nc = mj_model.tex_nchannel[texid]
  adr = mj_model.tex_adr[texid]
  data = mj_model.tex_data[adr : adr + w * h * nc].reshape(6, w, w, nc)

  # Average color per cube face. Empty faces stay at [0,0,0,0].
  face_colors = np.zeros((6, 4), dtype=np.uint8)
  has_color = np.zeros(6, dtype=bool)
  for i in range(6):
    mask = data[i, :, :, : min(nc, 3)].sum(axis=2) > 0
    if mask.any():
      avg = data[i][mask].mean(axis=0).astype(np.uint8)
      face_colors[i, :nc] = avg[:nc]
      if nc < 4:
        face_colors[i, 3] = 255
      has_color[i] = True

  if not has_color.any():
    return None

  # Per-triangle normals.
  v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
  normals = np.cross(v1 - v0, v2 - v0)
  normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)

  # For each triangle, pick the best aligned face that has color.
  dots = normals @ _CUBEMAP_AXES.T
  ranked = np.argsort(-dots, axis=1)
  nf = len(faces)
  valid = has_color[ranked]  # (nf, 6) bool
  first_valid = valid.argmax(axis=1)  # First True per row.
  best_face = ranked[np.arange(nf), first_valid]
  tri_colors = face_colors[best_face]
  # Zero out rows where no face has color.
  tri_colors[~valid.any(axis=1)] = 0

  # Duplicate vertices so each triangle gets its own color.
  new_verts = vertices[faces.ravel()]
  new_faces = np.arange(nf * 3).reshape(-1, 3)
  vert_colors = np.repeat(tri_colors, 3, axis=0)

  return new_verts, new_faces, vert_colors


# ------------------------------------------------------------------
# Mesh extraction
# ------------------------------------------------------------------


def _extract_mesh_data(
  mj_model: mujoco.MjModel, geom_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Extract vertices, faces, and optional UVs for a mesh geom.

  Returns (vertices, faces, uvs). UVs are None when the mesh has no
  texture coordinates. When UVs are present, vertices and faces are
  duplicated so each face-vertex gets its own UV.
  """
  mesh_id = mj_model.geom_dataid[geom_idx]
  vert_start = int(mj_model.mesh_vertadr[mesh_id])
  vert_count = int(mj_model.mesh_vertnum[mesh_id])
  face_start = int(mj_model.mesh_faceadr[mesh_id])
  face_count = int(mj_model.mesh_facenum[mesh_id])

  vertices = mj_model.mesh_vert[vert_start : vert_start + vert_count]
  faces = mj_model.mesh_face[face_start : face_start + face_count]

  texcoord_num = mj_model.mesh_texcoordnum[mesh_id]
  if texcoord_num == 0:
    return vertices, faces, None

  texcoord_adr = mj_model.mesh_texcoordadr[mesh_id]
  texcoords = mj_model.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_num]
  face_tc_idx = mj_model.mesh_facetexcoord[face_start : face_start + face_count]

  # Duplicate vertices so each face-vertex gets its own UV.
  new_verts = vertices[faces.flatten()]
  new_uvs = texcoords[face_tc_idx.flatten()]
  new_faces = np.arange(face_count * 3).reshape(-1, 3)

  return new_verts, new_faces, new_uvs


def mujoco_mesh_to_trimesh(mj_model: mujoco.MjModel, geom_idx: int) -> trimesh.Trimesh:
  """Convert a MuJoCo mesh geometry to a trimesh with visual appearance.

  Color resolution order:
    1. 2D texture with UVs (PBR material with baseColorTexture)
    2. Cube map texture projected by triangle normals
    3. Flat material color (mat_rgba)
    4. Flat geom color (geom_rgba)
  """
  vertices, faces, uvs = _extract_mesh_data(mj_model, geom_idx)
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

  matid = mj_model.geom_matid[geom_idx]

  # Path 1: mesh has UVs and material has a 2D texture.
  if uvs is not None and matid >= 0:
    texid_albedo = _get_texture_id(mj_model, matid)
    texid_normalmap = _get_texture_normalmap_id(mj_model, matid)

    image_albedo: Image.Image | None = None
    image_normalmap: Image.Image | None = None
    if texid_albedo >= 0:
      image_albedo = _extract_texture_image(mj_model, texid_albedo)
    if texid_normalmap >= 0:
      image_normalmap = _extract_texture_image(mj_model, texid_normalmap, flip=False)

    if image_albedo is not None:
      rgba = mj_model.mat_rgba[matid]
      geom_rgba = mj_model.geom_rgba[geom_idx]
      # Blend when the geom/material is translucent or the texture itself has
      # transparent pixels (e.g. cut-out decals stored in the alpha channel).
      use_blending = rgba[-1] < 0.99 or geom_rgba[-1] < 0.99 or _has_alpha(image_albedo)
      material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=rgba,
        baseColorTexture=image_albedo,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        normalTexture=image_normalmap,
        alphaMode="BLEND" if use_blending else "OPAQUE",
      )
      mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
      return mesh

  # Path 2: no UVs, try cube map projection.
  if uvs is None and matid >= 0:
    result = _cubemap_vertex_colors(mj_model, matid, vertices, faces)
    if result is not None:
      new_verts, new_faces, vert_colors = result
      mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
      mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vert_colors)
      return mesh

  # Path 3/4: flat color from material or geom.
  _apply_flat_color(mesh, _resolve_flat_rgba(mj_model, geom_idx))
  return mesh


def _create_shape_mesh(geom_type: int, size: np.ndarray) -> trimesh.Trimesh:
  """Create an uncolored mesh for a standard shape type."""
  if geom_type == mjtGeom.mjGEOM_SPHERE:
    return trimesh.creation.icosphere(radius=size[0], subdivisions=2)
  elif geom_type == mjtGeom.mjGEOM_BOX:
    return trimesh.creation.box(extents=2.0 * size)
  elif geom_type == mjtGeom.mjGEOM_CAPSULE:
    return trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_CYLINDER:
    return trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_ELLIPSOID:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.apply_scale(size)
    return mesh
  raise ValueError(f"Unsupported shape type: {geom_type}")


def create_primitive_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Create a mesh for primitive geom types.

  Supports sphere, box, capsule, cylinder, plane, ellipsoid, and
  heightfield.
  """
  geom_type = mj_model.geom_type[geom_id]

  if geom_type == mjtGeom.mjGEOM_HFIELD:
    return _create_heightfield_mesh(mj_model, geom_id)

  # Textured primitives (box/sphere/ellipsoid cube maps, plane 2D textures) are
  # dispatched in one place so this and get_geom_texture_id stay in sync.
  textured = _textured_primitive_mesh(mj_model, geom_id)
  if textured is not None:
    return textured

  if geom_type == mjtGeom.mjGEOM_PLANE:
    size = mj_model.geom_size[geom_id]
    plane_x = 2.0 * size[0] if size[0] > 0 else 20.0
    plane_y = 2.0 * size[1] if size[1] > 0 else 20.0
    mesh = trimesh.creation.box((plane_x, plane_y, 0.001))
  else:
    mesh = _create_shape_mesh(geom_type, mj_model.geom_size[geom_id])

  _apply_flat_color(mesh, _resolve_flat_rgba(mj_model, geom_id))
  return mesh


# OpenGL cube-map per-face selectors, indexed by face order
# (right, left, up, down, front, back). For a point/direction p on face fi:
#   sc = sc_sign * p[sc_axis];  tc = tc_sign * p[tc_axis];  ma = |p[major]|
#   s = (sc/ma + 1)/2;  t = (tc/ma + 1)/2   (t = 0 is the top row of the face)
# This is the convention MuJoCo's renderer uses for a geom cube texture; the box
# and sphere both route through it so they stay consistent.
_CUBE_FACE_ST: tuple[tuple[int, int, int, int, int], ...] = (
  (2, -1, 1, -1, 0),  # +X right:  sc=-z, tc=-y
  (2, 1, 1, -1, 0),  # -X left:   sc=+z, tc=-y
  (0, 1, 2, 1, 1),  # +Y up:     sc=+x, tc=+z
  (0, 1, 2, -1, 1),  # -Y down:   sc=+x, tc=-z
  (0, 1, 1, -1, 2),  # +Z front:  sc=+x, tc=-y
  (0, -1, 1, -1, 2),  # -Z back:   sc=-x, tc=-y
)


def _cube_face_st(
  positions: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Return per-point (s, t) in [0,1] for points lying on the given cube faces."""
  table = np.array(_CUBE_FACE_ST)
  sc_axis, sc_sign, tc_axis, tc_sign, major = (table[faces, k] for k in range(5))
  idx = np.arange(len(faces))
  sc = sc_sign * positions[idx, sc_axis]
  tc = tc_sign * positions[idx, tc_axis]
  ma = np.maximum(np.abs(positions[idx, major]), 1e-12)
  return (sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5


def _gl_cube_face(directions: np.ndarray) -> np.ndarray:
  """Return the cube face index (right, left, up, down, front, back) per direction."""
  x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
  ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
  x_major = (ax >= ay) & (ax >= az)
  y_major = (ay > ax) & (ay >= az)
  return np.where(
    x_major,
    np.where(x >= 0, 0, 1),
    np.where(y_major, np.where(y >= 0, 2, 3), np.where(z >= 0, 4, 5)),
  ).astype(np.int64)


def _extract_cubemap_atlas(mj_model: mujoco.MjModel, texid: int) -> Image.Image | None:
  """Build a vertical-strip PIL atlas from a cube map texture.

  Returns an image of size (w, 6*w) with face i pasted at rows
  [i*w, (i+1)*w) in PIL top-down coordinates, or None if the texture isn't a
  supported cube map.
  """
  if not _is_cubemap_texture(mj_model, texid):
    return None
  w = int(mj_model.tex_width[texid])
  h = int(mj_model.tex_height[texid])
  nc = int(mj_model.tex_nchannel[texid])
  adr = int(mj_model.tex_adr[texid])
  data = mj_model.tex_data[adr : adr + w * h * nc].reshape(6, w, w, nc)
  mode = {1: "L", 3: "RGB", 4: "RGBA"}[nc]

  atlas = Image.new(mode, (w, 6 * w))
  for i in range(6):
    # Face rows are stored top-down (PNG order); face i goes to atlas rows
    # [i*w, (i+1)*w). The box UVs sample this with the same (s, t) the sphere
    # sampler uses, so both share the OpenGL cube-map orientation.
    arr = data[i].astype(np.uint8)
    if nc == 1:
      arr = arr.reshape(w, w)
    atlas.paste(Image.fromarray(arr, mode=mode), (0, i * w))
  return atlas


def _create_cubemap_box_mesh(
  mj_model: mujoco.MjModel, geom_id: int
) -> trimesh.Trimesh | None:
  """Build a 6-quad textured box mesh from the geom's cube map material.

  Returns None when the geom has no material, no cube map texture, or
  the texture has an unsupported format. Each cube face gets its own
  slice of a vertical-strip atlas so the per-face images render with
  correct orientation.
  """
  matid = int(mj_model.geom_matid[geom_id])
  if matid < 0 or matid >= mj_model.nmat:
    return None
  texid = _get_texture_id(mj_model, matid)
  if texid < 0:
    return None
  atlas = _extract_cubemap_atlas(mj_model, texid)
  if atlas is None:
    return None

  size = mj_model.geom_size[geom_id]
  combos = ((-1, -1), (1, -1), (1, 1), (-1, 1))

  verts = np.zeros((24, 3), dtype=np.float64)
  uvs = np.zeros((24, 2), dtype=np.float64)
  faces = np.zeros((12, 3), dtype=np.int64)
  for fi, (_sc_axis, _sc_sign, _tc_axis, _tc_sign, major) in enumerate(_CUBE_FACE_ST):
    n = _CUBEMAP_AXES[fi]
    free = [k for k in range(3) if k != major]
    base = fi * 4
    corners = np.zeros((4, 3))
    for ci, (a, b) in enumerate(combos):
      corners[ci, major] = n[major] * size[major]
      corners[ci, free[0]] = a * size[free[0]]
      corners[ci, free[1]] = b * size[free[1]]
    verts[base : base + 4] = corners

    # Atlas UVs from the shared GL (s, t): face fi spans atlas rows
    # [fi*w, (fi+1)*w), so v = 1 - (fi + t) / 6 (glTF v-up over the strip).
    s, t = _cube_face_st(corners, np.full(4, fi))
    uvs[base : base + 4, 0] = s
    uvs[base : base + 4, 1] = 1.0 - (fi + t) / 6.0

    # Wind triangles so the face normal points outward.
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    if np.dot(normal, n) >= 0:
      faces[fi * 2 + 0] = (base + 0, base + 1, base + 2)
      faces[fi * 2 + 1] = (base + 0, base + 2, base + 3)
    else:
      faces[fi * 2 + 0] = (base + 0, base + 2, base + 1)
      faces[fi * 2 + 1] = (base + 0, base + 3, base + 2)

  mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
  rgba = mj_model.mat_rgba[matid]
  material = trimesh.visual.material.PBRMaterial(
    baseColorFactor=rgba,
    baseColorTexture=atlas,
    metallicFactor=0.0,
    roughnessFactor=1.0,
  )
  mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
  return mesh


def _cubemap_sample_colors(
  mj_model: mujoco.MjModel, texid: int, directions: np.ndarray
) -> np.ndarray | None:
  """Sample a cube map along directions, returning uint8 RGBA.

  Implements the OpenGL cube-map lookup MuJoCo uses for a geom (regular)
  cube texture: texcoords are the geom-local direction (x, y, z), faces
  are stored right, left, up, down, front, back, and the per-face (s, t)
  follow the GL spec. Face rows are stored top-down (PNG order), so t=0
  reads the top of the face image.
  """
  if not _is_cubemap_texture(mj_model, texid):
    return None
  w = int(mj_model.tex_width[texid])
  nc = int(mj_model.tex_nchannel[texid])
  adr = int(mj_model.tex_adr[texid])
  data = mj_model.tex_data[adr : adr + 6 * w * w * nc].reshape(6, w, w, nc)

  face = _gl_cube_face(directions)
  s, t = _cube_face_st(directions, face)
  col = np.clip((s * w).astype(int), 0, w - 1)
  row = np.clip((t * w).astype(int), 0, w - 1)
  px = data[face, row, col]

  out = np.full((len(directions), 4), 255, dtype=np.uint8)
  if nc == 1:
    out[:, :3] = px[:, :1]
  else:
    out[:, :nc] = px[:, :nc]
  return out


def _cubemap_to_equirect(
  mj_model: mujoco.MjModel,
  texid: int,
  scale: np.ndarray,
  width: int = 512,
  height: int = 256,
) -> Image.Image | None:
  """Bake a cube map into an equirectangular (lat-long) RGBA image.

  Row 0 is the north pole (+Z); columns sweep longitude 0..2*pi with 0 along
  +X. Sampling reuses _cubemap_sample_colors so the face layout matches the box
  renderer. The per-axis ``scale`` matches the geom's size: MuJoCo samples a
  geom cube texture by the geom-local position, so an ellipsoid's faces land at
  different angles than a sphere's.
  """
  lon = (np.arange(width) + 0.5) / width * (2.0 * np.pi)
  lat = (np.arange(height) + 0.5) / height * np.pi
  lon_g, lat_g = np.meshgrid(lon, lat)
  sin_lat = np.sin(lat_g)
  directions = np.stack(
    [sin_lat * np.cos(lon_g), sin_lat * np.sin(lon_g), np.cos(lat_g)], axis=-1
  ).reshape(-1, 3)
  colors = _cubemap_sample_colors(mj_model, texid, directions * scale)
  if colors is None:
    return None
  return Image.fromarray(colors.reshape(height, width, 4), "RGBA")


def _create_textured_sphere_mesh(
  mj_model: mujoco.MjModel, geom_id: int
) -> trimesh.Trimesh | None:
  """Build a UV-textured sphere or ellipsoid mesh from a cube map.

  Returns None when the geom has no material or no cube map texture. The
  cube map is baked into an equirectangular image and applied to a
  lat-long sphere with a duplicated seam column, so the texture stays
  crisp regardless of mesh density.
  """
  matid = int(mj_model.geom_matid[geom_id])
  if matid < 0 or matid >= mj_model.nmat:
    return None
  texid = _get_texture_id(mj_model, matid)
  if texid < 0 or not _is_cubemap_texture(mj_model, texid):
    return None
  size = mj_model.geom_size[geom_id]
  # Per-axis scale: sphere is uniform (radius), ellipsoid uses its three radii.
  is_ellipsoid = int(mj_model.geom_type[geom_id]) == mjtGeom.mjGEOM_ELLIPSOID
  scale = np.asarray(size, dtype=np.float64) if is_ellipsoid else np.full(3, size[0])
  image = _cubemap_to_equirect(mj_model, texid, scale)
  if image is None:
    return None

  n_lat, n_lon = 24, 48
  lat = np.linspace(0.0, np.pi, n_lat + 1)
  lon = np.linspace(0.0, 2.0 * np.pi, n_lon + 1)
  lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
  sin_lat = np.sin(lat_g)
  verts = np.stack(
    [sin_lat * np.cos(lon_g), sin_lat * np.sin(lon_g), np.cos(lat_g)], axis=-1
  ).reshape(-1, 3)
  # north pole (lat 0) -> top row of the image -> v=1 in glTF's v-up frame.
  uv = np.stack([lon_g / (2.0 * np.pi), 1.0 - lat_g / np.pi], axis=-1).reshape(-1, 2)

  ncols = n_lon + 1
  i, j = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
  a = (i * ncols + j).ravel()
  b, c, d = a + 1, a + ncols + 1, a + ncols
  # Wind triangles so their normals point outward (radius increases away
  # from the sphere center), otherwise the sphere renders inside-out.
  faces = np.concatenate(
    [np.stack([a, c, b], axis=1), np.stack([a, d, c], axis=1)], axis=0
  )

  mesh = trimesh.Trimesh(vertices=verts * scale, faces=faces, process=False)
  material = trimesh.visual.material.PBRMaterial(
    baseColorTexture=image,
    metallicFactor=0.0,
    roughnessFactor=1.0,
  )
  mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
  return mesh


def _create_textured_plane_mesh(
  mj_model: mujoco.MjModel, geom_id: int
) -> trimesh.Trimesh | None:
  """Build a textured quad for a plane geom with a 2D material texture.

  Returns None when the plane has no material, no 2D texture, or the
  texture is a cube map / unsupported format. UVs follow MuJoCo's
  texrepeat/texuniform semantics so the image tiles across the plane.
  The quad is double-sided so the plane stays visible from below.
  """
  matid = int(mj_model.geom_matid[geom_id])
  if matid < 0 or matid >= mj_model.nmat:
    return None
  texid = _get_texture_id(mj_model, matid)
  if texid < 0 or _is_cubemap_texture(mj_model, texid):
    return None
  image = _extract_texture_image(mj_model, texid)
  if image is None:
    return None

  size = mj_model.geom_size[geom_id]
  half_x = float(size[0]) if size[0] > 0 else 10.0
  half_y = float(size[1]) if size[1] > 0 else 10.0

  # texrepeat counts repetitions over the whole plane when texuniform is
  # false, or repetitions per unit length when true.
  rep = np.asarray(mj_model.mat_texrepeat[matid], dtype=np.float64)
  if mj_model.mat_texuniform[matid]:
    rep = rep * np.array([2.0 * half_x, 2.0 * half_y])

  corners = np.array(
    [
      [-half_x, -half_y, 0.0],
      [half_x, -half_y, 0.0],
      [half_x, half_y, 0.0],
      [-half_x, half_y, 0.0],
    ]
  )
  corner_uvs = np.array([[0.0, 0.0], [rep[0], 0.0], [rep[0], rep[1]], [0.0, rep[1]]])
  # Duplicate the corners so the reversed faces get an outward -Z normal.
  verts = np.vstack([corners, corners])
  uv = np.vstack([corner_uvs, corner_uvs])
  faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], dtype=np.int64)

  mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
  rgba = mj_model.mat_rgba[matid]
  geom_rgba = mj_model.geom_rgba[geom_id]
  # Blend when the material or geom is translucent, or the texture has alpha.
  # Mirror the mesh texture path, which checks both rgba sources.
  use_blending = rgba[-1] < 0.99 or geom_rgba[-1] < 0.99 or _has_alpha(image)
  material = trimesh.visual.material.PBRMaterial(
    baseColorFactor=rgba,
    baseColorTexture=image,
    metallicFactor=0.0,
    roughnessFactor=1.0,
    alphaMode="BLEND" if use_blending else "OPAQUE",
  )
  mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
  return mesh


# Primitive geom types that take a cube map; planes take 2D textures. Used by
# both _textured_primitive_mesh and get_geom_texture_id so they stay in sync.
_CUBEMAP_PRIMITIVE_TYPES = (
  mjtGeom.mjGEOM_BOX,
  mjtGeom.mjGEOM_SPHERE,
  mjtGeom.mjGEOM_ELLIPSOID,
)


def _textured_primitive_mesh(
  mj_model: mujoco.MjModel, geom_id: int
) -> trimesh.Trimesh | None:
  """Build a textured mesh for a primitive geom, or None if not textured.

  Box/sphere/ellipsoid use the geom's cube map; planes use a 2D texture.
  """
  geom_type = int(mj_model.geom_type[geom_id])
  if geom_type == mjtGeom.mjGEOM_BOX:
    return _create_cubemap_box_mesh(mj_model, geom_id)
  if geom_type in (mjtGeom.mjGEOM_SPHERE, mjtGeom.mjGEOM_ELLIPSOID):
    return _create_textured_sphere_mesh(mj_model, geom_id)
  if geom_type == mjtGeom.mjGEOM_PLANE:
    return _create_textured_plane_mesh(mj_model, geom_id)
  return None


def _create_heightfield_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Create a heightfield mesh, using the material texture when available."""
  hfield_id = mj_model.geom_dataid[geom_id]
  nrow = mj_model.hfield_nrow[hfield_id]
  ncol = mj_model.hfield_ncol[hfield_id]
  sx, sy, sz, _base = mj_model.hfield_size[hfield_id]

  offset = 0
  for k in range(hfield_id):
    offset += mj_model.hfield_nrow[k] * mj_model.hfield_ncol[k]
  hfield = mj_model.hfield_data[offset : offset + nrow * ncol].reshape(nrow, ncol)

  x_arr = np.linspace(-sx, sx, ncol)
  y_arr = np.linspace(-sy, sy, nrow)
  xx, yy = np.meshgrid(x_arr, y_arr)
  zz = hfield * sz

  vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

  ri, ci = np.mgrid[: nrow - 1, : ncol - 1]
  i0 = (ri * ncol + ci).ravel()
  faces = np.column_stack(
    [i0, i0 + 1, i0 + ncol + 1, i0, i0 + ncol + 1, i0 + ncol]
  ).reshape(-1, 3)

  # Try to use the material texture for coloring.
  matid = mj_model.geom_matid[geom_id]
  tex_image = None
  if matid >= 0:
    texid = _get_texture_id(mj_model, matid)
    if texid >= 0:
      tex_image = _extract_texture_image(mj_model, texid)

  if tex_image is not None:
    # UV-map the texture onto the heightfield grid.
    u = np.linspace(0, 1, ncol)
    v = np.linspace(0, 1, nrow)
    uu, vv = np.meshgrid(u, v)
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = mj_model.mat_rgba[matid]
    material = trimesh.visual.material.PBRMaterial(
      baseColorFactor=rgba,
      baseColorTexture=tex_image,
      metallicFactor=0.0,
      roughnessFactor=1.0,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh

  # Fallback: color by height using the geom/material flat color.
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  rgba = _resolve_flat_rgba(mj_model, geom_id)
  vertex_colors = np.tile(rgba, (vertices.shape[0], 1))
  mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
  return mesh


# ------------------------------------------------------------------
# Texture / visual utilities
# ------------------------------------------------------------------


def get_geom_texture_id(mj_model: mujoco.MjModel, geom_idx: int) -> int:
  """Return the texture ID the trimesh conversion will use, or -1.

  Returns -1 when the geom has no texture in the trimesh conversion
  path. Mesh geoms require UVs; heightfields use their material texture
  directly.
  """
  matid = mj_model.geom_matid[geom_idx]
  if matid < 0 or matid >= mj_model.nmat:
    return -1

  texid = _get_texture_id(mj_model, matid)
  if texid < 0:
    return -1

  geom_type = mj_model.geom_type[geom_idx]
  if geom_type == mjtGeom.mjGEOM_HFIELD:
    return texid

  # Box/sphere/ellipsoid take a cube map (see _textured_primitive_mesh).
  if geom_type in _CUBEMAP_PRIMITIVE_TYPES:
    return texid if _is_cubemap_texture(mj_model, texid) else -1

  # Planes take a 2D (non-cube-map) texture we can actually extract.
  if geom_type == mjtGeom.mjGEOM_PLANE:
    if _is_cubemap_texture(mj_model, texid):
      return -1
    return texid if _is_2d_texture_supported(mj_model, texid) else -1

  if geom_type != mjtGeom.mjGEOM_MESH:
    return -1

  mesh_id = mj_model.geom_dataid[geom_idx]
  if mj_model.mesh_texcoordnum[mesh_id] <= 0:
    return -1

  return texid


def group_geoms_by_visual_compat(
  mj_model: mujoco.MjModel, geom_ids: list[int]
) -> list[list[int]]:
  """Partition geom IDs into groups that can be safely merged.

  Geoms sharing the same texture ID are grouped together. Untextured
  opaque geoms form one group; untextured translucent geoms are split by
  color so each can be rendered with its own alpha-blended material.
  """
  groups: dict[object, list[int]] = {}
  for gid in geom_ids:
    tex_id = get_geom_texture_id(mj_model, gid)
    if tex_id >= 0:
      key: object = ("tex", tex_id)
    else:
      rgba = _resolve_flat_rgba(mj_model, gid)
      key = ("opaque",) if rgba[3] >= 255 else ("alpha", tuple(int(c) for c in rgba))
    groups.setdefault(key, []).append(gid)
  return list(groups.values())


# ------------------------------------------------------------------
# Merge / transform utilities
# ------------------------------------------------------------------


def _can_merge_vertices(mesh: trimesh.Trimesh) -> bool:
  """Check whether merge_vertices is safe (won't destroy per-vertex colors).

  trimesh's merge_vertices merges vertices at the same position regardless
  of vertex color.  This is unsafe when co-located vertices carry different
  colors (e.g. cubemap-textured meshes where adjacent faces have different
  colors at shared edges).
  """
  if not isinstance(mesh.visual, trimesh.visual.ColorVisuals):
    return True
  vc = mesh.visual.vertex_colors
  if vc is None or len(set(map(tuple, vc))) <= 1:
    return True
  # Check whether co-located vertices carry conflicting colors.
  rounded = np.round(mesh.vertices, decimals=6)
  _, inverse = np.unique(rounded, axis=0, return_inverse=True)
  color_hash = (
    vc[:, 0].astype(np.uint64) << 24
    | vc[:, 1].astype(np.uint64) << 16
    | vc[:, 2].astype(np.uint64) << 8
    | vc[:, 3].astype(np.uint64)
  )
  pairs = np.column_stack([inverse, color_hash])
  return len(np.unique(pairs, axis=0)) == len(np.unique(inverse))


def _merge_meshes(
  meshes: list[trimesh.Trimesh],
  positions: list[np.ndarray],
  quats: list[np.ndarray],
) -> trimesh.Trimesh:
  """Transform and merge meshes given positions and wxyz quaternions."""
  for mesh, pos, quat in zip(meshes, positions, quats, strict=True):
    transform = np.eye(4)
    transform[:3, :3] = vtf.SO3(quat).as_matrix()
    transform[:3, 3] = pos
    mesh.apply_transform(transform)

  result = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
  if _can_merge_vertices(result):
    result.merge_vertices()
  return result


def _apply_translucent_blend(mesh: trimesh.Trimesh) -> None:
  """Give a uniform-colored translucent mesh an alpha-blended material.

  trimesh exports per-vertex alpha as an opaque material, so a flat
  translucent color (e.g. goal nets at rgba alpha 0.3) renders solid.
  When every vertex shares one color with alpha < 1, swap to a PBR
  material with alphaMode=BLEND so the viewer renders it see-through.

  This only handles a single translucent color because trimesh can't carry
  per-vertex alpha plus a BLEND material. That is sufficient because
  group_geoms_by_visual_compat splits untextured translucent geoms by exact
  rgba, so every translucent merge reaching here is single-color
  (test_translucent_geoms_split_by_color guards that invariant).
  """
  vis = mesh.visual
  if not isinstance(vis, trimesh.visual.ColorVisuals):
    return
  vc = vis.vertex_colors
  if vc is None or len(vc) == 0:
    return
  unique = np.unique(vc, axis=0)
  if len(unique) != 1 or unique[0, 3] >= 255:
    return
  mesh.visual = trimesh.visual.TextureVisuals(
    uv=np.zeros((len(mesh.vertices), 2)),
    material=trimesh.visual.material.PBRMaterial(
      baseColorFactor=(unique[0] / 255.0).tolist(),
      metallicFactor=0.0,
      roughnessFactor=1.0,
      alphaMode="BLEND",
      doubleSided=True,
    ),
  )


def merge_geoms(mj_model: mujoco.MjModel, geom_ids: list[int]) -> trimesh.Trimesh:
  """Merge multiple geoms into a single trimesh in local body space."""
  meshes = []
  for geom_id in geom_ids:
    if mj_model.geom_type[geom_id] == mjtGeom.mjGEOM_MESH:
      meshes.append(mujoco_mesh_to_trimesh(mj_model, geom_id))
    else:
      meshes.append(create_primitive_mesh(mj_model, geom_id))

  result = _merge_meshes(
    meshes,
    [mj_model.geom_pos[gid] for gid in geom_ids],
    [mj_model.geom_quat[gid] for gid in geom_ids],
  )
  _apply_translucent_blend(result)
  return result


def _hull_trimesh_for_mesh_id(
  mj_model: mujoco.MjModel, mesh_id: int
) -> trimesh.Trimesh | None:
  """Return convex-hull polygon faces for a mesh asset as a trimesh.

  MuJoCo stores mesh hulls as polygon loops. This helper triangulates each
  polygon fan-style in the asset's local vertex index space.
  """
  if mj_model.nmeshpoly == 0 or int(mj_model.mesh_polynum[mesh_id]) == 0:
    return None

  vert_start = int(mj_model.mesh_vertadr[mesh_id])
  vert_count = int(mj_model.mesh_vertnum[mesh_id])
  vertices = mj_model.mesh_vert[vert_start : vert_start + vert_count].copy()

  poly_start = int(mj_model.mesh_polyadr[mesh_id])
  poly_count = int(mj_model.mesh_polynum[mesh_id])
  tri_faces: list[list[int]] = []

  for poly_id in range(poly_start, poly_start + poly_count):
    vert_adr = int(mj_model.mesh_polyvertadr[poly_id])
    vert_num = int(mj_model.mesh_polyvertnum[poly_id])
    poly_vertices = mj_model.mesh_polyvert[vert_adr : vert_adr + vert_num]
    for i in range(1, vert_num - 1):
      tri_faces.append(
        [int(poly_vertices[0]), int(poly_vertices[i]), int(poly_vertices[i + 1])]
      )

  if not tri_faces:
    return None

  return trimesh.Trimesh(
    vertices=vertices,
    faces=np.array(tri_faces, dtype=np.int32),
    process=False,
  )


def merge_geoms_hull(
  mj_model: mujoco.MjModel, geom_ids: list[int]
) -> trimesh.Trimesh | None:
  """Merge mesh-geom convex hulls into one trimesh in local body space."""
  meshes: list[trimesh.Trimesh] = []
  positions: list[np.ndarray] = []
  quats: list[np.ndarray] = []

  for geom_id in geom_ids:
    if int(mj_model.geom_type[geom_id]) != int(mjtGeom.mjGEOM_MESH):
      continue

    mesh_id = int(mj_model.geom_dataid[geom_id])
    if mesh_id < 0:
      continue

    hull = _hull_trimesh_for_mesh_id(mj_model, mesh_id)
    if hull is None:
      continue

    meshes.append(hull)
    positions.append(mj_model.geom_pos[geom_id])
    quats.append(mj_model.geom_quat[geom_id])

  if not meshes:
    return None

  return _merge_meshes(meshes, positions, quats)


def rotation_matrix_from_vectors(
  from_vec: np.ndarray, to_vec: np.ndarray
) -> np.ndarray:
  """3x3 rotation matrix that rotates from_vec to to_vec (Rodrigues)."""
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.eye(3)
  if np.allclose(from_vec, -to_vec):
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

  v = np.cross(from_vec, to_vec)
  s = np.linalg.norm(v)
  c = np.dot(from_vec, to_vec)
  vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def is_fixed_body(mj_model: mujoco.MjModel, body_id: int) -> bool:
  """Check if a body is fixed (welded to world, not attached to mocap)."""
  is_weld = mj_model.body_weldid[body_id] == 0
  root_id = mj_model.body_rootid[body_id]
  return bool(is_weld and mj_model.body_mocapid[root_id] < 0)


def get_body_name(mj_model: mujoco.MjModel, body_id: int) -> str:
  """Body name with fallback to ``body_{id}``."""
  name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
  return name if name else f"body_{body_id}"


def create_site_mesh(mj_model: mujoco.MjModel, site_id: int) -> trimesh.Trimesh:
  """Create a mesh for a single site."""
  rgba = mj_model.site_rgba[site_id].copy()
  if np.all(rgba == 0):
    rgba = np.array([0.5, 0.5, 0.5, 1.0])
  rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

  mesh = _create_shape_mesh(mj_model.site_type[site_id], mj_model.site_size[site_id])
  _apply_flat_color(mesh, rgba_uint8)
  return mesh


def merge_sites(mj_model: mujoco.MjModel, site_ids: list[int]) -> trimesh.Trimesh:
  """Merge multiple sites into a single trimesh in local body space."""
  return _merge_meshes(
    [create_site_mesh(mj_model, sid) for sid in site_ids],
    [mj_model.site_pos[sid] for sid in site_ids],
    [mj_model.site_quat[sid] for sid in site_ids],
  )


def get_site_name(mj_model: mujoco.MjModel, site_id: int) -> str:
  """Site name with fallback to ``site_{id}``."""
  name = mj_id2name(mj_model, mjtObj.mjOBJ_SITE, site_id)
  return name if name else f"site_{site_id}"
