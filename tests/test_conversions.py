import mujoco
import numpy as np
import pytest
import trimesh
import trimesh.visual
import trimesh.visual.material
from mujoco import mjtGeom
from PIL import Image

from mjviser.conversions import (
  _create_heightfield_mesh,
  _create_shape_mesh,
  _cubemap_vertex_colors,
  _extract_mesh_data,
  _extract_texture_image,
  _get_texture_id,
  _has_alpha,
  _merge_meshes,
  _resolve_flat_rgba,
  create_primitive_mesh,
  create_site_mesh,
  get_body_name,
  get_geom_texture_id,
  group_geoms_by_visual_compat,
  is_fixed_body,
  merge_geoms,
  merge_geoms_hull,
  merge_sites,
  mujoco_mesh_to_trimesh,
  rotation_matrix_from_vectors,
)
from mjviser.viewer import _format_speed

# Shape creation


@pytest.mark.parametrize(
  "geom_type",
  [
    mjtGeom.mjGEOM_SPHERE,
    mjtGeom.mjGEOM_BOX,
    mjtGeom.mjGEOM_CAPSULE,
    mjtGeom.mjGEOM_CYLINDER,
    mjtGeom.mjGEOM_ELLIPSOID,
  ],
)
def test_create_shape_mesh_types(geom_type):
  size = np.array([0.1, 0.2, 0.3])
  mesh = _create_shape_mesh(geom_type, size)
  assert isinstance(mesh, trimesh.Trimesh)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0


def test_create_shape_mesh_unsupported():
  with pytest.raises(ValueError, match="Unsupported shape type"):
    _create_shape_mesh(999, np.array([0.1, 0.1, 0.1]))


def test_create_primitive_mesh_plane(simple_model):
  mesh = create_primitive_mesh(simple_model, 0)
  assert isinstance(mesh, trimesh.Trimesh)
  z_extent = mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
  assert z_extent < 0.01


def test_create_primitive_mesh_hfield(hfield_model):
  mesh = _create_heightfield_mesh(hfield_model, 0)
  nrow, ncol = 10, 12
  assert len(mesh.vertices) == nrow * ncol
  expected_faces = 2 * (nrow - 1) * (ncol - 1)
  assert len(mesh.faces) == expected_faces


# Heightfield


def test_heightfield_faces_in_bounds(hfield_model):
  mesh = _create_heightfield_mesh(hfield_model, 0)
  assert mesh.faces.max() < len(mesh.vertices)
  assert mesh.faces.min() >= 0


# Color resolution


def test_resolve_flat_rgba_material_priority(simple_model):
  # Geom "sphere" (index 2) has material "red" (rgba 1,0,0,1).
  rgba = _resolve_flat_rgba(simple_model, 2)
  assert rgba[0] == 255
  assert rgba[1] == 0


def test_resolve_flat_rgba_no_material(simple_model):
  # Geom "box" (index 1) has no material, uses geom rgba (0,1,0,1).
  rgba = _resolve_flat_rgba(simple_model, 1)
  assert rgba[1] == 255
  assert rgba[0] == 0


# Cubemap


def test_cubemap_returns_none_no_texture(simple_model):
  result = _cubemap_vertex_colors(
    simple_model, -1, np.zeros((3, 3)), np.array([[0, 1, 2]])
  )
  assert result is None


def test_cubemap_vertex_colors_shape(cubemap_model):
  # Mesh geom 0 has a cube map texture and no UVs.
  verts, faces, uvs = _extract_mesh_data(cubemap_model, 0)
  assert uvs is None
  matid = cubemap_model.geom_matid[0]
  result = _cubemap_vertex_colors(cubemap_model, matid, verts, faces)
  assert result is not None
  new_verts, new_faces, colors = result
  assert len(new_verts) == 3 * len(new_faces)
  assert colors.shape == (len(new_verts), 4)


def test_cubemap_mesh_to_trimesh(cubemap_model):
  # Full pipeline: mesh geom with cube map goes through path 2.
  mesh = mujoco_mesh_to_trimesh(cubemap_model, 0)
  assert isinstance(mesh, trimesh.Trimesh)
  assert len(mesh.vertices) > 0


# 2D textures: normal maps and blending


def _texture_visual(mesh):
  assert isinstance(mesh.visual, trimesh.visual.TextureVisuals)
  return mesh.visual


def _pbr_material(mesh):
  mat = _texture_visual(mesh).material
  assert isinstance(mat, trimesh.visual.material.PBRMaterial)
  return mat


def test_textured_mesh_has_pbr_material(textured_model):
  assert _pbr_material(mujoco_mesh_to_trimesh(textured_model, 0)) is not None


def test_normalmap_applied_when_present(textured_model):
  # Geom 0 uses a material with a normal layer.
  assert _pbr_material(mujoco_mesh_to_trimesh(textured_model, 0)).normalTexture


def test_no_stray_normalmap_when_absent(textured_model):
  # Geom 1 has an albedo texture but no normal layer; the normalmap texid is
  # -1 and must not wrap around to another texture.
  assert _pbr_material(mujoco_mesh_to_trimesh(textured_model, 1)).normalTexture is None


def test_opaque_material_is_not_blended(textured_model):
  assert _pbr_material(mujoco_mesh_to_trimesh(textured_model, 1)).alphaMode == "OPAQUE"


def test_translucent_geom_enables_blending(textured_model):
  # Geom 2 has rgba alpha 0.5.
  assert _pbr_material(mujoco_mesh_to_trimesh(textured_model, 2)).alphaMode == "BLEND"


def test_has_alpha():
  assert not _has_alpha(Image.new("RGB", (4, 4), (255, 0, 0)))
  assert not _has_alpha(Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
  assert _has_alpha(Image.new("RGBA", (4, 4), (255, 0, 0, 128)))


def test_extract_texture_flip(textured_model):
  # Same texture extracted with and without the vertical flip differ.
  texid = _get_texture_id(textured_model, textured_model.geom_matid[0])
  flipped = np.asarray(_extract_texture_image(textured_model, texid, flip=True))
  unflipped = np.asarray(_extract_texture_image(textured_model, texid, flip=False))
  assert np.array_equal(unflipped, np.flipud(flipped))


# Cube map textures on box primitives


_CUBEMAP_BOX_XML = """
<mujoco>
  <asset>
    <texture name="cubetex" type="cube" builtin="flat" mark="cross"
             width="32" height="32" rgb1="0.8 0.2 0.2" markrgb="1 1 1"/>
    <texture name="tex2d" type="2d" builtin="checker"
             width="32" height="32" rgb1="0.8 0.2 0.2" rgb2="0.2 0.2 0.8"/>
    <material name="cubemat" texture="cubetex"/>
    <material name="flatmat" texture="tex2d"/>
  </asset>
  <worldbody>
    <geom name="cube" type="box" size="0.05 0.06 0.07" material="cubemat"/>
    <geom name="plain" type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
    <geom name="box2d" type="box" size="0.05 0.05 0.05" material="flatmat"/>
  </worldbody>
</mujoco>
"""

# MuJoCo cube face order with the outward axis each face sits on.
_CUBE_FACE_AXES = (
  (0, (1, 0, 0)),
  (1, (-1, 0, 0)),
  (2, (0, 1, 0)),
  (3, (0, -1, 0)),
  (4, (0, 0, 1)),
  (5, (0, 0, -1)),
)


def test_cubemap_box_mesh_structure():
  # A box with a cube map material becomes a textured 6-quad mesh whose
  # triangles all wind outward (closed, positive-volume box).
  model = mujoco.MjModel.from_xml_string(_CUBEMAP_BOX_XML)
  mesh = create_primitive_mesh(model, 0)
  material = _pbr_material(mesh)
  assert mesh.vertices.shape == (24, 3)
  assert mesh.faces.shape == (12, 3)
  assert _texture_visual(mesh).uv.shape == (24, 2)
  # 6 square faces stacked into a vertical strip: atlas is (w, 6*w).
  atlas = material.baseColorTexture
  assert atlas is not None and atlas.size == (32, 192)
  assert mesh.volume > 0
  assert mesh.is_winding_consistent


def test_cubemap_box_extents_match_geom_size():
  # The textured box spans the full geom size on every axis.
  model = mujoco.MjModel.from_xml_string(_CUBEMAP_BOX_XML)
  mesh = create_primitive_mesh(model, 0)
  lo, hi = mesh.bounds
  np.testing.assert_allclose(hi, [0.05, 0.06, 0.07])
  np.testing.assert_allclose(lo, [-0.05, -0.06, -0.07])


def test_cubemap_box_face_order_and_placement():
  # Paint each cube face a distinct color, then check each quad sits on the
  # correct outward axis and samples its own face color from the atlas.
  model = mujoco.MjModel.from_xml_string(_CUBEMAP_BOX_XML)
  colors = np.array(
    [
      [200, 0, 0],
      [0, 200, 0],
      [0, 0, 200],
      [200, 200, 0],
      [200, 0, 200],
      [0, 200, 200],
    ],
    dtype=np.uint8,
  )
  w = int(model.tex_width[0])
  adr = int(model.tex_adr[0])
  for i, color in enumerate(colors):
    start = adr + i * w * w * 3
    model.tex_data[start : start + w * w * 3] = np.tile(color, w * w)

  mesh = create_primitive_mesh(model, 0)
  atlas = np.asarray(_pbr_material(mesh).baseColorTexture)
  uv = _texture_visual(mesh).uv
  for fi, axis in _CUBE_FACE_AXES:
    quad = slice(fi * 4, fi * 4 + 4)
    centroid = mesh.vertices[quad].mean(axis=0)
    direction = centroid / np.linalg.norm(centroid)
    np.testing.assert_allclose(direction, axis, atol=1e-6)
    u, v = uv[quad].mean(axis=0)
    px = min(int(u * atlas.shape[1]), atlas.shape[1] - 1)
    py = min(int((1.0 - v) * atlas.shape[0]), atlas.shape[0] - 1)
    np.testing.assert_allclose(atlas[py, px], colors[fi], atol=2)


def test_box_without_cube_map_is_flat_colored():
  # Plain boxes and boxes with a 2D (non-cube) texture keep the flat fallback.
  model = mujoco.MjModel.from_xml_string(_CUBEMAP_BOX_XML)
  assert isinstance(create_primitive_mesh(model, 1).visual, trimesh.visual.ColorVisuals)
  assert isinstance(create_primitive_mesh(model, 2).visual, trimesh.visual.ColorVisuals)


def test_get_geom_texture_id_cube_map_box():
  # Boxes group by cube map texture; plain and 2D-textured boxes do not.
  model = mujoco.MjModel.from_xml_string(_CUBEMAP_BOX_XML)
  assert get_geom_texture_id(model, 0) == 0
  assert get_geom_texture_id(model, 1) == -1
  assert get_geom_texture_id(model, 2) == -1


# Mesh merging


def test_merge_meshes_single():
  mesh = trimesh.creation.box(extents=[1, 1, 1])
  result = _merge_meshes([mesh], [np.zeros(3)], [np.array([1, 0, 0, 0])])
  assert len(result.vertices) == len(mesh.vertices)


def test_merge_meshes_multiple():
  m1 = trimesh.creation.box(extents=[1, 1, 1])
  m2 = trimesh.creation.icosphere(radius=0.5)
  v1, f1 = len(m1.vertices), len(m1.faces)
  v2, f2 = len(m2.vertices), len(m2.faces)
  result = _merge_meshes(
    [m1, m2],
    [np.zeros(3), np.array([2, 0, 0])],
    [np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])],
  )
  assert len(result.vertices) == v1 + v2
  assert len(result.faces) == f1 + f2


def test_merge_geoms(simple_model):
  mesh = merge_geoms(simple_model, [1, 2])
  assert isinstance(mesh, trimesh.Trimesh)
  assert len(mesh.vertices) > 0


def test_group_geoms_by_visual_compat_splits_textured_hfield():
  xml = """
<mujoco>
  <asset>
    <hfield name="terrain" nrow="4" ncol="4" size="1 1 0.2 0.1"/>
    <texture name="tex" type="2d" builtin="checker"
             width="16" height="16" rgb1="0.1 0.8 0.2" rgb2="0.8 0.2 0.1"/>
    <material name="mat" texture="tex"/>
  </asset>
  <worldbody>
    <body name="terrain">
      <geom name="hf" type="hfield" hfield="terrain" material="mat"/>
      <geom name="wall" type="box" size="0.1 0.1 0.1" pos="1 0 0"
            rgba="0.2 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""
  model = mujoco.MjModel.from_xml_string(xml)
  groups = group_geoms_by_visual_compat(model, [0, 1])
  assert groups == [[0], [1]]


def test_merge_geoms_hull(cubemap_model):
  mesh = merge_geoms_hull(cubemap_model, [0])
  assert isinstance(mesh, trimesh.Trimesh)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0


def test_merge_sites(simple_model):
  mesh = merge_sites(simple_model, [0])
  assert isinstance(mesh, trimesh.Trimesh)
  assert len(mesh.vertices) > 0


def test_create_site_mesh_default_color(simple_model):
  mesh = create_site_mesh(simple_model, 0)
  assert isinstance(mesh, trimesh.Trimesh)


# Rotation


def test_rotation_matrix_identity():
  v = np.array([1.0, 0.0, 0.0])
  R = rotation_matrix_from_vectors(v, v)
  np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_rotation_matrix_opposite():
  v1 = np.array([1.0, 0.0, 0.0])
  v2 = np.array([-1.0, 0.0, 0.0])
  R = rotation_matrix_from_vectors(v1, v2)
  result = R @ v1
  np.testing.assert_allclose(result, v2, atol=1e-10)


def test_rotation_matrix_arbitrary():
  v1 = np.array([1.0, 0.0, 0.0])
  v2 = np.array([0.0, 1.0, 1.0])
  v2 = v2 / np.linalg.norm(v2)
  R = rotation_matrix_from_vectors(v1, v2)
  result = R @ v1
  np.testing.assert_allclose(result, v2, atol=1e-10)
  np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# Utilities


def test_is_fixed_body(simple_model):
  assert is_fixed_body(simple_model, 0)
  assert not is_fixed_body(simple_model, 1)


def test_get_body_name_named(simple_model):
  assert get_body_name(simple_model, 1) == "box_body"


def test_get_body_name_unnamed(simple_model):
  name = get_body_name(simple_model, 0)
  assert name == "world"


def test_format_speed():
  assert _format_speed(1.0) == "1x"
  assert _format_speed(0.5) == "1/2x"
  assert _format_speed(0.25) == "1/4x"
  assert _format_speed(0.125) == "1/8x"
  assert _format_speed(2.0) == "2x"
  assert _format_speed(4.0) == "4x"
  assert _format_speed(8.0) == "8x"
