import numpy as np
import pytest
import trimesh
from mujoco import mjtGeom

from mjviser.conversions import (
  _create_heightfield_mesh,
  _create_shape_mesh,
  _cubemap_vertex_colors,
  _extract_mesh_data,
  _merge_meshes,
  _resolve_flat_rgba,
  create_primitive_mesh,
  create_site_mesh,
  get_body_name,
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
