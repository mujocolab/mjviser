from pathlib import Path

import mujoco
import pytest
import viser

from mjviser import ViserMujocoScene

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

_SIMPLE_XML = """
<mujoco>
  <asset>
    <material name="red" rgba="1 0 0 1"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="box_body" pos="0 0 1">
      <joint type="free"/>
      <geom name="box" type="box" size="0.1 0.1 0.1" rgba="0 1 0 1"/>
      <geom name="sphere" type="sphere" size="0.05" pos="0.2 0 0"
            material="red"/>
      <geom name="capsule" type="capsule" size="0.03 0.1" pos="-0.2 0 0"/>
      <site name="tip" pos="0 0 0.1" size="0.02"/>
    </body>
  </worldbody>
</mujoco>
"""

_HFIELD_XML = """
<mujoco>
  <asset>
    <hfield name="terrain" nrow="10" ncol="12" size="2 2 0.5 0.1"/>
  </asset>
  <worldbody>
    <geom type="hfield" hfield="terrain"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def humanoid_model():
  return mujoco.MjModel.from_xml_path(str(_EXAMPLES / "humanoid.xml"))


@pytest.fixture
def humanoid_data(humanoid_model):
  data = mujoco.MjData(humanoid_model)
  mujoco.mj_forward(humanoid_model, data)
  return data


@pytest.fixture
def simple_model():
  return mujoco.MjModel.from_xml_string(_SIMPLE_XML)


@pytest.fixture
def simple_data(simple_model):
  data = mujoco.MjData(simple_model)
  mujoco.mj_forward(simple_model, data)
  return data


_CUBEMAP_XML = """
<mujoco>
  <asset>
    <texture name="cubetex" type="cube" builtin="flat" mark="cross"
             width="64" height="64" rgb1="0.8 0.2 0.2" markrgb="1 1 1"/>
    <material name="cubemat" texture="cubetex"/>
    <mesh name="box"
          vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                  -1 -1 1   1 -1 1   1 1 1   -1 1 1"
          face="0 3 2  0 2 1  4 5 6  4 6 7
                0 1 5  0 5 4  2 3 7  2 7 6
                0 4 7  0 7 3  1 2 6  1 6 5"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="box" material="cubemat"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def hfield_model():
  return mujoco.MjModel.from_xml_string(_HFIELD_XML)


@pytest.fixture
def cubemap_model():
  return mujoco.MjModel.from_xml_string(_CUBEMAP_XML)


def _stop_server(server: viser.ViserServer) -> None:
  try:
    server.stop()
  except RuntimeError:
    pass  # Event loop already closed (Python 3.10).


@pytest.fixture
def scene(simple_model):
  server = viser.ViserServer(port=0)
  s = ViserMujocoScene(server, simple_model, num_envs=1)
  yield s
  _stop_server(server)
