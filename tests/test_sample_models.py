"""The procedural sample parts: closed solids with the volumes computed by hand."""

import math

import pytest
import trimesh

from scripts import make_sample_models as samples

# Cross-section area of a hole: a regular polygon inscribed in the nominal circle.
HOLE_AREA = (
    0.5
    * samples.HOLE_SECTIONS
    * samples.HOLE_RADIUS**2
    * math.sin(2 * math.pi / samples.HOLE_SECTIONS)
)


@pytest.mark.parametrize("name", sorted(samples.SAMPLES))
def test_samples_are_closed_solids_resting_on_z0(name):
    mesh = samples.SAMPLES[name]()
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert mesh.bounds[0].tolist() == pytest.approx([0, 0, 0], abs=1e-9)


def test_l_bracket_volume():
    t = samples.BRACKET_THICKNESS
    solid = samples.BRACKET_WIDTH * t * (samples.BRACKET_LENGTH + samples.BRACKET_HEIGHT - t)
    assert solid == 19000
    assert samples.l_bracket().volume == pytest.approx(solid - 2 * HOLE_AREA * t, rel=1e-6)


def test_bridge_volume():
    assert samples.bridge().volume == pytest.approx(2 * 10 * 20 * 20 + 50 * 20 * 5)


def test_enclosure_volume():
    assert samples.enclosure().volume == pytest.approx(80 * 50 * 30 - 76 * 46 * 28)


def test_write_samples(tmp_path):
    paths = samples.write_samples(tmp_path)
    assert sorted(p.name for p in paths) == ["bridge.stl", "enclosure.stl", "l_bracket.stl"]
    reloaded = trimesh.load(paths[0], force="mesh")
    assert reloaded.is_watertight
