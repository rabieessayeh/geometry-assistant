"""The geometry tools, checked against values computed by hand."""

import json
import math

import numpy as np
import pytest

from app import geometry
from app.geometry import Catalog, MeshError, for_llm, load_mesh, run_tool, to_stl
from tests.test_sample_models import HOLE_AREA

# ------------------------------------------------------------ model_info


def test_model_info_of_a_box(cat):
    out = run_tool(cat, "model_info", {"model": "box"})
    assert out["faces"] == 12
    assert out["vertices"] == 8
    assert out["bounding_box_mm"] == {"min": [0, 0, 0], "max": [10, 20, 30], "size": [10, 20, 30]}
    assert out["surface_area_mm2"] == pytest.approx(2 * (10 * 20 + 10 * 30 + 20 * 30))
    assert out["volume_mm3"] == pytest.approx(10 * 20 * 30)
    assert out["center_of_mass_mm"] == pytest.approx([5, 10, 15])
    assert out["watertight"] is True
    assert out["view_action"] == {
        "type": "bounding_box",
        "model": "box",
        "min": [0, 0, 0],
        "max": [10, 20, 30],
    }


def test_model_info_has_no_volume_for_an_open_mesh(cat):
    out = run_tool(cat, "model_info", {"model": "open_box"})
    assert out["watertight"] is False
    assert out["volume_mm3"] is None
    assert out["center_of_mass_mm"] is None
    assert "not watertight" in out["summary"]


# --------------------------------------------------- highlight_by_normal


def test_top_faces_of_a_box(cat, meshes):
    out = run_tool(cat, "highlight_by_normal", {"model": "box", "direction": "+z"})
    assert out["faces"] == 2
    assert out["area_mm2"] == pytest.approx(10 * 20)
    assert out["area_pct"] == pytest.approx(100 * 200 / 2200, abs=0.01)
    action = out["view_action"]
    assert action["type"] == "highlight_faces"
    assert np.allclose(meshes["box"].face_normals[action["faces"]], [0, 0, 1])


@pytest.mark.parametrize(("max_angle", "faces"), [(44, 0), (46, 2), (89, 2)])
def test_angle_tolerance(cat, max_angle, faces):
    """The slope of the wedge is at 45 degrees from -z."""
    args = {"model": "wedge", "direction": "-z", "max_angle_deg": max_angle}
    assert run_tool(cat, "highlight_by_normal", args)["faces"] == faces


# ----------------------------------------------------- overhang_analysis


def test_bridge_deck_needs_support(cat, meshes):
    out = run_tool(cat, "overhang_analysis", {"model": "bridge"})
    assert out["needs_support"] is True
    assert out["area_mm2"] == pytest.approx(30 * 20)  # underside of the deck between the pillars
    mesh = meshes["bridge"]
    faces = out["view_action"]["faces"]
    assert np.allclose(mesh.face_normals[faces], [0, 0, -1])
    assert np.allclose(mesh.triangles[faces][:, :, 2], 20)  # none of them is on the build plate


def test_faces_on_the_build_plate_need_no_support(cat):
    for model in ("box", "enclosure"):
        out = run_tool(cat, "overhang_analysis", {"model": model})
        assert out["needs_support"] is False
        assert out["view_action"]["faces"] == []


def test_printing_the_bridge_upside_down_needs_no_support(cat):
    out = run_tool(cat, "overhang_analysis", {"model": "bridge", "build_direction": "-z"})
    assert out["faces"] == 0


@pytest.mark.parametrize(("threshold", "area"), [(30, 5 * math.sqrt(200)), (45, 0), (60, 0)])
def test_overhang_threshold(cat, threshold, area):
    """The slope of the wedge is a 45 degree overhang: support is needed only below that."""
    out = run_tool(cat, "overhang_analysis", {"model": "wedge", "threshold_deg": threshold})
    assert out["area_mm2"] == pytest.approx(area, abs=1e-3)


# -------------------------------------------------------------- slice_at


def test_slice_of_the_enclosure(cat):
    out = run_tool(cat, "slice_at", {"model": "enclosure", "axis": "z", "value_mm": 15})
    assert out["contours"] == 2  # outer wall and cavity
    assert out["closed"] is True
    assert out["section_area_mm2"] == pytest.approx(80 * 50 - 76 * 46)
    assert out["perimeter_mm"] == pytest.approx(2 * (80 + 50) + 2 * (76 + 46))
    action = out["view_action"]
    assert action["type"] == "slice"
    for polyline in action["polylines"]:
        points = np.array(polyline)
        assert np.allclose(points[:, 2], 15)
        assert np.allclose(points[0], points[-1])


def test_slice_through_a_hole(cat):
    out = run_tool(cat, "slice_at", {"model": "l_bracket", "axis": "z", "value_mm": 2.5})
    assert out["contours"] == 2
    assert out["section_area_mm2"] == pytest.approx(60 * 40 - HOLE_AREA, abs=1e-3)


@pytest.mark.parametrize(
    ("axis", "value", "area"),
    [("x", 5, 20 * 30), ("y", 10, 10 * 30), ("z", 15, 10 * 20), ("z", 0, 10 * 20)],
)
def test_slices_of_a_box(cat, axis, value, area):
    out = run_tool(cat, "slice_at", {"model": "box", "axis": axis, "value_mm": value})
    assert out["contours"] == 1
    assert out["section_area_mm2"] == pytest.approx(area)


def test_slice_on_a_flat_face_returns_the_section_just_above(cat):
    """z = 20 mm is both the top of the pillars and the underside of the deck."""
    out = run_tool(cat, "slice_at", {"model": "bridge", "axis": "z", "value_mm": 20})
    assert out["section_area_mm2"] == pytest.approx(50 * 20)


def test_slice_outside_the_model(cat):
    out = run_tool(cat, "slice_at", {"model": "box", "axis": "z", "value_mm": 31})
    assert out == {"error": "The plane z = 31 mm misses the model (z = 0 to 30 mm)."}
    out = run_tool(cat, "slice_at", {"model": "box", "axis": "z", "value_mm": 30})
    assert "only touches the boundary" in out["error"]


def test_slice_of_an_open_mesh_has_no_area(cat):
    out = run_tool(cat, "slice_at", {"model": "open_box", "axis": "x", "value_mm": 5})
    assert out["closed"] is False
    assert out["section_area_mm2"] is None
    assert out["perimeter_mm"] == pytest.approx(20 + 30 + 30)


# ------------------------------------------- measure_distance, set_view


def test_measure_distance(cat):
    args = {"model": "box", "point_a": [0, 0, 0], "point_b": [3, 4, 12]}
    out = run_tool(cat, "measure_distance", args)
    assert out["distance_mm"] == pytest.approx(13)
    assert out["delta_mm"] == {"x": 3, "y": 4, "z": 12}
    assert out["view_action"] == {
        "type": "dimension",
        "model": "box",
        "point_a": [0, 0, 0],
        "point_b": [3, 4, 12],
        "distance_mm": 13,
    }


def test_set_view(cat):
    out = run_tool(cat, "set_view", {"view": "top"})
    assert out["view_action"] == {"type": "camera", "view": "top", "target": None}
    out = run_tool(cat, "set_view", {"view": "iso", "target": [1, 2, 3]})
    assert out["view_action"]["target"] == [1, 2, 3]


# ------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("model_info", {"model": "teapot"}, "Unknown model 'teapot'"),
        ("model_info", {}, "model: Field required"),
        ("model_info", {"model": "box", "units": "inch"}, "units: Extra inputs are not permitted"),
        ("highlight_by_normal", {"model": "box", "direction": "up"}, "direction: Input should be"),
        (
            "highlight_by_normal",
            {"model": "box", "direction": "+z", "max_angle_deg": 120},
            "max_angle_deg: Input should be less than or equal to 90",
        ),
        ("overhang_analysis", {"model": "box", "threshold_deg": 90}, "threshold_deg"),
        ("slice_at", {"model": "box", "axis": "w", "value_mm": 1}, "axis: Input should be"),
        ("slice_at", {"model": "box", "axis": "z", "value_mm": 1e9}, "value_mm"),
        ("slice_at", {"model": "box", "axis": "z", "value_mm": "high"}, "value_mm"),
        (
            "measure_distance",
            {"model": "box", "point_a": [0, 0], "point_b": [1, 1, 1]},
            "point_a: List should have at least 3 items",
        ),
        (
            "measure_distance",
            {"model": "box", "point_a": [0, 0, math.nan], "point_b": [1, 1, 1]},
            "point_a.2",
        ),
        ("set_view", {"view": "bottom"}, "view: Input should be"),
        ("run_python", {"code": "import os"}, "Unknown tool 'run_python'"),
    ],
)
def test_invalid_calls_return_an_error(cat, tool, args, message):
    out = run_tool(cat, tool, args)
    assert set(out) == {"error"}
    assert message in out["error"]


def test_unexpected_failures_are_contained(cat, monkeypatch):
    def boom(cat, args):
        raise RuntimeError("secret detail")

    monkeypatch.setitem(
        geometry.TOOLS, "set_view", geometry.Tool(boom, geometry.SetViewArgs, "broken")
    )
    assert run_tool(cat, "set_view", {"view": "top"}) == {
        "error": "Tool 'set_view' failed unexpectedly."
    }


# ------------------------------------------------- what the LLM receives


def test_llm_never_receives_face_indices_or_polylines(cat):
    calls = [
        ("highlight_by_normal", {"model": "l_bracket", "direction": "-x", "max_angle_deg": 90}),
        ("overhang_analysis", {"model": "l_bracket"}),
        ("slice_at", {"model": "l_bracket", "axis": "z", "value_mm": 2.5}),
    ]
    for tool, args in calls:
        result = run_tool(cat, tool, args)
        assert "view_action" in result
        compact = for_llm(result)
        assert "view_action" not in compact
        assert not any(isinstance(v, list) and len(v) > 3 for v in compact.values())
        assert len(json.dumps(compact)) < 500


def test_tool_schemas_use_the_portable_subset():
    assert [s["function"]["name"] for s in geometry.TOOL_SCHEMAS] == list(geometry.TOOLS)
    text = json.dumps(geometry.TOOL_SCHEMAS)
    for keyword in ("anyOf", "$ref", "minimum", "default", "title"):
        assert f'"{keyword}"' not in text
    slice_schema = geometry.TOOL_SCHEMAS[3]["function"]["parameters"]
    assert slice_schema["required"] == ["model", "axis", "value_mm"]
    assert slice_schema["properties"]["axis"]["enum"] == ["x", "y", "z"]
    view_schema = geometry.TOOL_SCHEMAS[5]["function"]["parameters"]
    assert view_schema["required"] == ["view"]
    assert view_schema["properties"]["target"]["items"] == {"type": "number"}


# ------------------------------------------------------ files and catalog


@pytest.mark.parametrize("file_type", ["stl", "obj"])
def test_load_mesh_round_trip(meshes, file_type):
    data = meshes["bridge"].export(file_type=file_type)
    data = data.encode() if isinstance(data, str) else data
    mesh = load_mesh(data, file_type, max_faces=1000)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(13000)


def test_to_stl_keeps_the_face_order(meshes):
    """The viewer relies on triangle i of the STL being face i of the mesh."""
    mesh = meshes["l_bracket"]
    reloaded = load_mesh(to_stl(mesh), "stl", max_faces=1000)
    assert np.allclose(reloaded.triangles, mesh.triangles, atol=1e-4)


@pytest.mark.parametrize(
    ("data", "file_type", "message"),
    [
        (b"", "stl", "no triangles"),
        (b"this is not a mesh", "stl", "no triangles"),
        (b"solid empty\nendsolid empty\n", "stl", "no triangles"),
        (b"\x00" * 80 + b"\xff\xff\xff\x7f" + b"\x00" * 100, "stl", "could not be read as STL"),
        (b"hello\n", "obj", "no triangles"),
        (b"v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3\n", "obj", "no surface"),
        (b"v 0 0 0\nv 1 0 0\nv 0 9e9 0\nf 1 2 3\n", "obj", "Coordinates must be finite"),
        (b"anything", "step", "Unsupported file type"),
    ],
)
def test_load_mesh_rejects_unusable_files(data, file_type, message):
    with pytest.raises(MeshError, match=message):
        load_mesh(data, file_type, max_faces=1000)


def test_load_mesh_enforces_the_face_limit(meshes):
    with pytest.raises(MeshError, match="290 faces; the limit is 100"):
        load_mesh(to_stl(meshes["l_bracket"]), "stl", max_faces=100)


def test_catalog_ids_are_unique_slugs(meshes):
    cat = Catalog()
    assert cat.add("My Part (v2)", meshes["box"]) == "my_part_v2"
    assert cat.add("my-part v2", meshes["box"]) == "my_part_v2_2"
    assert cat.add("bridge", meshes["box"], reserved=["bridge", "bridge_2"]) == "bridge_3"
    assert cat.add("***", meshes["box"]) == "model"


def test_catalog_from_folder(tmp_path, meshes):
    meshes["box"].export(tmp_path / "Box.stl")
    meshes["wedge"].export(tmp_path / "wedge.obj")
    (tmp_path / "notes.txt").write_text("ignored")
    cat = Catalog.from_folder(tmp_path)
    assert sorted(cat.models) == ["box", "wedge"]
    with pytest.raises(FileNotFoundError, match="make_sample_models"):
        Catalog.from_folder(tmp_path / "missing")


def test_merged_catalog(meshes):
    merged = Catalog({"a": meshes["box"]}).merged(Catalog({"b": meshes["wedge"]}))
    assert sorted(merged.models) == ["a", "b"]
