"""Geometry tools exposed to the LLM.

Every tool is a plain, deterministic Python function working on triangle
meshes (trimesh + numpy). The LLM never computes geometry: it only chooses a
tool from the `TOOLS` whitelist and its arguments, which are validated against
a Pydantic model before anything runs.

A tool returns compact numbers for the LLM and a `view_action` for the 3D
viewer (faces to highlight, a contour to draw, a camera move...). The view
action can be large, so it goes to the client and never to the LLM.

Lengths are in millimetres and Z is up, as in 3D printing.
"""

from __future__ import annotations

import io
import logging
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from trimesh.intersections import mesh_plane

logger = logging.getLogger(__name__)

MESH_FILE_TYPES = ("stl", "obj")
MAX_COORD_MM = 100_000.0  # coordinates accepted in arguments and in uploaded meshes
VIEW_ACTION = "view_action"  # key of a tool result that is sent to the viewer only

DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}

Direction = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
Axis = Literal["x", "y", "z"]
View = Literal["top", "front", "side", "iso"]
Coordinate = Annotated[float, Field(ge=-MAX_COORD_MM, le=MAX_COORD_MM)]
Point = Annotated[list[Coordinate], Field(min_length=3, max_length=3)]


class ToolError(ValueError):
    """A tool call that cannot be executed; the message is safe to show the LLM."""


class MeshError(ValueError):
    """A file that cannot be used as a model; the message is safe to show the user."""


# -------------------------------------------------------------- catalog


def _slug(name: str) -> str:
    """Turn a file name into a short identifier the LLM can copy reliably."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "model"


@dataclass
class Catalog:
    """In-memory catalogue of named meshes (millimetres)."""

    models: dict[str, trimesh.Trimesh] = field(default_factory=dict)

    def add(self, name: str, mesh: trimesh.Trimesh, reserved: Iterable[str] = ()) -> str:
        """Register a mesh and return its id, made unique among the catalogue and `reserved`."""
        taken = set(self.models) | set(reserved)
        base = _slug(name)
        model_id, n = base, 2
        while model_id in taken:
            model_id, n = f"{base}_{n}", n + 1
        self.models[model_id] = mesh
        logger.info("Loaded model '%s' (%d faces)", model_id, len(mesh.faces))
        return model_id

    def get(self, name: str) -> trimesh.Trimesh:
        """Return a mesh by id, or raise a `ToolError` listing valid ids."""
        if name not in self.models:
            raise ToolError(f"Unknown model '{name}'. Available: {sorted(self.models)}")
        return self.models[name]

    def merged(self, other: Catalog) -> Catalog:
        """New catalogue holding the models of both (the meshes are shared, not copied)."""
        return Catalog({**self.models, **other.models})

    @classmethod
    def from_folder(cls, folder: str | Path, max_faces: int = 1_000_000) -> Catalog:
        """Load every .stl / .obj file of a folder as a model."""
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Model folder '{folder}' does not exist. Run scripts/make_sample_models.py "
                "or set SAMPLES_DIR."
            )
        cat = cls()
        for path in sorted(folder.iterdir()):
            file_type = path.suffix.lower().lstrip(".")
            if file_type in MESH_FILE_TYPES:
                cat.add(path.stem, load_mesh(path.read_bytes(), file_type, max_faces))
        if not cat.models:
            logger.warning("No model found in '%s'", folder)
        return cat


def load_mesh(data: bytes, file_type: str, max_faces: int) -> trimesh.Trimesh:
    """Parse and validate an STL or OBJ file.

    Args:
        data: Content of the file.
        file_type: `"stl"` or `"obj"`.
        max_faces: Largest number of triangles accepted.

    Raises:
        MeshError: If the file is not a usable triangle mesh.
    """
    if file_type not in MESH_FILE_TYPES:
        raise MeshError(f"Unsupported file type '{file_type}'. Use STL or OBJ.")
    options = {"skip_materials": True} if file_type == "obj" else {}
    try:
        mesh = trimesh.load(io.BytesIO(data), file_type=file_type, force="mesh", **options)
    except Exception as exc:
        # trimesh raises many exception types on malformed files.
        logger.info("Rejected %s file: %s", file_type, exc)
        raise MeshError(f"The file could not be read as {file_type.upper()}.") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise MeshError("The file contains no triangles.")
    if len(mesh.faces) > max_faces:
        raise MeshError(f"The mesh has {len(mesh.faces)} faces; the limit is {max_faces}.")
    if not np.isfinite(mesh.vertices).all() or np.abs(mesh.vertices).max() > MAX_COORD_MM:
        raise MeshError(f"Coordinates must be finite and within ±{MAX_COORD_MM:g} mm.")
    if mesh.area <= 0:
        raise MeshError("The mesh has no surface.")
    return mesh


def to_stl(mesh: trimesh.Trimesh) -> bytes:
    """Binary STL of a mesh; triangle i of the file is face i of the mesh."""
    return mesh.export(file_type="stl")


def describe_model(model_id: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Short description of a model: size and counts, cheap to compute."""
    lower, upper = mesh.bounds
    return {
        "id": model_id,
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "bounding_box_mm": {
            "min": _vec(lower),
            "max": _vec(upper),
            "size": _vec(upper - lower),
        },
        "watertight": bool(mesh.is_watertight),
    }


# -------------------------------------------------------------- helpers


def _num(value: float, digits: int = 3) -> float:
    """Round to a JSON-friendly float (and avoid `-0.0`)."""
    return round(float(value), digits) + 0.0


def _vec(values: Iterable[float], digits: int = 3) -> list[float]:
    """Round a vector."""
    return [_num(v, digits) for v in values]


def _fmt(values: Iterable[float], separator: str = " × ") -> str:
    """Compact text for a vector, e.g. `60 × 40 × 5`."""
    return separator.join(f"{v:g}" for v in values)


def _face_stats(mesh: trimesh.Trimesh, mask: np.ndarray) -> dict[str, Any]:
    """Count and area of a subset of faces."""
    area = float(mesh.area_faces[mask].sum())
    return {
        "faces": int(mask.sum()),
        "area_mm2": _num(area),
        "area_pct": _num(100 * area / mesh.area, 2),
    }


def _highlight(model: str, mask: np.ndarray, kind: str, label: str) -> dict[str, Any]:
    """View action colouring a subset of faces."""
    return {
        "type": "highlight_faces",
        "model": model,
        "kind": kind,
        "label": label,
        "faces": np.flatnonzero(mask).tolist(),
    }


def _chain(segments: np.ndarray) -> list[np.ndarray]:
    """Join oriented segments end to start into polylines.

    Args:
        segments: Array of shape (n, 2, 3); segment i goes from `[i, 0]` to `[i, 1]`.

    Returns:
        Polylines as (m, 3) arrays. A closed contour repeats its first point at the end.
    """
    keys = [tuple(p) for p in np.round(segments, 6).reshape(-1, 3).tolist()]
    starts, ends = keys[0::2], keys[1::2]
    by_start: dict[tuple[float, ...], list[int]] = {}
    for i, key in enumerate(starts):
        by_start.setdefault(key, []).append(i)

    used = np.zeros(len(segments), dtype=bool)
    polylines = []
    for first in range(len(segments)):
        if used[first]:
            continue
        points, current = [segments[first, 0]], first
        while True:
            used[current] = True
            points.append(segments[current, 1])
            following = [i for i in by_start.get(ends[current], []) if not used[i]]
            if not following:
                break
            current = following[0]
        polylines.append(np.array(points))
    return polylines


# ------------------------------------------------------------ arguments


class _Args(BaseModel):
    """Base class of tool arguments: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class _ModelArgs(_Args):
    """Arguments of a tool working on one model."""

    model: str = Field(min_length=1, max_length=64, description="Id of the model.")


class ModelInfoArgs(_ModelArgs):
    """Arguments of `model_info`."""


class HighlightByNormalArgs(_ModelArgs):
    """Arguments of `highlight_by_normal`."""

    direction: Direction = Field(description="Direction the faces look towards; +z is up.")
    max_angle_deg: float = Field(
        default=10.0,
        ge=0,
        le=90,
        description="Largest angle between the face normal and the direction (default 10).",
    )


class OverhangAnalysisArgs(_ModelArgs):
    """Arguments of `overhang_analysis`."""

    threshold_deg: float = Field(
        default=45.0,
        ge=0,
        le=89,
        description="Overhang angle from the vertical above which support is needed (default 45).",
    )
    build_direction: Direction = Field(
        default="+z", description="Direction in which layers are stacked (default +z)."
    )


class SliceAtArgs(_ModelArgs):
    """Arguments of `slice_at`."""

    axis: Axis = Field(description="Axis perpendicular to the cutting plane.")
    value_mm: Coordinate = Field(description="Position of the plane along the axis, in mm.")


class MeasureDistanceArgs(_ModelArgs):
    """Arguments of `measure_distance`."""

    point_a: Point = Field(description="First point [x, y, z] in mm.")
    point_b: Point = Field(description="Second point [x, y, z] in mm.")


class SetViewArgs(_Args):
    """Arguments of `set_view`."""

    view: View = Field(description="top: from +z; front: from -y; side: from +x; iso: 3/4 view.")
    target: Point | None = Field(
        default=None,
        description="Optional point [x, y, z] in mm to look at (default: the model centre).",
    )


# ---------------------------------------------------------------- tools


def model_info(cat: Catalog, args: ModelInfoArgs) -> dict[str, Any]:
    """Size, surface area, volume and centre of mass of a model."""
    mesh = cat.get(args.model)
    info = describe_model(args.model, mesh)
    box, watertight = info["bounding_box_mm"], info["watertight"]
    summary = (
        f"'{args.model}': {info['faces']} faces, {info['vertices']} vertices, "
        f"bounding box {_fmt(box['size'])} mm, surface area {mesh.area:.2f} mm²"
    )
    if watertight:
        volume = abs(float(mesh.volume))
        summary += f", volume {volume:.2f} mm³, watertight."
    else:
        # Volume and centre of mass are only defined for a closed surface.
        summary += ", not watertight (volume undefined)."
    return {
        "summary": summary,
        "model": args.model,
        "faces": info["faces"],
        "vertices": info["vertices"],
        "bounding_box_mm": box,
        "surface_area_mm2": _num(mesh.area),
        "watertight": watertight,
        "volume_mm3": _num(volume) if watertight else None,
        "center_of_mass_mm": _vec(mesh.center_mass) if watertight else None,
        VIEW_ACTION: {
            "type": "bounding_box",
            "model": args.model,
            "min": box["min"],
            "max": box["max"],
        },
    }


def highlight_by_normal(cat: Catalog, args: HighlightByNormalArgs) -> dict[str, Any]:
    """Faces whose normal is within `max_angle_deg` of a direction."""
    mesh = cat.get(args.model)
    cosines = mesh.face_normals @ np.array(DIRECTIONS[args.direction], dtype=float)
    mask = cosines >= math.cos(math.radians(args.max_angle_deg)) - 1e-9
    stats = _face_stats(mesh, mask)
    label = f"{args.direction} within {args.max_angle_deg:g}°"
    return {
        "summary": f"{stats['faces']} of {len(mesh.faces)} faces of '{args.model}' face {label}: "
        f"{stats['area_mm2']:g} mm², {stats['area_pct']:g}% of the surface.",
        **stats,
        VIEW_ACTION: _highlight(args.model, mask, "normal", label),
    }


def overhang_analysis(cat: Catalog, args: OverhangAnalysisArgs) -> dict[str, Any]:
    """Faces that need support material when printing along `build_direction`.

    A face needs support when it looks downwards and its overhang angle,
    measured from the vertical, exceeds the threshold (a horizontal ceiling is
    a 90 degree overhang). Faces lying on the build plate are excluded.
    """
    mesh = cat.get(args.model)
    up = np.array(DIRECTIONS[args.build_direction], dtype=float)
    # Overhang angle = 90° - angle(normal, down), so "overhang > threshold" reads:
    down_facing = -(mesh.face_normals @ up) > math.sin(math.radians(args.threshold_deg)) + 1e-9
    heights = mesh.vertices @ up
    tolerance = 1e-6 * max(1.0, float(mesh.extents.max()))
    on_plate = (heights[mesh.faces] <= heights.min() + tolerance).all(axis=1)
    mask = down_facing & ~on_plate
    stats = _face_stats(mesh, mask)
    label = f"overhang > {args.threshold_deg:g}°, build {args.build_direction}"
    return {
        "summary": f"{stats['faces']} of {len(mesh.faces)} faces of '{args.model}' need support "
        f"({label}): {stats['area_mm2']:g} mm², {stats['area_pct']:g}% of the surface.",
        "needs_support": stats["faces"] > 0,
        **stats,
        VIEW_ACTION: _highlight(args.model, mask, "overhang", label),
    }


def slice_at(cat: Catalog, args: SliceAtArgs) -> dict[str, Any]:
    """Cross-section of a model by a plane perpendicular to an axis.

    When the plane coincides with a flat face of the model, the section just
    above it (towards +axis) is returned.
    """
    mesh = cat.get(args.model)
    axis = "xyz".index(args.axis)
    lower, upper = (float(v) for v in mesh.bounds[:, axis])
    span = f"{args.axis} = {lower:g} to {upper:g} mm"
    if not lower <= args.value_mm <= upper:
        raise ToolError(f"The plane {args.axis} = {args.value_mm:g} mm misses the model ({span}).")

    normal = np.zeros(3)
    normal[axis] = 1.0
    segments, face_index = mesh_plane(
        mesh, plane_normal=normal, plane_origin=normal * args.value_mm, return_faces=True
    )
    if len(segments) == 0:
        raise ToolError(
            f"The plane {args.axis} = {args.value_mm:g} mm only touches the boundary of the "
            f"model; choose a value strictly inside {span}."
        )

    # Orient every segment so the material is on its left when seen from +axis.
    # Outer contours then run counter-clockwise and holes clockwise, and the
    # signed areas add up to the area of the section.
    along = np.cross(normal, mesh.face_normals[face_index])
    backwards = np.einsum("ij,ij->i", segments[:, 1] - segments[:, 0], along) < 0
    segments = np.where(backwards[:, None, None], segments[:, ::-1], segments)

    polylines = _chain(segments)
    closed = all(np.allclose(p[0], p[-1], atol=1e-5) for p in polylines)
    area = abs(0.5 * float(np.cross(segments[:, 0], segments[:, 1]).sum(axis=0) @ normal))
    perimeter = float(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum())
    points = segments.reshape(-1, 3)

    where = f"'{args.model}' at {args.axis} = {args.value_mm:g} mm"
    if closed:
        summary = (
            f"Section of {where}: {len(polylines)} closed contour(s), area {area:.2f} mm², "
            f"perimeter {perimeter:.2f} mm."
        )
    else:
        summary = f"Section of {where}: open contours (the mesh is not closed), area undefined."
    return {
        "summary": summary,
        "contours": len(polylines),
        "closed": closed,
        "section_area_mm2": _num(area) if closed else None,
        "perimeter_mm": _num(perimeter),
        "section_bounds_mm": {"min": _vec(points.min(axis=0)), "max": _vec(points.max(axis=0))},
        VIEW_ACTION: {
            "type": "slice",
            "model": args.model,
            "axis": args.axis,
            "value_mm": args.value_mm,
            "polylines": [np.round(p, 4).tolist() for p in polylines],
        },
    }


def measure_distance(cat: Catalog, args: MeasureDistanceArgs) -> dict[str, Any]:
    """Straight-line distance between two points, with its components along the axes."""
    cat.get(args.model)  # the measurement is attached to a model, which must exist
    a, b = np.array(args.point_a), np.array(args.point_b)
    delta = b - a
    distance = float(np.linalg.norm(delta))
    return {
        "summary": f"Distance between ({_fmt(_vec(a), ', ')}) and ({_fmt(_vec(b), ', ')}): "
        f"{distance:.3f} mm.",
        "distance_mm": _num(distance),
        "delta_mm": dict(zip("xyz", _vec(delta), strict=True)),
        VIEW_ACTION: {
            "type": "dimension",
            "model": args.model,
            "point_a": _vec(a, 4),
            "point_b": _vec(b, 4),
            "distance_mm": _num(distance),
        },
    }


def set_view(cat: Catalog, args: SetViewArgs) -> dict[str, Any]:
    """Move the camera to a standard view; computes nothing."""
    return {
        "summary": f"Camera moved to the {args.view} view.",
        "view": args.view,
        VIEW_ACTION: {"type": "camera", "view": args.view, "target": args.target},
    }


# ------------------------------------------------------------- registry


@dataclass(frozen=True)
class Tool:
    """A whitelisted tool: its function, argument model and LLM-facing description."""

    func: Callable[[Catalog, Any], dict[str, Any]]
    args_model: type[_Args]
    description: str


TOOLS: dict[str, Tool] = {
    "model_info": Tool(
        model_info,
        ModelInfoArgs,
        "Faces, vertices, bounding box, surface area, volume, watertightness and centre of "
        "mass of a model.",
    ),
    "highlight_by_normal": Tool(
        highlight_by_normal,
        HighlightByNormalArgs,
        "Highlight and measure the faces oriented towards a direction, e.g. +z for the top "
        "faces, -z for the bottom faces.",
    ),
    "overhang_analysis": Tool(
        overhang_analysis,
        OverhangAnalysisArgs,
        "3D printing: highlight the faces that need support material and report their area.",
    ),
    "slice_at": Tool(
        slice_at,
        SliceAtArgs,
        "Cut a model with a plane perpendicular to an axis: draws the contours and returns "
        "the area and perimeter of the cross-section.",
    ),
    "measure_distance": Tool(
        measure_distance,
        MeasureDistanceArgs,
        "Distance in mm between two 3D points, e.g. the points picked by the user.",
    ),
    "set_view": Tool(
        set_view,
        SetViewArgs,
        "Move the camera to a standard view: top, front, side or iso.",
    ),
}

# Keys of a JSON schema kept for the LLM. Providers differ in what they accept
# (titles, defaults, numeric bounds...), so only the common subset is sent;
# the bounds are still enforced by the Pydantic models.
_SCHEMA_KEYS = {"type", "description", "enum", "items"}


def _llm_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Reduce the JSON schema of one argument to the subset every provider understands."""
    if "anyOf" in prop:  # optional argument: describe the non-null variant
        variant = next(v for v in prop["anyOf"] if v.get("type") != "null")
        prop = {**variant, "description": prop.get("description", "")}
    out = {k: v for k, v in prop.items() if k in _SCHEMA_KEYS}
    if "items" in out:
        out["items"] = _llm_property(out["items"])
    return out


def _llm_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema of a tool's arguments, as sent to the LLM."""
    schema = model.model_json_schema()
    properties = {name: _llm_property(prop) for name, prop in schema.get("properties", {}).items()}
    out: dict[str, Any] = {"type": "object", "properties": properties}
    if schema.get("required"):
        out["required"] = schema["required"]
    return out


# Tool definitions in the OpenAI tool-calling format, generated from the registry.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.description,
            "parameters": _llm_schema(tool.args_model),
        },
    }
    for name, tool in TOOLS.items()
]


def _validation_message(exc: ValidationError) -> str:
    """One short line per invalid argument, readable by the LLM."""
    problems = [
        f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
        for err in exc.errors()
    ]
    return "Invalid arguments: " + "; ".join(problems)


def run_tool(cat: Catalog, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a whitelisted tool; errors are returned as `{"error": ...}`, not raised."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available: {sorted(TOOLS)}"}
    try:
        return tool.func(cat, tool.args_model.model_validate(args))
    except ValidationError as exc:
        return {"error": _validation_message(exc)}
    except ToolError as exc:
        return {"error": str(exc)}
    except Exception:
        logger.exception("Tool '%s' failed with args %s", name, args)
        return {"error": f"Tool '{name}' failed unexpectedly."}


def for_llm(result: dict[str, Any]) -> dict[str, Any]:
    """The part of a tool result the LLM sees: everything but the view action."""
    return {k: v for k, v in result.items() if k != VIEW_ACTION}
