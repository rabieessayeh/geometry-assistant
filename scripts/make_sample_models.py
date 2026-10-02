"""Generate the sample parts of the demo, procedurally (no third-party model).

Three original parts are built from boxes and cylinders with boolean
operations, in millimetres, resting on the plane z = 0:

- `l_bracket`: an L-shaped bracket with one hole in each flange;
- `bridge`: two pillars carrying a deck, whose underside needs support when printed;
- `enclosure`: an open box with thin walls.

Their dimensions are module constants, so the tests can check the tools
against volumes and areas computed by hand.

Usage:  python scripts/make_sample_models.py [--out data/samples]
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import box, cylinder

logger = logging.getLogger("make_sample_models")

ROOT = Path(__file__).resolve().parents[1]

# L-bracket: a base flange along x and an upright flange along z, same thickness.
BRACKET_LENGTH = 60.0  # x
BRACKET_WIDTH = 40.0  # y
BRACKET_HEIGHT = 40.0  # z
BRACKET_THICKNESS = 5.0
HOLE_RADIUS = 3.0
HOLE_SECTIONS = 32  # the holes are 32-sided prisms
BASE_HOLE_XY = (40.0, 20.0)
UPRIGHT_HOLE_YZ = (20.0, 25.0)

# Bridge: two pillars and a deck spanning them.
PILLAR_SIZE = (10.0, 20.0, 20.0)
BRIDGE_LENGTH = 50.0
DECK_THICKNESS = 5.0

# Enclosure: an open box.
ENCLOSURE_SIZE = (80.0, 50.0, 30.0)
ENCLOSURE_WALL = 2.0


def _box(lower: tuple[float, float, float], upper: tuple[float, float, float]) -> trimesh.Trimesh:
    """Axis-aligned box between two corners."""
    return box(bounds=np.array([lower, upper]))


def _drill(centre: tuple[float, float, float], axis: str) -> trimesh.Trimesh:
    """Cylinder used to cut a hole, longer than the flange it goes through."""
    transform = np.eye(4)
    if axis == "x":
        transform = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
    transform[:3, 3] = centre
    return cylinder(
        radius=HOLE_RADIUS,
        height=4 * BRACKET_THICKNESS,
        sections=HOLE_SECTIONS,
        transform=transform,
    )


def l_bracket() -> trimesh.Trimesh:
    """L-shaped bracket with a hole through each flange."""
    t = BRACKET_THICKNESS
    base = _box((0, 0, 0), (BRACKET_LENGTH, BRACKET_WIDTH, t))
    upright = _box((0, 0, 0), (t, BRACKET_WIDTH, BRACKET_HEIGHT))
    body = trimesh.boolean.union([base, upright])
    holes = [
        _drill((*BASE_HOLE_XY, t / 2), axis="z"),
        _drill((t / 2, *UPRIGHT_HOLE_YZ), axis="x"),
    ]
    return trimesh.boolean.difference([body, *holes])


def bridge() -> trimesh.Trimesh:
    """Two pillars carrying a deck: the underside of the deck is a 90 degree overhang."""
    px, py, pz = PILLAR_SIZE
    left = _box((0, 0, 0), (px, py, pz))
    right = _box((BRIDGE_LENGTH - px, 0, 0), (BRIDGE_LENGTH, py, pz))
    deck = _box((0, 0, pz), (BRIDGE_LENGTH, py, pz + DECK_THICKNESS))
    return trimesh.boolean.union([left, right, deck])


def enclosure() -> trimesh.Trimesh:
    """Open box: a block with a cavity cut from the top."""
    sx, sy, sz = ENCLOSURE_SIZE
    w = ENCLOSURE_WALL
    outer = _box((0, 0, 0), (sx, sy, sz))
    # The cavity is taller than the block so the top face is cut cleanly.
    cavity = _box((w, w, w), (sx - w, sy - w, 2 * sz))
    return trimesh.boolean.difference([outer, cavity])


SAMPLES: dict[str, Callable[[], trimesh.Trimesh]] = {
    "l_bracket": l_bracket,
    "bridge": bridge,
    "enclosure": enclosure,
}


def write_samples(out_dir: Path) -> list[Path]:
    """Build every sample and write it as a binary STL file; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build in SAMPLES.items():
        mesh = build()
        path = out_dir / f"{name}.stl"
        mesh.export(path)
        logger.info(
            "%s: %d faces, volume %.1f mm3, watertight=%s",
            path,
            len(mesh.faces),
            mesh.volume,
            mesh.is_watertight,
        )
        paths.append(path)
    return paths


def main() -> None:
    """Write the sample models to the output folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "samples")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    write_samples(args.out)


if __name__ == "__main__":
    main()
