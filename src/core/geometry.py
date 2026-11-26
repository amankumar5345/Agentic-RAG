"""Pure geometry helpers for the Unreal bridge.

Nothing here talks to LangChain, Streamlit or the network, so every function
is unit-testable on its own.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"
DEFAULT_MAX_SIZE = 5000.0
PACKET_FILENAME = "unreal_mailbox.json"


def max_size() -> float:
    """Largest allowed ``size`` value, read from LITHO_MAX_SIZE at call time."""
    raw = os.environ.get("LITHO_MAX_SIZE", "").strip()
    if not raw:
        return DEFAULT_MAX_SIZE
    return float(raw)


class ShapeCommand(BaseModel):
    shape: Literal["line", "polygon", "circle"]
    size: float = Field(gt=0, description="Length for line/polygon, radius for circle.")
    angle: float = Field(default=0.0, description="Absolute angle; 0 = right, 90 = up.")

    @field_validator("angle")
    @classmethod
    def _normalise_angle(cls, value: float) -> float:
        return float(value) % 360.0


class _CommandList(BaseModel):
    commands: list[ShapeCommand]

    @field_validator("commands")
    @classmethod
    def _enforce_max_size(cls, commands: list[ShapeCommand]) -> list[ShapeCommand]:
        limit = max_size()
        for index, command in enumerate(commands):
            if command.size > limit:
                raise ValueError(
                    f"commands[{index}].size {command.size} exceeds LITHO_MAX_SIZE={limit:g}"
                )
        return commands


def validate_commands(commands: Sequence[dict | ShapeCommand]) -> list[ShapeCommand]:
    """Validate and normalise a list of drawing commands.

    Raises ``pydantic.ValidationError`` for an unknown shape, a missing size,
    a size <= 0, a size above ``LITHO_MAX_SIZE``, or a non-numeric field.
    Angles are wrapped into [0, 360).
    """
    return _CommandList(commands=list(commands)).commands


def build_packet(commands: Sequence[dict | ShapeCommand]) -> dict:
    """Build the JSON packet Unreal consumes. Validates the commands first."""
    validated = validate_commands(commands)
    return {
        "schema_version": SCHEMA_VERSION,
        "command_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commands": [command.model_dump() for command in validated],
    }


def write_packet(packet: dict, mailbox_dir: str | os.PathLike) -> Path:
    """Write ``packet`` atomically to <mailbox_dir>/unreal_mailbox.json."""
    directory = Path(mailbox_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / PACKET_FILENAME

    fd, temp_name = tempfile.mkstemp(dir=directory, prefix=".unreal_mailbox.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(packet, handle, indent=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


def replay(commands: Sequence[dict | ShapeCommand], origin: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Replay commands as a polyline with absolute angles (0 = right, 90 = up).

    ``line`` and ``polygon`` advance the pen by ``size`` at ``angle``; the
    command schema has no vertex list, so a polygon is replayed as one edge.
    A ``circle`` records its centre (the current pen position) and ``size`` as
    its radius, and does not move the pen.

    Returns ``{"vertices": [(x, y), ...], "circles": [{"center": (x, y), "radius": r}]}``.
    The first vertex is ``origin``.
    """
    validated = validate_commands(commands)
    x, y = float(origin[0]), float(origin[1])
    vertices: list[tuple[float, float]] = [(x, y)]
    circles: list[dict] = []

    for command in validated:
        if command.shape == "circle":
            circles.append({"center": (x, y), "radius": command.size})
            continue

        radians = math.radians(command.angle)
        x += command.size * math.cos(radians)
        y += command.size * math.sin(radians)
        vertices.append((x, y))

    return {"vertices": vertices, "circles": circles}
