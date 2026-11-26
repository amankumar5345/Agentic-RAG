import json
import os
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from src.core.geometry import build_packet, write_packet


def _mailbox_dir() -> Path:
    """Directory Unreal polls for geometry payloads."""
    return Path(os.environ.get("LITHO_MAILBOX_DIR", "./runtime/unreal_mailbox")).expanduser()


def _missing_sizes(commands: list[dict]) -> list[dict]:
    missing = []
    for index, command in enumerate(commands):
        if isinstance(command, dict) and command.get("size") is None:
            missing.append({"index": index, "shape": command.get("shape")})
    return missing


def dispatch_geometry(commands: list[dict]) -> dict:
    """Validate commands, write the packet, and return a structured result.

    Returns ``status`` one of ``ok``, ``needs_clarification``, ``invalid`` or
    ``error``. Nothing is written unless the result is ``ok``.
    """
    missing = _missing_sizes(commands)
    if missing:
        return {
            "status": "needs_clarification",
            "reason": "missing_size",
            "message": (
                "No size was provided for the listed command(s), so nothing was drawn. "
                "Ask the user for the size and do not guess it."
            ),
            "missing": missing,
        }

    try:
        packet = build_packet(commands)
    except ValidationError as exc:
        return {
            "status": "invalid",
            "message": "The geometry commands did not validate, so nothing was drawn.",
            "errors": [
                {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                for error in exc.errors()
            ],
        }

    try:
        mailbox_path = write_packet(packet, _mailbox_dir())
    except OSError as exc:
        return {
            "status": "error",
            "reason": "write_failed",
            "message": f"Could not write the geometry packet: {exc}",
        }

    return {
        "status": "ok",
        "command_id": packet["command_id"],
        "mailbox_path": str(mailbox_path),
        "command_count": len(packet["commands"]),
    }


@tool
def spawn_geometry(commands: list[dict]) -> str:
    """
    Trigger this tool strictly when the user explicitly commands the system to draw, create, or spawn a geometric shape in the viewport.

    Rules:
    - NEVER guess or hallucinate dimensions.
    - If the user gives no size, omit 'size' (or set it to null). The tool returns
      {"status": "needs_clarification", ...}, draws nothing, and you must ask the
      user for the size before calling it again.
    - 'size' must be a positive number and must not exceed LITHO_MAX_SIZE.
    - All angles are ABSOLUTE with respect to the x-axis (0.0 = Right/Horizontal, 90.0 = Up/Vertical, 180.0 = Left, 270.0 = Down). Values are wrapped into [0, 360).
    - If the user does not specify an angle, omit it; the angle defaults to 0.0.

    Args:
        commands (list[dict]): A chronological list of drawing steps to execute. Each dictionary in the list MUST contain:
            - 'shape' (str): Exactly "line", "polygon", or "circle". Do not capitalize. (Map "square", "rectangle", "triangle" to "polygon" or break them into a series of "line" commands).
            - 'size' (float): The numerical length or radius. MUST be a positive float and <= LITHO_MAX_SIZE. Omit it when the user has not given one.
            - 'angle' (float, optional): The angle of the shape with respect to the x-axis in an anticlockwise direction. Extract directly from the user's prompt. Defaults to 0.0.

    Examples:
        User: "Draw a square of size 200"
        commands = [
            {"shape": "line", "size": 200.0, "angle": 0.0},
            {"shape": "line", "size": 200.0, "angle": 90.0},
            {"shape": "line", "size": 200.0, "angle": 180.0},
            {"shape": "line", "size": 200.0, "angle": 270.0}
        ]
        # -> {"status": "ok", ...}

        User: "Draw a square"
        commands = [
            {"shape": "line", "angle": 0.0},
            {"shape": "line", "angle": 90.0},
            {"shape": "line", "angle": 180.0},
            {"shape": "line", "angle": 270.0}
        ]
        # -> {"status": "needs_clarification", "reason": "missing_size", ...}
        # Ask: "What size should the square be?"

        User: "Draw a spiral starting from 500, reducing by 50 each time until it reaches 50."
        commands = [
            {"shape": "line", "size": 500.0, "angle": 0.0},
            {"shape": "line", "size": 450.0, "angle": 90.0},
            {"shape": "line", "size": 400.0, "angle": 180.0},
            {"shape": "line", "size": 350.0, "angle": 270.0},
            {"shape": "line", "size": 300.0, "angle": 0.0},
            {"shape": "line", "size": 250.0, "angle": 90.0},
            {"shape": "line", "size": 200.0, "angle": 180.0},
            {"shape": "line", "size": 150.0, "angle": 270.0},
            {"shape": "line", "size": 100.0, "angle": 0.0},
            {"shape": "line", "size": 50.0, "angle": 90.0}
        ]
    """
    return json.dumps(dispatch_geometry(commands))
