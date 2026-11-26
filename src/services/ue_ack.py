"""Wait for Unreal Engine to acknowledge a rendered command packet.

The bridge contract is a JSON file beside the mailbox:

    <mailbox_dir>/unreal_ack.json
    {
      "command_id": "<uuid from unreal_mailbox.json>",
      "received_at": "<UTC ISO-8601>",
      "rendered_at": "<UTC ISO-8601>"
    }
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

DEFAULT_POLL_INTERVAL_S = 0.1


def resolve_mailbox_dir(mailbox_dir: str | os.PathLike | None = None) -> Path:
    if mailbox_dir is not None:
        return Path(mailbox_dir).expanduser()
    return Path(os.environ.get("LITHO_MAILBOX_DIR", "./runtime/unreal_mailbox")).expanduser()


def ack_path(mailbox_dir: str | os.PathLike | None = None) -> Path:
    return resolve_mailbox_dir(mailbox_dir) / "unreal_ack.json"


def read_ack(mailbox_dir: str | os.PathLike | None = None) -> dict | None:
    path = ack_path(mailbox_dir)
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def wait_for_ack(
    command_id: str,
    timeout_s: float,
    mailbox_dir: str | os.PathLike | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> dict | None:
    """Poll for an ack matching ``command_id`` with a ``rendered_at`` timestamp.

    Returns the ack dict, or ``None`` once ``timeout_s`` expires.
    """
    deadline = time.monotonic() + float(timeout_s)

    while True:
        ack = read_ack(mailbox_dir)
        if (
            ack
            and str(ack.get("command_id")) == str(command_id)
            and ack.get("rendered_at")
        ):
            return ack

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(poll_interval_s, remaining))
