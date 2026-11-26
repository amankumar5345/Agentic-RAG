"""Fake Unreal receiver for tests and CI.

Watches the geometry mailbox and, after a configurable delay, writes the ack
file the real Unreal blueprint is expected to write. Never touches the network.

Usage:
    python tools/fake_unreal.py --once --delay 0.2
    python tools/fake_unreal.py --interval 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

PACKET_FILENAME = "unreal_mailbox.json"
ACK_FILENAME = "unreal_ack.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_packet(mailbox_dir: str | os.PathLike) -> dict | None:
    path = Path(mailbox_dir).expanduser() / PACKET_FILENAME
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_ack(
    mailbox_dir: str | os.PathLike,
    command_id: str,
    delay_s: float = 0.0,
) -> dict:
    """Ack ``command_id`` after ``delay_s`` seconds, atomically."""
    directory = Path(mailbox_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    received_at = _now()
    if delay_s > 0:
        time.sleep(delay_s)

    payload = {
        "command_id": str(command_id),
        "received_at": received_at,
        "rendered_at": _now(),
    }

    target = directory / ACK_FILENAME
    temp = directory / f".{ACK_FILENAME}.tmp"
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, target)
    return payload


def process_once(
    mailbox_dir: str | os.PathLike,
    delay_s: float = 0.0,
) -> dict | None:
    """Ack the current packet once. Returns the ack, or None if no packet."""
    packet = read_packet(mailbox_dir)
    if not packet or not packet.get("command_id"):
        return None
    return write_ack(mailbox_dir, packet["command_id"], delay_s)


def watch(
    mailbox_dir: str | os.PathLike,
    delay_s: float = 0.0,
    interval_s: float = 0.1,
) -> None:
    """Ack every new command_id until interrupted."""
    last_command_id = None
    while True:
        packet = read_packet(mailbox_dir)
        command_id = packet.get("command_id") if packet else None
        if command_id and command_id != last_command_id:
            write_ack(mailbox_dir, command_id, delay_s)
            last_command_id = command_id
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake Unreal ack writer")
    parser.add_argument(
        "--mailbox-dir",
        default=os.environ.get("LITHO_MAILBOX_DIR", "./runtime/unreal_mailbox"),
    )
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        ack = process_once(args.mailbox_dir, args.delay)
        print(json.dumps(ack, indent=2) if ack else "no packet found")
        return

    watch(args.mailbox_dir, args.delay, args.interval)


if __name__ == "__main__":
    main()
