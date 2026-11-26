"""Offline tests for the Unreal ack contract and the fake receiver."""

import json

from src.core.geometry import build_packet, write_packet
from src.services.ue_ack import wait_for_ack
from tools.fake_unreal import process_once


def test_fake_unreal_round_trip(tmp_path):
    packet = build_packet([{"shape": "line", "size": 10}])
    write_packet(packet, tmp_path)

    ack = process_once(tmp_path, delay_s=0.0)

    assert ack is not None
    assert ack["command_id"] == packet["command_id"]
    assert ack["received_at"] and ack["rendered_at"]

    received = wait_for_ack(
        packet["command_id"],
        timeout_s=1.0,
        mailbox_dir=tmp_path,
        poll_interval_s=0.01,
    )

    assert received == ack


def test_wait_for_ack_times_out_on_unknown_id(tmp_path):
    packet = build_packet([{"shape": "line", "size": 10}])
    write_packet(packet, tmp_path)
    process_once(tmp_path, delay_s=0.0)

    result = wait_for_ack(
        "different-id",
        timeout_s=0.05,
        mailbox_dir=tmp_path,
        poll_interval_s=0.01,
    )

    assert result is None


def test_malformed_ack_is_ignored(tmp_path):
    (tmp_path / "unreal_ack.json").write_text("{not json", encoding="utf-8")

    result = wait_for_ack(
        "any-id",
        timeout_s=0.05,
        mailbox_dir=tmp_path,
        poll_interval_s=0.01,
    )

    assert result is None


def test_ack_file_shape(tmp_path):
    packet = build_packet([{"shape": "line", "size": 10}])
    write_packet(packet, tmp_path)
    process_once(tmp_path, delay_s=0.0)

    ack = json.loads((tmp_path / "unreal_ack.json").read_text(encoding="utf-8"))

    assert set(ack) == {"command_id", "received_at", "rendered_at"}
