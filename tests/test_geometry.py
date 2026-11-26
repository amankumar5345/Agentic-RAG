"""Offline tests for the pure geometry helpers and the tool wrapper."""

import pytest
from pydantic import ValidationError

from src.core import geometry
from src.core.geometry import build_packet, replay, validate_commands, write_packet
from src.core.tools import dispatch_geometry


def test_angle_450_is_normalised_to_90():
    commands = validate_commands([{"shape": "line", "size": 10, "angle": 450}])
    assert commands[0].angle == 90.0


@pytest.mark.parametrize(
    "command",
    [
        {"shape": "line", "size": -1},
        {"shape": "triangle", "size": 10},
        {"shape": "line"},
        {"shape": "line", "size": "big"},
    ],
)
def test_invalid_commands_raise(command):
    with pytest.raises(ValidationError):
        validate_commands([command])


def test_size_above_env_limit_raises(monkeypatch):
    monkeypatch.setenv("LITHO_MAX_SIZE", "100")
    with pytest.raises(ValidationError):
        validate_commands([{"shape": "line", "size": 101}])

    commands = validate_commands([{"shape": "line", "size": 100}])
    assert commands[0].size == 100


def test_packet_schema():
    packet = build_packet([{"shape": "line", "size": 10}])
    assert packet["schema_version"] == "1.0"
    assert packet["commands"][0] == {"shape": "line", "size": 10.0, "angle": 0.0}
    assert "command_id" in packet and "created_at" in packet


def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    packet = build_packet([{"shape": "line", "size": 10}])

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(geometry.os, "replace", fail_replace)

    with pytest.raises(OSError):
        write_packet(packet, tmp_path)

    assert not (tmp_path / "unreal_mailbox.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_tool_missing_size_needs_clarification(tmp_path, monkeypatch):
    monkeypatch.setenv("LITHO_MAILBOX_DIR", str(tmp_path))

    result = dispatch_geometry(
        [
            {"shape": "line", "angle": 0.0},
            {"shape": "line", "size": 50.0, "angle": 90.0},
        ]
    )

    assert result["status"] == "needs_clarification"
    assert result["missing"] == [{"index": 0, "shape": "line"}]
    assert list(tmp_path.iterdir()) == []


def test_tool_rejects_unknown_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("LITHO_MAILBOX_DIR", str(tmp_path))

    result = dispatch_geometry([{"shape": "triangle", "size": 10}])

    assert result["status"] == "invalid"
    assert list(tmp_path.iterdir()) == []


def test_tool_writes_packet_and_creates_mailbox(tmp_path, monkeypatch):
    mailbox = tmp_path / "nested" / "mailbox"
    monkeypatch.setenv("LITHO_MAILBOX_DIR", str(mailbox))

    result = dispatch_geometry([{"shape": "line", "size": 10}])

    assert result["status"] == "ok"
    assert (mailbox / "unreal_mailbox.json").exists()


def test_replay_square_from_docstring():
    commands = [
        {"shape": "line", "size": 200.0, "angle": 0.0},
        {"shape": "line", "size": 200.0, "angle": 90.0},
        {"shape": "line", "size": 200.0, "angle": 180.0},
        {"shape": "line", "size": 200.0, "angle": 270.0},
    ]

    result = replay(commands)

    assert result["vertices"][0] == pytest.approx((0.0, 0.0))
    assert result["vertices"][1] == pytest.approx((200.0, 0.0), abs=1e-9)
    assert result["vertices"][2] == pytest.approx((200.0, 200.0), abs=1e-9)
    assert result["vertices"][3] == pytest.approx((0.0, 200.0), abs=1e-9)
    assert result["vertices"][4] == pytest.approx((0.0, 0.0), abs=1e-9)


def test_replay_spiral_from_docstring():
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
        {"shape": "line", "size": 50.0, "angle": 90.0},
    ]

    result = replay(commands)

    expected = [
        (0.0, 0.0),
        (500.0, 0.0),
        (500.0, 450.0),
        (100.0, 450.0),
        (100.0, 100.0),
        (400.0, 100.0),
        (400.0, 350.0),
        (200.0, 350.0),
        (200.0, 200.0),
        (300.0, 200.0),
        (300.0, 250.0),
    ]
    assert len(result["vertices"]) == len(expected)
    for got, want in zip(result["vertices"], expected):
        assert got == pytest.approx(want, abs=1e-9)


def test_replay_circle_returns_center_and_radius():
    result = replay(
        [
            {"shape": "line", "size": 10.0, "angle": 0.0},
            {"shape": "circle", "size": 5.0},
        ]
    )

    assert result["vertices"][0] == (0.0, 0.0)
    assert result["vertices"][1] == pytest.approx((10.0, 0.0), abs=1e-9)
    assert result["circles"] == [{"center": (10.0, 0.0), "radius": 5.0}]
