# Unreal acknowledgement contract

The geometry bridge is one-way today: Python writes `unreal_mailbox.json` and
Unreal renders it. To close the loop, Unreal writes one small file after a
packet has actually been rendered.

## Files and location

Both files live in the mailbox directory (`LITHO_MAILBOX_DIR`, default
`./runtime/unreal_mailbox`):

```
<mailbox_dir>/unreal_mailbox.json   # written by Python (src/core/geometry.py)
<mailbox_dir>/unreal_ack.json       # written by Unreal (this document)
```

## Ack schema

```json
{
  "command_id": "0b6f0c1e-8f5e-4a5e-9d0e-2d1c3b4a5f60",
  "received_at": "2026-11-26T13:01:02.123456+00:00",
  "rendered_at": "2026-11-26T13:01:02.987654+00:00"
}
```

- `command_id` must be copied verbatim from the packet just rendered.
- `received_at` is when the blueprint picked the packet up.
- `rendered_at` is when the actors/splines finished rendering.
- Both timestamps are UTC, ISO-8601. `src/services/ue_ack.py` only accepts an
  ack that has a matching `command_id` and a non-empty `rendered_at`.

## Blueprint change

In the blueprint that polls the mailbox on tick (the same one that reads
`unreal_mailbox.json`):

1. When a new packet arrives, parse and keep its `command_id`.
2. Record `received_at = FDateTime::UtcNow()` (or your file plugin's UTC now).
3. Render the spline/mesh actors as you do today.
4. After rendering, record `rendered_at = FDateTime::UtcNow()`.
5. Serialise the three fields as JSON and write them **atomically**:
   - write the text to `unreal_ack.json.tmp` in the same directory;
   - then rename/move it over `unreal_ack.json`.

   If your file plugin cannot rename, writing `unreal_ack.json` directly is
   tolerable, but a reader can then see a half-written file. `read_ack()`
   ignores malformed JSON and keeps waiting, so the rename is preferred but
   not strictly required for correctness.
6. Overwrite the ack for each new packet. Python matches on `command_id`, so
   stale acks are harmless.

The file must stay small and be rewritten, not appended.

## Python side

```python
from src.services.ue_ack import wait_for_ack

ack = wait_for_ack(command_id, timeout_s=5.0)
if ack is None:
    ...  # timed out
```

`wait_for_ack` accepts optional `mailbox_dir=` and `poll_interval_s=` for
tests. It is not called by `spawn_geometry` yet, so the tool does not block.

## Testing without Unreal

`tools/fake_unreal.py` emulates the blueprint:

```bash
# Ack whatever packet is present now
python tools/fake_unreal.py --once --delay 0

# Keep watching and ack every new command_id
python tools/fake_unreal.py --interval 0.1
```

Programmatically:

```python
from tools.fake_unreal import process_once

ack = process_once(mailbox_dir, delay_s=0.0)
```

`--mailbox-dir` defaults to `LITHO_MAILBOX_DIR` or `./runtime/unreal_mailbox`.
