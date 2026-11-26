"""Session mode, page setup, sidebar metadata and live-mode limits."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv()


def _apply_secrets() -> None:
    try:
        for key, value in st.secrets.items():
            os.environ.setdefault(str(key), str(value))
    except Exception:
        pass


def resolve_mode() -> tuple[str, str]:
    requested = os.environ.get("LITHO_MODE", "").strip().lower()
    has_keys = bool(os.environ.get("GROQ_API_KEY")) and bool(os.environ.get("DEEPSEEK_API_KEY"))
    if requested in {"live", "demo"}:
        mode = requested
    else:
        mode = "live" if has_keys else "demo"
    if mode == "live" and not has_keys:
        return "demo", "missing API keys"
    return mode, ""


def _meta() -> dict:
    import json

    for name in ("A5.json", "comparison.json"):
        path = REPO_ROOT / "results" / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def setup_page(title: str) -> None:
    try:
        st.set_page_config(page_title=f"Litho Agentic RAG · {title}", layout="wide")
    except Exception:
        pass

    _apply_secrets()
    mode, reason = resolve_mode()
    st.session_state.setdefault("mode", mode)
    st.session_state.setdefault("mode_reason", reason)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("rate_times", [])

    if mode == "live":
        st.success("LIVE mode — real Groq, DeepSeek and Ollama calls")
    else:
        detail = f" ({reason})" if reason else ""
        st.info(f"DEMO mode — replaying recorded results and cache{detail}")

    meta = _meta()
    models = meta.get("models", {})
    with st.sidebar:
        st.header("Litho Agentic RAG")
        st.caption(f"Mode: **{mode.upper()}**")
        st.caption(f"Generator: `{models.get('generator', 'not recorded')}`")
        st.caption(f"Judge: `{models.get('judge', 'not recorded')}`")
        st.caption(f"Dataset: `{meta.get('dataset_hash', 'not recorded')}`")
        st.caption(f"Results commit: `{meta.get('git_commit', 'not recorded')}`")
        st.divider()
        st.caption("No API keys are rendered. Prompt logging is off unless LITHO_LOG=1.")


def allow_live_call() -> bool:
    if st.session_state.get("mode") != "live":
        return True
    limit = int(os.environ.get("LITHO_RATE_LIMIT", "20"))
    now = time.time()
    times = [stamp for stamp in st.session_state.rate_times if now - stamp < 60]
    if len(times) >= limit:
        st.warning("Live-mode rate limit reached for this session. Try again shortly.")
        st.session_state.rate_times = times
        return False
    times.append(now)
    st.session_state.rate_times = times
    return True


def log_prompt(prompt: str) -> None:
    if os.environ.get("LITHO_LOG") != "1":
        return
    path = REPO_ROOT / ".cache" / "ui_prompts.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prompt.replace("\n", " ") + "\n")
