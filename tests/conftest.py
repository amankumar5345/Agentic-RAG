"""Offline test bootstrap.

Dummy credentials are set before anything from ``src`` is imported because
``src/core/graph.py`` builds its ChatGroq clients at import time and calls
``load_dotenv(override=True)``. Tests must never reach the network.

``LITELLM_LOCAL_MODEL_COST_MAP`` is litellm's supported offline switch: without
it, importing litellm downloads its model cost map over HTTP.
"""

import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
os.environ.setdefault("LITHO_MODE", "demo")
