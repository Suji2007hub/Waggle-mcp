from __future__ import annotations

import json
from pathlib import Path


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "apps" / "mcp" / "claude-desktop-extension" / "manifest.json"


def test_manifest_user_config_fields_are_wired_to_runtime_env() -> None:
    """Prevent silent drift between advertised Claude Desktop user_config and runtime env.

    The acceptance issue: manifest.json declares `user_config` fields, but the generated
    MCP runtime config (`server.mcp_config.env`) forwards only a subset.

    This test enforces: every `user_config` key must either be wired to an env var
    or be explicitly allowlisted as intentionally unused.
    """

    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    manifest = json.loads(raw)

    user_config = manifest.get("user_config")
    assert isinstance(user_config, dict), "manifest.json must define top-level user_config as an object"

    mcp_config = manifest.get("server", {}).get("mcp_config")
    assert isinstance(mcp_config, dict), "manifest.json must define server.mcp_config as an object"

    env = mcp_config.get("env")
    assert isinstance(env, dict), "manifest.json must define server.mcp_config.env as an object"

    # If a user_config field is intentionally unused by the runtime, add it here.
    intentionally_unused: set[str] = set()

    # Expected wiring pattern: ${user_config.<key>}
    # We accept direct or indirect string interpolation; we only need to know that
    # `<key>` is referenced somewhere in the env var mapping.
    referenced_user_config_keys: set[str] = set()
    for _env_name, env_value in env.items():
        if not isinstance(env_value, str):
            continue
        # cheap parse: look for '${user_config.' substrings
        # examples: '${user_config.db_path}'
        marker = "${user_config."
        start = 0
        while True:
            idx = env_value.find(marker, start)
            if idx == -1:
                break
            idx_key_start = idx + len(marker)
            idx_key_end = env_value.find("}", idx_key_start)
            if idx_key_end == -1:
                break
            key = env_value[idx_key_start:idx_key_end]
            referenced_user_config_keys.add(key)
            start = idx_key_end + 1

    missing = sorted(set(user_config.keys()) - intentionally_unused - referenced_user_config_keys)

    assert missing == [], (
        "manifest.json user_config keys must be wired into server.mcp_config.env. "
        f"Missing: {missing}. "
        "Either add env passthroughs, remove the unused user_config keys, "
        "or add them to intentionally_unused."
    )

