from __future__ import annotations

import json


def render_graph_editor_html(
    *,
    mode: str = "edit",
    project: str = "",
    agent_id: str = "",
    session_id: str = "",
) -> str:
    page_mode = "view" if mode.strip().lower() == "view" else "edit"
    config = json.dumps(
        {
            "mode": page_mode,
            "project": project,
            "agent_id": agent_id,
            "session_id": session_id,
        }
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Waggle Graph Studio</title>
  <link rel="stylesheet" href="/graph-assets/app.css">
</head>
<body>
  <div id="root"></div>
  <script>
    window.__WAGGLE_GRAPH_CONFIG__ = {config};
  </script>
  <script type="module" src="/graph-assets/app.js"></script>
</body>
</html>"""
