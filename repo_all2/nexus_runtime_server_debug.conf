"""Helpers for rendering and wiring the runtime server debug GUI page."""

from __future__ import annotations


def build_novnc_path(websocket_path: str, *, vnc_html_path: str = '/debug/gui/vnc/vnc.html') -> str:
    """Build the noVNC iframe URL for the runtime debug page."""
    return f'{vnc_html_path}?autoconnect=1&resize=remote&path={websocket_path}'


def validate_debug_port(port: int) -> None:
    """Validate a runtime-local TCP port used by the debug proxy."""
    if port < 1 or port > 65535:
        raise ValueError(f'invalid debug proxy port: {port}')


def aiomonitor_webui_port(loop_name: str, *, base_port: int) -> int:
    """Return the local aiomonitor web UI port for a named runtime loop."""
    if loop_name == 'main':
        return base_port
    raise ValueError('aiomonitor_loop_not_found')


def rewrite_aiomonitor_html(html: str, *, base_path: str) -> str:
    """Rewrite aiomonitor root-relative asset and API paths under a public proxy base."""
    base_path = base_path.rstrip('/')
    replacements = {
        'src="/': f'src="{base_path}/',
        'href="/': f'href="{base_path}/',
        'hx-get="/': f'hx-get="{base_path}/',
        'hx-post="/': f'hx-post="{base_path}/',
        'hx-delete="/': f'hx-delete="{base_path}/',
        'hx-put="/': f'hx-put="{base_path}/',
        'hx-patch="/': f'hx-patch="{base_path}/',
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    return html


def debug_port_origin(port: int) -> str:
    """Return the runtime-local HTTP origin for a debugged TCP port."""
    return f'http://127.0.0.1:{port}'


def debug_port_ws_origin(port: int) -> str:
    """Return the runtime-local WebSocket origin for a debugged TCP port."""
    return f'ws://127.0.0.1:{port}'


def debug_port_upstream_url(*, port: int, asset_path: str, query: str = '', websocket: bool = False) -> str:
    """Build the upstream URL for a runtime-local debug port proxy request."""
    path = asset_path.lstrip('/')
    origin = debug_port_ws_origin(port) if websocket else debug_port_origin(port)
    upstream_url = f'{origin}/{path}'
    if query:
        upstream_url = f'{upstream_url}?{query}'
    return upstream_url


def build_debug_page(*, novnc_path: str, websocket_path: str) -> str:
    """Render the lightweight runtime debug page that embeds noVNC."""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sandbox Desktop</title>
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        height: 100%;
        background: #111827;
        color: #e5e7eb;
        font-family: ui-sans-serif, system-ui, sans-serif;
      }}
      .layout {{
        height: 100%;
        display: grid;
        grid-template-rows: auto 1fr;
      }}
      .header {{
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }}
      .frame {{
        width: 100%;
        height: 100%;
        border: 0;
        background: #0b1220;
      }}
      code {{
        font-size: 12px;
      }}
    </style>
  </head>
  <body>
    <div class="layout">
      <div class="header">
        <strong>Sandbox Desktop</strong>
        <span>Browser-accessible desktop session · implementation: noVNC / VNC</span>
        <div><code data-role="websocket-path">{websocket_path}</code></div>
      </div>
      <iframe class="frame" title="Sandbox Desktop" data-fallback-src="{novnc_path}"></iframe>
    </div>
    <script>
      const debugGuiPath = location.pathname.replace(/\\/+$/, "");
      const websocketPath = `${{debugGuiPath}}/vnc/websockify`;
      const frameUrl = `${{debugGuiPath}}/vnc/vnc.html?autoconnect=1&resize=remote&path=${{encodeURIComponent(websocketPath)}}`;
      document.querySelector('[data-role="websocket-path"]').textContent = websocketPath;
      document.querySelector('.frame').src = frameUrl;
    </script>
  </body>
</html>
"""


__all__ = [
    'aiomonitor_webui_port',
    'build_debug_page',
    'build_novnc_path',
    'debug_port_origin',
    'debug_port_upstream_url',
    'debug_port_ws_origin',
    'rewrite_aiomonitor_html',
    'validate_debug_port',
]
