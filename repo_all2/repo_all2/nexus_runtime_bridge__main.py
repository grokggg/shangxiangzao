from __future__ import annotations

import argparse
import logging

import uvicorn

from nexus.runtime.bridge.app import create_bridge_asgi_app


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Nexus LLM Bridge')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--request-timeout', type=float, default=300.0)
    parser.add_argument('--conversion-log-level', default='INFO')
    parser.add_argument('--thinking-config', default='hunyuan')
    parser.add_argument('--launch-id', default='')
    parser.add_argument('--log-level', default='info')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    app = create_bridge_asgi_app(
        request_timeout=args.request_timeout,
        conversion_log_level=args.conversion_log_level,
        thinking_config=args.thinking_config,
        launch_id=args.launch_id,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        # Allow large websocket frames (LLM responses); uvicorn defaults to 16 MiB.
        ws_max_size=100 * 1024 * 1024,
    )


if __name__ == '__main__':
    main()
