from __future__ import annotations

import uvicorn

from infra.logging import configure_logging
from label.app import create_app
from label.config import LabelConfig
from nexus.utils.otel import initialize_if_enabled


def main() -> None:
    # 只读启动期参数（端口/日志级别要先确定）；provider 业务配置由
    # lifespan 在七彩石客户端就绪后加载，因此这里无需事件循环。
    config = LabelConfig.from_env()
    configure_logging(
        config.service_name,
        level=config.log_level,
        log_file=config.log_file,
        backup_count=config.log_backup_count,
    )
    initialize_if_enabled()
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        log_config=None,
    )


if __name__ == '__main__':
    main()
