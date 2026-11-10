#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os


TOOL_NAME = "yuanbao_convert"


def _default_sandbox_root_dir():
    """Prefer coding-agent layout (/data/workspace), then legacy /workspace."""
    for candidate in ("/data/workspace", "/workspace"):
        if os.path.isdir(candidate):
            return candidate.rstrip("/") + "/"
    return "/workspace/"


# 文件路径
SANDBOX_ROOT_DIR = os.environ.get("SANDBOX_ROOT_DIR") or _default_sandbox_root_dir()
COS_URL_MAPPING_FILE_SUBDIR = os.environ.get("COS_URL_MAPPING_FILE_SUBDIR", ".mapping/")
# 字符串格式化范式
STRING_FORMAT_PASCALCASE = "pascalcase"  # 首字母大写，驼峰
STRING_FORMAT_CAMELCASE = "camelcase"  # 首字母小写，驼峰
# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": f"[%(asctime)s] [%(levelname)s] [%(lineno)d] %(message)s",
        },
        "simple": {
            "format": f"%(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "filters": {},
    "handlers": {
        "stream_info": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",  # 输出到标准输出
        },
        "stream_error": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "simple",
            "stream": "ext://sys.stderr",  # 输出到标准错误
        },
        "logFile": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "WARNING",
            "formatter": "standard",
            "filename": os.path.join("/tmp", f"{TOOL_NAME}.log"),
            "when": "D",
            "interval": 1,
            "backupCount": 10,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "logFile": {
            "handlers": ["logFile"],
            "level": "WARNING",
        },
        "streamFile": {
            "handlers": ["stream_info", "stream_error"],
            "level": "INFO",
        },
        # 禁用第三方库的日志输出
        "pikepdf": {
            "handlers": [],
            "level": "WARNING",
            "propagate": False,
        },
        "PIL": {
            "handlers": [],
            "level": "WARNING",
            "propagate": False,
        },
        "numexpr": {
            "handlers": [],
            "level": "WARNING",
            "propagate": False,
        },
    },
}

# ========================== 腾讯云配置 ==========================

TENCENT_SECRET_ID = os.environ.get("TENCENT_SECRET_ID", "unknown")
TENCENT_SECRET_KEY = os.environ.get("TENCENT_SECRET_KEY", "unknown")
DEFAULT_REGION = os.environ.get("DEFAULT_REGION", "ap-guangzhou")
DEFAULT_COS_BUCKET = os.environ.get("DEFAULT_COS_BUCKET", "hunyuan-test-1258344703")
DEFAULT_COS_URL_EXPIRE_SEC = os.environ.get(
    "DEFAULT_COS_URL_EXPIRE_SEC", 3600 * 24 * 90
)
DEFAULT_COS_DOWNLOAD_RETRY_TIME = os.environ.get("DEFAULT_COS_DOWNLOAD_RETRY_TIME", 5)
DEFAULT_COS_TASK_TIMEOUT = os.environ.get(
    "DEFAULT_COS_TASK_TIMEOUT", 300
)  # default 5min
AUTH_PROXY = os.environ.get("X_IDE_AUTH_PROXY", "")
AGS_GATEWAY_ENDPOINT = os.environ.get("AGS_GATEWAY_ENDPOINT", "")
# AGS Gateway 上 yuanbao open API 的 target 名称（/targets/<name>/...）
AGS_GATEWAY_YUANBAO_OPEN_API_TARGET = os.environ.get(
    "AGS_GATEWAY_YUANBAO_OPEN_API_TARGET", "yuanbao-one-agent-yuanbao-open-api"
)

# ========================== 腾讯文档配置 ==========================

# 代理配置
TENCENT_DOCS_PROXY_PREFIX = os.environ.get(
    "TENCENT_DOCS_PROXY_PREFIX", "http://auth.proxy/yuanbao"
)  # 代理前缀
TENCENT_DOCS_COS_PROXY_HEADER = "x-cs-doc-cos-host"  # COS 代理请求头名称

# COS 代理地址（可通过环境变量覆盖，默认使用 PROXY_PREFIX + /cos）
TENCENT_DOCS_COS_PROXY_URL = os.environ.get(
    "TENCENT_DOCS_COS_PROXY_URL", f"{TENCENT_DOCS_PROXY_PREFIX}/cos"
)

# 训练沙箱等无法访问 COS 公网时：直连预签名 URL 建连改走内网域名，Host 仍用原公网 host 以保留签名
# 支持 1/true/yes/on（大小写不敏感）
TENCENT_DOCS_COS_USE_INTERNAL = os.environ.get(
    "TENCENT_DOCS_COS_USE_INTERNAL", ""
).strip().lower() in ("1", "true", "yes", "on")

# 文档转换接口地址（可通过环境变量覆盖，默认使用 PROXY_PREFIX + 路径）
TENCENT_DOCS_API_BASE_URL = os.environ.get(
    "TENCENT_DOCS_API_BASE_URL",
    f"{TENCENT_DOCS_PROXY_PREFIX}/doc/open_api/doc_proxy/openapi/drive/v2/files",
)

# 测试环境配置
TENCENT_DOCS_ENV_ID = os.environ.get("TENCENT_DOCS_ENV_ID", "")  # 测试环境ID
TENCENT_DOCS_ROUTE_ENV = os.environ.get("TENCENT_DOCS_ROUTE_ENV", "")  # 测试路由环境

# 超时配置（秒）
TENCENT_DOCS_TIMEOUT = int(
    os.environ.get("TENCENT_DOCS_TIMEOUT", "60")
)  # 单次请求超时（上传/提交/查询/下载）
TENCENT_DOCS_TOTAL_TIMEOUT = int(
    os.environ.get("TENCENT_DOCS_TOTAL_TIMEOUT", "300")
)  # 阻塞转换整体超时

# 元宝开放 API 地址（AGS Gateway / 直连加签模式使用，可经环境变量覆盖）
YUANBAO_OPEN_API_BASE_URL = os.environ.get(
    "YUANBAO_OPEN_API_BASE_URL",
    "https://open-api.yuanbao.tencent.com/open_api/doc_proxy/openapi/drive/v2/files",
)

# 直连加签（不经过 AGS Gateway；见 SIGNING.md）
YUANBAO_APP_KEY = os.environ.get("YUANBAO_APP_KEY", "")
YUANBAO_SECRET = os.environ.get("YUANBAO_SECRET", "")
