"""SDK 异常类型。"""

from __future__ import annotations


class AccountPoolError(Exception):
    """SDK 通用基类异常。"""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.code is not None:
            return f'[code={self.code}] {self.message}'
        return self.message


class AcquireError(AccountPoolError):
    """acquire 失败。"""


class NoAvailableAccountError(AcquireError):
    """池中没有可用账号（区别于其他 acquire 错误，便于业务重试判断）。"""


class ReleaseError(AccountPoolError):
    """release 失败。"""


class HeartbeatError(AccountPoolError):
    """心跳续租失败。"""


class ReservationError(AccountPoolError):
    """批量预留 / 取消预留失败。"""


class InvalidateError(AccountPoolError):
    """账号置无效 / 恢复失败。"""
