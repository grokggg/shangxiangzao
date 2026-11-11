#!/usr/bin/env python
# -*- coding:utf-8 -*-


import traceback
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)

from yuanbao_convert.drivers.cloud_sdk_driver import CloudSdkDriver
from yuanbao_convert.utils.log import logger
from yuanbao_convert.utils.exceptions import CallOCRServiceException


class OcrDriver(object):
    _component = "ocr"

    @classmethod
    def request(cls, **kwargs) -> dict:
        interface = kwargs.pop("interface", None)

        try:
            return CloudSdkDriver.request(cls._component, interface, **kwargs)
        except TencentCloudSDKException as ex:
            logger.error(
                "ocr exception: %s, traceback detail: %s"
                % (str(ex), traceback.format_exc())
            )
            if ex.get_code() == "FailedOperation.EmptyImageError":
                raise CallOCRServiceException.EmptyImageError()
            elif ex.get_code() == "FailedOperation.ImageSizeTooLarge":
                raise CallOCRServiceException.ImageSizeTooLarge()
            elif ex.get_code() == "LimitExceeded.TooLargeFileError":
                raise CallOCRServiceException.TooLargeFileError()
            else:
                # 统一返回调用OCR服务失败
                raise CallOCRServiceException()

    @classmethod
    def recognize_table_accurate_ocr(cls, **request_data):
        request_data["interface"] = "RecognizeTableAccurateOCR"
        return cls.request(**request_data)
