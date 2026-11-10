#!/usr/bin/env python
# -*- coding:utf-8 -*-

import inspect

import requests
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos.cos_exception import CosClientError, CosServiceError

from yuanbao_convert.adapter import AGSGatewayAdapter, AGSGatewayType
from yuanbao_convert.etc import conf as CONF
from yuanbao_convert.utils.log import logger
from yuanbao_convert.utils import exceptions
from yuanbao_convert.utils.string_util import transform_string_format_in_recursion


class CosSdkDriver(object):
    @classmethod
    def _pack(cls, **kwargs):
        return transform_string_format_in_recursion(
            kwargs, CONF.STRING_FORMAT_PASCALCASE
        )

    @classmethod
    def _unpack(cls, result, ignore_keys=None):
        return transform_string_format_in_recursion(
            result, CONF.STRING_FORMAT_CAMELCASE, ignore_keys
        )

    @classmethod
    def _get_session_with_gateway(cls):
        """
        Get a requests session configured with AGS Gateway adapter.
        
        Returns:
            requests.Session: Session with gateway adapter mounted, or regular session if no gateway configured
        """
        session = requests.Session()
        gateway_endpoint = CONF.AGS_GATEWAY_ENDPOINT
        if not gateway_endpoint:
            return session
        adapter = AGSGatewayAdapter(gateway_endpoint, AGSGatewayType.COS)
        session.mount("http://", adapter=adapter)
        session.mount("https://", adapter=adapter)
        return session

    @classmethod
    def request(cls, interface: str, **kwargs) -> dict:
        region = kwargs.pop("region", None)
        if not region:
            region = CONF.DEFAULT_REGION

        # If AGS Gateway is configured, use gateway mode
        if CONF.AGS_GATEWAY_ENDPOINT:
            config = CosConfig(
                Region=region,
                SecretId=CONF.TENCENT_SECRET_ID,
                SecretKey=CONF.TENCENT_SECRET_KEY,
            )
            client = CosS3Client(config, session=cls._get_session_with_gateway())
        elif CONF.AUTH_PROXY:
            # Legacy AUTH_PROXY mode
            if interface in [
                "ci_create_doc_job",
                "ci_get_doc_job",
            ]:
                domain = "%s/tencent_cloud/cos_ci" % CONF.AUTH_PROXY
            else:
                domain = "%s/tencent_cloud/cos" % CONF.AUTH_PROXY
            config = CosConfig(
                Region=region,
                SecretId=CONF.TENCENT_SECRET_ID,
                SecretKey=CONF.TENCENT_SECRET_KEY,
                Domain=domain,
                Scheme="http",
            )
            client = CosS3Client(config)
        else:
            # Direct mode without proxy
            config = CosConfig(
                Region=region,
                SecretId=CONF.TENCENT_SECRET_ID,
                SecretKey=CONF.TENCENT_SECRET_KEY,
            )
            client = CosS3Client(config)

        method = getattr(client, interface, None)

        request_data = cls._pack(**kwargs)
        # 提取cos接口所需的参数
        method_signature = inspect.signature(method)
        valid_params = {
            k: v for k, v in request_data.items() if k in method_signature.parameters
        }
        response = method(**valid_params)
        return cls._unpack(response)

    @classmethod
    def create_doc_process_jobs(cls, **kwargs):
        # CreateDocProcessJobs
        return cls.request("ci_create_doc_job", **kwargs)

    @classmethod
    def describe_doc_process_job(cls, bucket, job_id):
        # DescribeDocProcessJob
        req_data = {
            "bucket": bucket,
            "jobID": job_id,
        }
        return cls.request("ci_get_doc_job", **req_data)

    @classmethod
    def get_download_url(cls, bucket, key, expire_seconds=0):
        # GetPresignedURL
        req_data = {
            "method": "GET",
            "bucket": bucket,
            "key": key,
        }
        if expire_seconds > 0:
            req_data["expired"] = expire_seconds
        return cls.request("get_presigned_url", **req_data)

    @classmethod
    def list_objects(cls, bucket, prefix="", delimiter="", marker=""):
        # prefix：cos目录，delimiter：默认为空，如果为"/"则代表要列出prefix下的子目录
        req_data = {
            "bucket": bucket,
            "prefix": prefix,
            "delimiter": delimiter,
            "marker": marker,
        }
        return cls.request("list_objects", **req_data)

    @classmethod
    def download_file(cls, bucket, key, dest_filepath):
        # Get Object
        req_data = {
            "bucket": bucket,
            "key": key,
            "destFilePath": dest_filepath,
        }
        success_flag = False
        # 使用高级接口断点续传，失败重试时不会下载已成功的分块
        for _ in range(CONF.DEFAULT_COS_DOWNLOAD_RETRY_TIME):
            try:
                _ = cls.request("download_file", **req_data)
                success_flag = True
                break
            except CosClientError or CosServiceError as err:
                logger.error(f"download {key} from cos happens exception {err}")
        if not success_flag:
            raise exceptions.CallCOSServiceException.DownloadCOSFileError(file=key)


if __name__ == "__main__":
    data = {
        "bucket": "example-1251001002",
        "key": "example.pdf",
        "expire_seconds": 3600,
    }
    print(CosSdkDriver.get_download_url(**data))
