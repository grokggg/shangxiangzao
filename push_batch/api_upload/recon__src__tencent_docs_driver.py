#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
腾讯文档开放平台 Driver

公开方法：
- local_convert_upload: 获取 COS 上传信息
- upload_file_to_cos: 将文件 PUT 到腾讯文档返回的 COS 链接
- local_convert_submit_task: 提交异步转换任务
- local_convert_upload_progress: 查询异步任务进度
- blocking_convert: 预上传 -> 上传 -> 提交 -> 轮询 -> 下载，返回结果
"""

import re
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests

from yuanbao_convert.adapter import (
    AGSGatewayAdapter,
    AGSGatewayType,
    YuanbaoSignAdapter,
)
from yuanbao_convert.etc import conf as CONF
from yuanbao_convert.utils.exceptions import CallTencentDocsException
from yuanbao_convert.utils.log import logger


class TencentDocsDriver(object):
    _component = "tencent_docs"
    # 沙箱环境对应的腾讯文档地址
    _base_url = CONF.TENCENT_DOCS_API_BASE_URL
    # 元宝开放 API 地址（网关 / 直连加签；可经 YUANBAO_OPEN_API_BASE_URL 配置）
    _yuanbao_open_api_base_url = CONF.YUANBAO_OPEN_API_BASE_URL
    _default_poll_interval = 2  # 轮询查询间隔

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
        total_timeout: Optional[int] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        初始化驱动实例，配置通用参数，避免在各接口重复传参。
        """
        # 网关或直连加签：使用可配置的元宝开放 API 地址
        if CONF.AGS_GATEWAY_ENDPOINT or (
            CONF.YUANBAO_APP_KEY and CONF.YUANBAO_SECRET
        ):
            self.base_url = base_url or CONF.YUANBAO_OPEN_API_BASE_URL
        else:
            self.base_url = base_url or CONF.TENCENT_DOCS_API_BASE_URL
        self.timeout = (
            timeout if timeout is not None else CONF.TENCENT_DOCS_TIMEOUT
        )
        self.poll_interval = (
            poll_interval
            if poll_interval is not None
            else self._default_poll_interval
        )
        self.total_timeout = (
            total_timeout
            if total_timeout is not None
            else CONF.TENCENT_DOCS_TOTAL_TIMEOUT
        )
        self.default_headers = default_headers or {}

        self._session = self._create_session()

    def _use_direct_sign(self) -> bool:
        return bool(
            not CONF.AGS_GATEWAY_ENDPOINT
            and CONF.YUANBAO_APP_KEY
            and CONF.YUANBAO_SECRET
        )

    def _use_direct_cos(self) -> bool:
        return bool(CONF.AGS_GATEWAY_ENDPOINT or self._use_direct_sign())

    @staticmethod
    def _use_direct_cos_static() -> bool:
        return bool(
            CONF.AGS_GATEWAY_ENDPOINT
            or (CONF.YUANBAO_APP_KEY and CONF.YUANBAO_SECRET)
        )

    def _create_session(self) -> requests.Session:
        """
        创建 requests session：AGS Gateway 或直连加签时挂载对应适配器。
        """
        session = requests.Session()
        gateway_endpoint = CONF.AGS_GATEWAY_ENDPOINT
        if gateway_endpoint:
            adapter = AGSGatewayAdapter(
                gateway_endpoint, AGSGatewayType.YUANBAO_OPEN_API
            )
            session.mount("http://", adapter=adapter)
            session.mount("https://", adapter=adapter)
        elif CONF.YUANBAO_APP_KEY and CONF.YUANBAO_SECRET:
            adapter = YuanbaoSignAdapter(
                CONF.YUANBAO_APP_KEY, CONF.YUANBAO_SECRET
            )
            session.mount("http://", adapter=adapter)
            session.mount("https://", adapter=adapter)
        return session

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        if self.default_headers:
            headers.update(self.default_headers)
        return headers

    # 公网: {bucket}.cos.{region}.myqcloud.com
    # 内网: {bucket}.cos-internal.{region}.tencentcos.cn
    _COS_PUBLIC_HOST_RE = re.compile(
        r"^(.+)\.cos\.([a-z0-9-]+)\.myqcloud\.com$",
        re.IGNORECASE,
    )

    @staticmethod
    def _replace_cos_url_with_proxy(cos_url: str):
        """
        将 COS URL 的 host 替换为代理地址，并返回原始 host

        Args:
            cos_url: 原始 COS URL，例如：
                     https://tencent-docs-1251316161.cos.ap-guangzhou.myqcloud.com/tmp/xxx?params

        Returns:
            tuple: (代理 URL, 原始 host)
                   例如: ("http://auth.proxy/yuanbao/cos/tmp/xxx?params",
                          "https://tencent-docs-1251316161.cos.ap-guangzhou.myqcloud.com")
        """
        proxy_base = CONF.TENCENT_DOCS_COS_PROXY_URL

        # 解析原始 URL
        parsed = urlparse(cos_url)

        # 构建原始 host（包含 scheme）
        original_host = f"{parsed.scheme}://{parsed.netloc}"

        # 构建新的代理 URL：代理地址 + 路径 + 查询参数
        proxy_url = f"{proxy_base}{parsed.path}"
        if parsed.query:
            proxy_url = f"{proxy_url}?{parsed.query}"

        logger.debug(f"COS URL replaced: {original_host} -> {proxy_base}")

        return proxy_url, original_host

    @classmethod
    def _prepare_direct_cos_request(cls, cos_url: str) -> Tuple[str, Optional[str]]:
        """
        将 COS 公网预签名 URL 转为内网建连地址，并返回签名用的原 Host。

        Returns:
            (request_url, host_header)：
            - 公网 COS URL：request_url 为内网域名，host_header 为原公网 netloc
            - 已是内网 COS：原样返回，host_header 为当前 netloc
            - 非 COS URL：原样返回，host_header 为 None
        """
        parsed = urlparse(cos_url)
        match = cls._COS_PUBLIC_HOST_RE.match(parsed.netloc)
        if match:
            bucket, region = match.group(1), match.group(2)
            internal_netloc = f"{bucket}.cos-internal.{region}.tencentcos.cn"
            request_url = urlunparse(
                (
                    parsed.scheme,
                    internal_netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
            logger.debug(
                f"COS URL rewritten for internal access: {parsed.netloc} -> {internal_netloc}"
            )
            return request_url, parsed.netloc

        if ".cos-internal." in parsed.netloc.lower():
            return cos_url, parsed.netloc

        return cos_url, None

    @classmethod
    def _direct_cos_request_args(cls, cos_url: str) -> Tuple[str, Dict[str, str]]:
        """
        直连 COS 时按配置决定是否改走内网；开启时附带原 Host 以保留预签名。
        """
        if not CONF.TENCENT_DOCS_COS_USE_INTERNAL:
            return cos_url, {}

        request_url, host = cls._prepare_direct_cos_request(cos_url)
        headers: Dict[str, str] = {}
        if host:
            headers["Host"] = host
        return request_url, headers

    @classmethod
    def _parse_response(cls, response, action: str) -> Dict[str, Any]:
        try:
            logger.debug(
                f"TencentDocs {action} parse_response -> "
                f"status: {response.status_code}, "
                f"headers: {dict(response.headers)}, "
                f"body: {response.text}"
            )
            response.raise_for_status()
        except (
            requests.RequestException
        ) as exc:  # pragma: no cover - requests raises HTTPError
            raise CallTencentDocsException.NetworkError(message=str(exc))

        try:
            payload = response.json()
        except ValueError:
            raise CallTencentDocsException.RequestFailed(
                msg=f"{action} failed, response is not json: {response.text}"
            )

        ret = payload.get("ret")
        msg = payload.get("msg", "")
        if ret != 0:
            raise CallTencentDocsException.RequestFailed(ret=ret, msg=msg)

        return payload.get("data", {})

    # 文件预上传
    def local_convert_upload(self, *, file_name: str, file_size: int) -> dict:
        url = f"{self.base_url}/localConvert-upload"
        headers = self._build_headers()
        data = {"fileName": file_name, "fileSize": file_size}

        logger.info(
            f"TencentDocs local_convert_upload -> url: {url}, data: {data}, "
            f"headers: {list(headers.keys())}"
        )
        try:
            response = self._session.post(
                url, headers=headers, data=data, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise CallTencentDocsException.NetworkError(message=str(exc))

        result = self._parse_response(response, "localConvert-upload")
        logger.info(f"TencentDocs local_convert_upload <- resp: {result}")
        return result

    @classmethod
    def upload_file_to_cos(
        cls,
        *,
        cos_put_url: str,
        custom_header: Optional[Dict[str, str]] = None,
        file_data: Optional[bytes] = None,
        file_path: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        if custom_header is None:
            custom_header = {}
        if timeout is None:
            timeout = CONF.TENCENT_DOCS_TIMEOUT

        if file_data is None and file_path:
            with open(file_path, "rb") as f:
                file_data = f.read()

        if file_data is None:
            raise CallTencentDocsException.UploadToCOSFailed(
                message="file_data or file_path must be provided"
            )

        # 直接使用腾讯文档返回的预签名 URL 上传
        # 预签名 URL 已包含签名信息，无需额外凭证
        if cls._use_direct_cos_static():
            # 网关 / 直连加签：预签名 URL 直连（不经 COS 代理）
            request_url, internal_headers = cls._direct_cos_request_args(cos_put_url)
            headers = custom_header.copy()
            headers.update(internal_headers)
            logger.info(
                f"TencentDocs upload_file_to_cos (direct presigned URL) -> url: {request_url}, "
                f"headers: {list(headers.keys())}"
            )
            try:
                response = requests.put(
                    request_url, data=file_data, headers=headers, timeout=timeout
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise CallTencentDocsException.UploadToCOSFailed(message=str(exc))
        else:
            # 代理模式：替换 COS URL 为代理地址，并获取原始 host
            proxy_url, original_host = cls._replace_cos_url_with_proxy(cos_put_url)

            headers = custom_header.copy()
            headers[CONF.TENCENT_DOCS_COS_PROXY_HEADER] = original_host

            logger.info(
                f"TencentDocs upload_file_to_cos -> url: {proxy_url}, "
                f"original_host: {original_host}, "
                f"headers: {dict(headers)}"
            )
            try:
                response = requests.put(
                    proxy_url, data=file_data, headers=headers, timeout=timeout
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise CallTencentDocsException.UploadToCOSFailed(message=str(exc))

        logger.info("TencentDocs upload_file_to_cos <- success")
        return {"uploaded": True}

    # 提交任务
    def local_convert_submit_task(
        self,
        *,
        file_name: str,
        cos_file_key: str,
        convert_src_type: str,
        convert_dst_type: str,
    ) -> dict:
        url = f"{self.base_url}/localConvert"
        headers = self._build_headers()
        data = {
            "fileName": file_name,
            "COSFileKey": cos_file_key,
            "convertSrcType": convert_src_type,
            "convertDestType": convert_dst_type,
        }

        logger.info(
            f"TencentDocs local_convert_submit_task -> url: {url}, data: {data}"
        )
        try:
            response = self._session.post(
                url, headers=headers, data=data, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise CallTencentDocsException.NetworkError(message=str(exc))

        result = self._parse_response(response, "localConvert")
        logger.info(f"TencentDocs local_convert_submit_task <- resp: {result}")
        return result

    # 查询任务进度
    def local_convert_upload_progress(self, *, progress_query_id: str) -> dict:
        url = f"{self.base_url}/localConvert-progress"
        headers = self._build_headers()
        params = {"progressQueryID": progress_query_id}

        logger.info(
            f"TencentDocs local_convert_upload_progress -> url: {url}, params: {params}"
        )
        try:
            response = self._session.get(
                url, headers=headers, params=params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise CallTencentDocsException.NetworkError(message=str(exc))

        result = self._parse_response(response, "localConvert-progress")
        logger.info(f"TencentDocs local_convert_upload_progress <- resp: {result}")
        return result

    @classmethod
    def _download_converted_file(cls, url: str, timeout: int) -> bytes:
        """
        下载转换后的文件

        Args:
            url: 下载 URL（腾讯文档返回的预签名 COS URL）
            timeout: 超时时间

        Returns:
            bytes: 文件内容
        """
        # 直接使用腾讯文档返回的预签名 URL 下载
        # 预签名 URL 已包含签名信息，无需额外凭证
        if cls._use_direct_cos_static():
            # 网关 / 直连加签：预签名 URL 直连（不经 COS 代理）
            request_url, headers = cls._direct_cos_request_args(url)
            logger.info(
                f"TencentDocs download converted file (direct presigned URL) -> url: {request_url}"
            )
            try:
                response = requests.get(
                    request_url, headers=headers or None, timeout=timeout
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise CallTencentDocsException.DownloadFailed(message=str(exc))
            return response.content
        else:
            # 代理模式：替换 COS URL 为代理地址
            proxy_url, original_host = cls._replace_cos_url_with_proxy(url)

            headers = {CONF.TENCENT_DOCS_COS_PROXY_HEADER: original_host}

            logger.info(
                f"TencentDocs download converted file -> url: {proxy_url}, "
                f"original_host: {original_host}, "
                f"headers: {dict(headers)}"
            )
            try:
                response = requests.get(proxy_url, headers=headers, timeout=timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise CallTencentDocsException.DownloadFailed(message=str(exc))
            return response.content

    # 阻塞转换: 将上面的接口串联起来，实现阻塞转换
    # 输入参数: inputFileName, inputFileData, convertSrcType, convertDstType
    # 输出参数: outputFileName, outputFileData
    def blocking_convert(
        self,
        *,
        inputFileName: str,
        convertSrcType: str,
        convertDstType: str,
        inputFileData: Optional[bytes] = None,
        inputFilePath: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_file_name = inputFileName
        input_file_data = inputFileData
        input_file_path = inputFilePath
        src_type = convertSrcType
        dst_type = convertDstType

        if input_file_data is None and input_file_path:
            with open(input_file_path, "rb") as f:
                input_file_data = f.read()

        if input_file_data is None:
            raise CallTencentDocsException(
                message="inputFileData or inputFilePath must be provided"
            )

        file_size = len(input_file_data)

        cos_info = self.local_convert_upload(
            file_name=input_file_name,
            file_size=file_size,
        )

        cos_put_url = cos_info.get("COSPutURL") or cos_info.get("cosPutURL")
        cos_file_key = cos_info.get("COSFileKey") or cos_info.get("cosFileKey")
        custom_header = cos_info.get("CustomHeader") or cos_info.get("customHeader", {})

        if not cos_put_url or not cos_file_key:
            raise CallTencentDocsException.RequestFailed(
                msg="Missing COS upload info from localConvert-upload response"
            )

        self.upload_file_to_cos(
            cos_put_url=cos_put_url,
            custom_header=custom_header,
            file_data=input_file_data,
        )

        submit_resp = self.local_convert_submit_task(
            file_name=input_file_name,
            cos_file_key=cos_file_key,
            convert_src_type=src_type,
            convert_dst_type=dst_type,
        )
        progress_query_id = submit_resp.get("progressQueryID") or submit_resp.get(
            "progress_query_id"
        )
        if not progress_query_id:
            raise CallTencentDocsException.RequestFailed(
                msg="progressQueryID missing in localConvert response"
            )

        start_ts = time.time()
        while True:
            progress_resp = self.local_convert_upload_progress(
                progress_query_id=progress_query_id,
            )

            progress = progress_resp.get("progress", 0)
            if progress >= 100:
                file_url = progress_resp.get("url")
                if not file_url:
                    raise CallTencentDocsException.DownloadFailed(
                        message="progress response missing download url"
                    )
                file_bytes = self._download_converted_file(file_url, self.timeout)

                output_name = progress_resp.get("title") or input_file_name
                file_type = progress_resp.get("type") or dst_type
                if file_type and not output_name.lower().endswith(f".{file_type}"):
                    output_name = f"{output_name}.{file_type}"

                return {
                    "outputFileName": output_name,
                    "outputFileData": file_bytes,
                    "progressDetail": progress_resp,
                }

            if time.time() - start_ts > self.total_timeout:
                raise CallTencentDocsException.ConvertTimeout(
                    msg=f"blocking convert timeout after {self.total_timeout}s"
                )

            time.sleep(self.poll_interval)
