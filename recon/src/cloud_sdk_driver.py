#!/usr/bin/env python
# -*- coding:utf-8 -*-

import requests

from tencentcloud.common import credential
from tencentcloud.common.common_client import CommonClient
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

from yuanbao_convert.adapter import AGSGatewayAdapter, AGSGatewayType
from yuanbao_convert.etc import conf as CONF
from yuanbao_convert.utils.log import logger
from yuanbao_convert.utils.string_util import transform_string_format_in_recursion


class CloudSdkDriver(object):
    _service_version = {
        "ocr": "2018-11-19",
    }

    @classmethod
    def _get_credential(cls, secret_id: str, secret_key: str):
        return credential.Credential(secret_id, secret_key)

    @classmethod
    def _pack(cls, **kwargs):
        return transform_string_format_in_recursion(
            kwargs, CONF.STRING_FORMAT_PASCALCASE
        )

    @classmethod
    def _unpack(cls, result, ignore_keys=None):
        return transform_string_format_in_recursion(
            result["Response"], CONF.STRING_FORMAT_CAMELCASE, ignore_keys
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
        adapter = AGSGatewayAdapter(gateway_endpoint, AGSGatewayType.TENCENTCLOUD)
        session.mount("http://", adapter=adapter)
        session.mount("https://", adapter=adapter)
        return session

    @classmethod
    def request(cls, component: str, interface: str, **kwargs) -> dict:
        request_headers = kwargs.pop("headers", None)
        if not request_headers:
            request_headers = {}

        region = kwargs.pop("region", None)
        if not region:
            region = CONF.DEFAULT_REGION

        request_data = cls._pack(**kwargs)
        cred = cls._get_credential(CONF.TENCENT_SECRET_ID, CONF.TENCENT_SECRET_KEY)

        version = cls._service_version.get(component, "2017-03-12")

        # If AGS Gateway is configured, use gateway mode
        if CONF.AGS_GATEWAY_ENDPOINT:
            common_client = CommonClient(component, version, cred, region)
            common_client.request.conn._session = cls._get_session_with_gateway()
        elif CONF.AUTH_PROXY:
            # Legacy AUTH_PROXY mode
            http_profile = HttpProfile()
            http_profile.scheme = "http"
            http_profile.endpoint = CONF.AUTH_PROXY

            client_profile = ClientProfile()
            client_profile.httpProfile = http_profile
            common_client = CommonClient(
                component, version, cred, region, profile=client_profile
            )
            common_client._requestPath = "/tencent_cloud/%s" % component.lower()
        else:
            # Direct mode without proxy
            common_client = CommonClient(component, version, cred, region)

        logger.info(
            f"Call interface [{interface}] of [{component}], request: {request_data}"
        )
        result = common_client.call_json(
            interface, request_data, headers=request_headers
        )
        logger.info(
            f"Call interface [{interface}] of [{component}], response: {result}"
        )
        return cls._unpack(result)


if __name__ == "__main__":
    print(
        CloudSdkDriver.request(component="ocr", interface="RecognizeTableAccurateOCR")
    )
