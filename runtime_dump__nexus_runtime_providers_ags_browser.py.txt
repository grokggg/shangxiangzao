"""AGS browser runtime implementations."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

from dataclasses import dataclass
from typing import Any

from nexus.runtime.providers.ags.base import AGSRawRuntime, _BaseAGSRuntime
from nexus.runtime.providers.ags.handle import AGSInstanceHandle


logger = logging.getLogger(__name__)


@dataclass
class _BrowserSessionState:
    session: Any
    owns_context: bool


class AGSBrowserRuntime(_BaseAGSRuntime):
    """Limited runtime backed by AGS native browser sandbox."""

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        cdp_url: str,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(handle, request_timeout=request_timeout, resource_spec=resource_spec)
        self.cdp_url = cdp_url
        self._playwright: Any = None
        self._browser: Any = None
        self._browser_lock = asyncio.Lock()
        self._sessions: dict[str, _BrowserSessionState] = {}

    @property
    def base_url(self) -> str | None:
        return self.cdp_url

    async def _ensure_browser(self) -> Any:
        async with self._browser_lock:
            if self._browser is not None:
                return self._browser

            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise ImportError(
                    'playwright is required for AGS browser runtime. Install nexus with the extensions extra.'
                ) from exc

            self._playwright = await async_playwright().start()
            self._playwright.selectors.set_test_id_attribute('bid')
            self._browser = await self._playwright.chromium.connect_over_cdp(
                self.cdp_url,
                headers=self.handle.headers,
                timeout=int(self.request_timeout * 1000),
            )
            return self._browser

    async def _create_context(self, request: Any) -> tuple[Any, bool]:
        browser = await self._ensure_browser()
        context_kwargs = {
            'viewport': {'width': request.viewport_width, 'height': request.viewport_height},
            'ignore_https_errors': True,
            'user_agent': request.user_agent,
        }

        try:
            context = await browser.new_context(**context_kwargs)
            context.set_default_timeout(request.timeout)
            return context, True
        except Exception:
            if browser.contexts:
                context = browser.contexts[0]
                with contextlib.suppress(Exception):
                    context.set_default_timeout(request.timeout)
                return context, False
            raise

    async def _destroy_session(self, session_id: str) -> bool:
        state = self._sessions.pop(session_id, None)
        if state is None:
            return False

        try:
            if state.owns_context:
                await state.session.close()
            else:
                for page in list(state.session.pages):
                    with contextlib.suppress(Exception):
                        await page.close()
                state.session.pages.clear()
        except Exception:
            logger.exception('Failed to destroy AGS browser session %s', session_id)
            return False

        return True

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            await self._destroy_session(session_id)

        if self._browser is not None:
            with contextlib.suppress(Exception):
                await self._browser.close()
            self._browser = None

        if self._playwright is not None:
            with contextlib.suppress(Exception):
                await self._playwright.stop()
            self._playwright = None

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        from nexus.runtime.executors.browser import (
            AsyncBrowserSession,
            ClickRequest,
            ClickResponse,
            CloseTabRequest,
            CloseTabResponse,
            CreateBrowserSessionRequest,
            CreateBrowserSessionResponse,
            DestroyBrowserSessionRequest,
            DestroyBrowserSessionResponse,
            FillRequest,
            FillResponse,
            FocusTabRequest,
            FocusTabResponse,
            GetObservationRequest,
            GetObservationResponse,
            GoBackRequest,
            GoBackResponse,
            GoForwardRequest,
            GoForwardResponse,
            GotoRequest,
            GotoResponse,
            HoverRequest,
            HoverResponse,
            KeyboardTypeRequest,
            KeyboardTypeResponse,
            ListTabsRequest,
            ListTabsResponse,
            NewTabRequest,
            NewTabResponse,
            PressKeyRequest,
            PressKeyResponse,
            ScreenshotRequest,
            ScreenshotResponse,
            ScrollRequest,
            ScrollResponse,
            SelectOptionRequest,
            SelectOptionResponse,
            TabInfo,
        )

        if executor_name == 'create_browser_session':
            request = CreateBrowserSessionRequest.model_validate(arg)
            context, owns_context = await self._create_context(request)
            session = AsyncBrowserSession(
                context=context,
                viewport_width=request.viewport_width,
                viewport_height=request.viewport_height,
                timeout=request.timeout,
                demo_mode=request.demo_mode,  # type: ignore[arg-type]
            )
            await session.initialize()
            with contextlib.suppress(Exception):
                await session.current_page.set_viewport_size({
                    'width': request.viewport_width,
                    'height': request.viewport_height,
                })
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = _BrowserSessionState(session=session, owns_context=owns_context)
            return CreateBrowserSessionResponse(session_id=session_id).model_dump()

        if executor_name == 'destroy_browser_session':
            request = DestroyBrowserSessionRequest.model_validate(arg)
            return DestroyBrowserSessionResponse(success=await self._destroy_session(request.session_id)).model_dump()

        session_id = arg.get('session_id') if isinstance(arg, dict) else None
        state = self._sessions.get(session_id or '')
        if state is None:
            raise ValueError(f'Session not found: {session_id}')
        session = state.session

        if executor_name == 'browser_goto':
            request = GotoRequest.model_validate(arg)
            try:
                final_url = await session.goto(request.url, request.timeout)
                return GotoResponse(success=True, url=final_url).model_dump()
            except Exception:
                logger.exception('AGS browser goto failed')
                return GotoResponse(success=False, url=session.current_page.url).model_dump()

        if executor_name == 'browser_go_back':
            request = GoBackRequest.model_validate(arg)
            try:
                url = await session.go_back()
                return GoBackResponse(success=True, url=url).model_dump()
            except Exception:
                logger.exception('AGS browser go_back failed')
                return GoBackResponse(success=False, url=session.current_page.url).model_dump()

        if executor_name == 'browser_go_forward':
            request = GoForwardRequest.model_validate(arg)
            try:
                url = await session.go_forward()
                return GoForwardResponse(success=True, url=url).model_dump()
            except Exception:
                logger.exception('AGS browser go_forward failed')
                return GoForwardResponse(success=False, url=session.current_page.url).model_dump()

        if executor_name == 'browser_click':
            request = ClickRequest.model_validate(arg)
            try:
                await session.click(request.locator, request.timeout, request.button)
                return ClickResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser click failed')
                return ClickResponse(success=False).model_dump()

        if executor_name == 'browser_fill':
            request = FillRequest.model_validate(arg)
            try:
                await session.fill(request.locator, request.value, request.timeout)
                return FillResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser fill failed')
                return FillResponse(success=False).model_dump()

        if executor_name == 'browser_scroll':
            request = ScrollRequest.model_validate(arg)
            try:
                await session.scroll(request.direction, request.amount)
                return ScrollResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser scroll failed')
                return ScrollResponse(success=False).model_dump()

        if executor_name == 'browser_press_key':
            request = PressKeyRequest.model_validate(arg)
            try:
                await session.press_key(request.key, request.locator)
                return PressKeyResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser press_key failed')
                return PressKeyResponse(success=False).model_dump()

        if executor_name == 'browser_hover':
            request = HoverRequest.model_validate(arg)
            try:
                await session.hover(request.locator, request.timeout)
                return HoverResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser hover failed')
                return HoverResponse(success=False).model_dump()

        if executor_name == 'browser_select_option':
            request = SelectOptionRequest.model_validate(arg)
            try:
                await session.select_option(request.locator, request.value, request.timeout)
                return SelectOptionResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser select_option failed')
                return SelectOptionResponse(success=False).model_dump()

        if executor_name == 'browser_keyboard_type':
            request = KeyboardTypeRequest.model_validate(arg)
            try:
                await session.keyboard_type(request.text, request.delay)
                return KeyboardTypeResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser keyboard_type failed')
                return KeyboardTypeResponse(success=False).model_dump()

        if executor_name == 'browser_new_tab':
            request = NewTabRequest.model_validate(arg)
            try:
                tab_index = await session.new_tab(request.url)
                return NewTabResponse(success=True, tab_index=tab_index).model_dump()
            except Exception:
                logger.exception('AGS browser new_tab failed')
                return NewTabResponse(success=False, tab_index=-1).model_dump()

        if executor_name == 'browser_close_tab':
            request = CloseTabRequest.model_validate(arg)
            try:
                await session.close_tab(request.tab_index)
                return CloseTabResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser close_tab failed')
                return CloseTabResponse(success=False).model_dump()

        if executor_name == 'browser_focus_tab':
            request = FocusTabRequest.model_validate(arg)
            try:
                await session.focus_tab(request.tab_index)
                return FocusTabResponse(success=True).model_dump()
            except Exception:
                logger.exception('AGS browser focus_tab failed')
                return FocusTabResponse(success=False).model_dump()

        if executor_name == 'browser_list_tabs':
            request = ListTabsRequest.model_validate(arg)
            _ = request
            try:
                tabs = [TabInfo(**tab) for tab in await session.list_tabs()]
                return ListTabsResponse(
                    success=True,
                    tabs=tabs,
                    active_tab_index=session.current_page_index,
                ).model_dump()
            except Exception:
                logger.exception('AGS browser list_tabs failed')
                return ListTabsResponse(success=False, tabs=[], active_tab_index=-1).model_dump()

        if executor_name == 'browser_get_observation':
            request = GetObservationRequest.model_validate(arg)
            observation = await session.get_observation(
                include_screenshot=request.include_screenshot,
                screenshot_scale_factor=request.screenshot_scale_factor,
                viewport_only=request.viewport_only,
            )
            return GetObservationResponse(**observation).model_dump()

        if executor_name == 'browser_screenshot':
            request = ScreenshotRequest.model_validate(arg)
            try:
                screenshot = await session.screenshot(request.playwright_options())
                return ScreenshotResponse(success=True, **screenshot).model_dump()
            except Exception as exc:
                logger.exception('AGS browser screenshot failed')
                return ScreenshotResponse(success=False, error=str(exc), url=session.current_page.url).model_dump()

        raise NotImplementedError(f'Unsupported AGS browser executor: {executor_name!r}')


class AGSRawBrowserRuntime(AGSRawRuntime):
    """Raw AGS browser runtime that exposes CDP and direct Playwright access."""

    def __init__(
        self,
        handle: AGSInstanceHandle,
        *,
        browser_port: int = 9000,
        request_timeout: float = 1800.0,
        resource_spec: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            handle,
            kind='browser',
            default_port=browser_port,
            request_timeout=request_timeout,
            resource_spec=resource_spec,
        )
        self.browser_port = browser_port
        self.cdp_url = self.handle.url(browser_port, '/cdp')

    async def json_version(self) -> Any:
        return await self.request_json('GET', '/cdp/json/version', port=self.browser_port)

    async def json_list(self) -> Any:
        return await self.request_json('GET', '/cdp/json/list', port=self.browser_port)

    async def connect_playwright(self) -> Any:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ImportError(
                'playwright is required for AGS raw browser runtime. Install nexus with the extensions extra.'
            ) from exc

        playwright = await async_playwright().start()
        try:
            playwright.selectors.set_test_id_attribute('bid')
            browser = await playwright.chromium.connect_over_cdp(
                self.cdp_url,
                headers=self.handle.headers,
                timeout=int(self.request_timeout * 1000),
            )
        except Exception:
            with contextlib.suppress(Exception):
                await playwright.stop()
            raise

        return browser, playwright
