"""Browser executor request/response models.

Pure Pydantic — does NOT import Playwright.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus.runtime.executors.models.base import ExecutorRequest


# ============================================================================
# Element Locator Types
# ============================================================================


class BidLocator(BaseModel):
    """Locate element by BrowserGym ID (bid)."""

    type: Literal['bid'] = 'bid'
    bid: str = Field(..., description='Browser element ID (browsergym id)')


class XYLocator(BaseModel):
    """Locate by pixel coordinates."""

    type: Literal['xy'] = 'xy'
    x: float = Field(..., description='X coordinate in pixels')
    y: float = Field(..., description='Y coordinate in pixels')


class SelectorLocator(BaseModel):
    """Locate element by CSS selector."""

    type: Literal['selector'] = 'selector'
    selector: str = Field(..., description='CSS selector string')
    index: int = Field(default=0, description='Element index if selector matches multiple elements')


# Discriminated Union for element locators
ElementLocator = Annotated[BidLocator | XYLocator | SelectorLocator, Field(discriminator='type')]


# ============================================================================
# Request/Response Models
# ============================================================================


class CreateBrowserSessionResponse(BaseModel):
    session_id: str = Field(..., description='Unique session identifier')


class CreateBrowserSessionRequest(ExecutorRequest[CreateBrowserSessionResponse]):
    executor_name: ClassVar[str] = 'create_browser_session'
    response_type: ClassVar[type[BaseModel]] = CreateBrowserSessionResponse

    headless: bool = Field(default=True, description='Whether to run browser in headless mode')
    viewport_width: int = Field(default=1280, description='Browser viewport width')
    viewport_height: int = Field(default=720, description='Browser viewport height')
    timeout: int = Field(default=10000, description='Default timeout in milliseconds')
    executable_path: str | None = Field(
        default='/nix/runtime/nix-env/bin/chromium', description='Path to browser executable'
    )
    user_agent: str | None = Field(default=None, description='Custom User-Agent string for the browser')
    demo_mode: str = Field(
        default='off',
        description="Demo mode: 'off', 'default', 'all_blue', or 'only_visible_elements'",
    )
    record_har: bool = Field(default=False, description='Whether to record HAR (HTTP Archive) file')
    har_path: str | None = Field(default=None, description='Path to save HAR file (required if record_har is True)')
    har_content: str = Field(
        default='embed',
        description="HAR content option: 'omit', 'embed', 'attach'",
    )
    har_url_filter: str | None = Field(default=None, description='Regex pattern to filter URLs in HAR')


class DestroyBrowserSessionResponse(BaseModel):
    success: bool = Field(..., description='Whether destruction was successful')
    har_path: str | None = Field(default=None, description='Path where HAR file was saved')


class DestroyBrowserSessionRequest(ExecutorRequest[DestroyBrowserSessionResponse]):
    executor_name: ClassVar[str] = 'destroy_browser_session'
    response_type: ClassVar[type[BaseModel]] = DestroyBrowserSessionResponse

    session_id: str = Field(..., description='Session ID to destroy')


class GotoResponse(BaseModel):
    success: bool = Field(..., description='Whether navigation succeeded')
    url: str = Field(..., description='Final URL after navigation')


class GotoRequest(ExecutorRequest[GotoResponse]):
    executor_name: ClassVar[str] = 'browser_goto'
    response_type: ClassVar[type[BaseModel]] = GotoResponse

    session_id: str = Field(..., description='Browser session ID')
    url: str = Field(..., description='URL to navigate to')
    timeout: int | None = Field(default=None, description='Navigation timeout in ms')


class ClickResponse(BaseModel):
    success: bool = Field(..., description='Whether click succeeded')


class ClickRequest(ExecutorRequest[ClickResponse]):
    executor_name: ClassVar[str] = 'browser_click'
    response_type: ClassVar[type[BaseModel]] = ClickResponse

    session_id: str = Field(..., description='Browser session ID')
    locator: ElementLocator | None = Field(default=None, description='Element locator (bid, xy, or selector)')
    bid: str | None = Field(default=None, description='[Deprecated] Browser element ID, use locator instead')
    timeout: int | None = Field(default=None, description='Click timeout in ms')
    button: Literal['left', 'right', 'middle'] = Field(default='left', description='Mouse button to click')

    @model_validator(mode='after')
    def validate_locator(self) -> ClickRequest:
        if self.locator is None and self.bid is None:
            raise ValueError('Either locator or bid must be provided')
        if self.locator is None and self.bid is not None:
            object.__setattr__(self, 'locator', BidLocator(bid=self.bid))
        return self


class FillResponse(BaseModel):
    success: bool = Field(..., description='Whether fill succeeded')


class FillRequest(ExecutorRequest[FillResponse]):
    executor_name: ClassVar[str] = 'browser_fill'
    response_type: ClassVar[type[BaseModel]] = FillResponse

    session_id: str = Field(..., description='Browser session ID')
    locator: BidLocator | SelectorLocator | None = Field(
        default=None, description='Element locator (bid or selector, xy not supported)'
    )
    bid: str | None = Field(default=None, description='[Deprecated] Browser element ID, use locator instead')
    value: str = Field(..., description='Value to fill')
    timeout: int | None = Field(default=None, description='Fill timeout in ms')

    @model_validator(mode='after')
    def validate_locator(self) -> FillRequest:
        if self.locator is None and self.bid is None:
            raise ValueError('Either locator or bid must be provided')
        if self.locator is None and self.bid is not None:
            object.__setattr__(self, 'locator', BidLocator(bid=self.bid))
        return self


class ScrollResponse(BaseModel):
    success: bool = Field(..., description='Whether scroll succeeded')


class ScrollRequest(ExecutorRequest[ScrollResponse]):
    executor_name: ClassVar[str] = 'browser_scroll'
    response_type: ClassVar[type[BaseModel]] = ScrollResponse

    session_id: str = Field(..., description='Browser session ID')
    direction: str = Field(..., description="Scroll direction: 'up' or 'down'")
    amount: int = Field(default=500, description='Scroll amount in pixels')


class PressKeyResponse(BaseModel):
    success: bool = Field(..., description='Whether key press succeeded')


class PressKeyRequest(ExecutorRequest[PressKeyResponse]):
    executor_name: ClassVar[str] = 'browser_press_key'
    response_type: ClassVar[type[BaseModel]] = PressKeyResponse

    session_id: str = Field(..., description='Browser session ID')
    key: str = Field(..., description="Key to press (e.g., 'Enter', 'Escape')")
    locator: BidLocator | SelectorLocator | None = Field(
        default=None, description='Optional element locator to focus first (bid or selector)'
    )
    bid: str | None = Field(default=None, description='[Deprecated] Optional element to focus first, use locator')

    @model_validator(mode='after')
    def validate_locator(self) -> PressKeyRequest:
        if self.locator is None and self.bid is not None:
            object.__setattr__(self, 'locator', BidLocator(bid=self.bid))
        return self


class HoverResponse(BaseModel):
    success: bool = Field(..., description='Whether hover succeeded')


class HoverRequest(ExecutorRequest[HoverResponse]):
    executor_name: ClassVar[str] = 'browser_hover'
    response_type: ClassVar[type[BaseModel]] = HoverResponse

    session_id: str = Field(..., description='Browser session ID')
    locator: ElementLocator | None = Field(default=None, description='Element locator (bid, xy, or selector)')
    bid: str | None = Field(default=None, description='[Deprecated] Browser element ID, use locator instead')
    timeout: int | None = Field(default=None, description='Hover timeout in ms')

    @model_validator(mode='after')
    def validate_locator(self) -> HoverRequest:
        if self.locator is None and self.bid is None:
            raise ValueError('Either locator or bid must be provided')
        if self.locator is None and self.bid is not None:
            object.__setattr__(self, 'locator', BidLocator(bid=self.bid))
        return self


class SelectOptionResponse(BaseModel):
    success: bool = Field(..., description='Whether selection succeeded')


class SelectOptionRequest(ExecutorRequest[SelectOptionResponse]):
    executor_name: ClassVar[str] = 'browser_select_option'
    response_type: ClassVar[type[BaseModel]] = SelectOptionResponse

    session_id: str = Field(..., description='Browser session ID')
    locator: BidLocator | SelectorLocator | None = Field(
        default=None, description='Element locator (bid or selector, xy not supported)'
    )
    bid: str | None = Field(default=None, description='[Deprecated] Browser element ID, use locator instead')
    value: str = Field(..., description='Option value to select')
    timeout: int | None = Field(default=None, description='Select timeout in ms')

    @model_validator(mode='after')
    def validate_locator(self) -> SelectOptionRequest:
        if self.locator is None and self.bid is None:
            raise ValueError('Either locator or bid must be provided')
        if self.locator is None and self.bid is not None:
            object.__setattr__(self, 'locator', BidLocator(bid=self.bid))
        return self


class KeyboardTypeResponse(BaseModel):
    success: bool = Field(..., description='Whether typing succeeded')


class KeyboardTypeRequest(ExecutorRequest[KeyboardTypeResponse]):
    executor_name: ClassVar[str] = 'browser_keyboard_type'
    response_type: ClassVar[type[BaseModel]] = KeyboardTypeResponse

    session_id: str = Field(..., description='Browser session ID')
    text: str = Field(..., description='Text to type')
    delay: int = Field(default=0, description='Delay between key presses in milliseconds')


class NewTabResponse(BaseModel):
    success: bool = Field(..., description='Whether new tab was created')
    tab_index: int = Field(..., description='Index of the new tab')


class NewTabRequest(ExecutorRequest[NewTabResponse]):
    executor_name: ClassVar[str] = 'browser_new_tab'
    response_type: ClassVar[type[BaseModel]] = NewTabResponse

    session_id: str = Field(..., description='Browser session ID')
    url: str | None = Field(default=None, description='Optional URL to navigate to')


class CloseTabResponse(BaseModel):
    success: bool = Field(..., description='Whether tab was closed')


class CloseTabRequest(ExecutorRequest[CloseTabResponse]):
    executor_name: ClassVar[str] = 'browser_close_tab'
    response_type: ClassVar[type[BaseModel]] = CloseTabResponse

    session_id: str = Field(..., description='Browser session ID')
    tab_index: int | None = Field(default=None, description='Tab index to close (default: current)')


class FocusTabResponse(BaseModel):
    success: bool = Field(..., description='Whether tab was focused')


class FocusTabRequest(ExecutorRequest[FocusTabResponse]):
    executor_name: ClassVar[str] = 'browser_focus_tab'
    response_type: ClassVar[type[BaseModel]] = FocusTabResponse

    session_id: str = Field(..., description='Browser session ID')
    tab_index: int = Field(..., description='Tab index to focus')


class TabInfo(BaseModel):
    index: int = Field(..., description='Tab index (0-based)')
    url: str = Field(..., description='Current URL of the tab')
    title: str = Field(..., description='Page title of the tab')
    is_active: bool = Field(..., description='Whether this is the currently active tab')


class ListTabsResponse(BaseModel):
    success: bool = Field(..., description='Whether listing succeeded')
    tabs: list[TabInfo] = Field(default_factory=list, description='List of tab information')
    active_tab_index: int = Field(..., description='Index of the currently active tab')


class ListTabsRequest(ExecutorRequest[ListTabsResponse]):
    executor_name: ClassVar[str] = 'browser_list_tabs'
    response_type: ClassVar[type[BaseModel]] = ListTabsResponse

    session_id: str = Field(..., description='Browser session ID')


class GoBackResponse(BaseModel):
    success: bool = Field(..., description='Whether navigation succeeded')
    url: str = Field(..., description='URL after going back')


class GoBackRequest(ExecutorRequest[GoBackResponse]):
    executor_name: ClassVar[str] = 'browser_go_back'
    response_type: ClassVar[type[BaseModel]] = GoBackResponse

    session_id: str = Field(..., description='Browser session ID')


class GoForwardResponse(BaseModel):
    success: bool = Field(..., description='Whether navigation succeeded')
    url: str = Field(..., description='URL after going forward')


class GoForwardRequest(ExecutorRequest[GoForwardResponse]):
    executor_name: ClassVar[str] = 'browser_go_forward'
    response_type: ClassVar[type[BaseModel]] = GoForwardResponse

    session_id: str = Field(..., description='Browser session ID')


class GetObservationResponse(BaseModel):
    url: str = Field(..., description='Current page URL')
    title: str = Field(..., description='Page title')
    axtree: str = Field(..., description='Accessibility tree representation')
    focused_element_bid: str = Field(default='', description='BID of the currently focused element')
    screenshot: str | None = Field(default=None, description='Base64 encoded screenshot')


class GetObservationRequest(ExecutorRequest[GetObservationResponse]):
    executor_name: ClassVar[str] = 'browser_get_observation'
    response_type: ClassVar[type[BaseModel]] = GetObservationResponse

    session_id: str = Field(..., description='Browser session ID')
    include_screenshot: bool = Field(default=False, description='Whether to include screenshot')
    screenshot_scale_factor: float = Field(default=1.0, description='Scale factor for higher resolution screenshots')
    viewport_only: bool = Field(
        default=False,
        description='If True, only include elements visible in the current viewport in the accessibility tree',
    )


class ScreenshotClip(BaseModel):
    """Clip rectangle for browser screenshot."""

    x: float = Field(..., description='X coordinate of the top-left corner')
    y: float = Field(..., description='Y coordinate of the top-left corner')
    width: float = Field(..., description='Clip width in pixels')
    height: float = Field(..., description='Clip height in pixels')


class ScreenshotResponse(BaseModel):
    success: bool = Field(..., description='Whether screenshot capture succeeded')
    screenshot: str | None = Field(default=None, description='Base64 encoded screenshot')
    media_type: str = Field(default='image/png', description='Screenshot media type')
    url: str = Field(default='', description='Current page URL')
    title: str = Field(default='', description='Current page title')
    path: str | None = Field(default=None, description='Path where screenshot was saved, if requested')
    error: str | None = Field(default=None, description='Error message if screenshot failed')


class ScreenshotRequest(ExecutorRequest[ScreenshotResponse]):
    """Capture a screenshot via Playwright page.screenshot().

    Known fields are validated and documented. ``options`` plus extra top-level
    fields are forwarded to Playwright for less common screenshot kwargs.
    """

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    executor_name: ClassVar[str] = 'browser_screenshot'
    response_type: ClassVar[type[BaseModel]] = ScreenshotResponse

    session_id: str = Field(..., description='Browser session ID')
    options: dict[str, Any] = Field(
        default_factory=dict,
        description='Additional Playwright page.screenshot kwargs. Explicit fields take precedence.',
    )
    image_type: Literal['png', 'jpeg'] | None = Field(default=None, alias='type', description='Screenshot image type')
    quality: int | None = Field(default=None, ge=0, le=100, description='JPEG quality from 0 to 100')
    full_page: bool | None = Field(default=None, description='Whether to take a full-page screenshot')
    clip: ScreenshotClip | None = Field(default=None, description='Optional clip rectangle')
    omit_background: bool | None = Field(default=None, description='Whether to hide the default white background')
    timeout: int | None = Field(default=None, description='Screenshot timeout in ms')
    animations: Literal['allow', 'disabled'] | None = Field(default=None, description='Animation handling')
    caret: Literal['hide', 'initial'] | None = Field(default=None, description='Caret handling')
    scale: Literal['css', 'device'] | None = Field(default=None, description='Screenshot scaling mode')
    mask_color: str | None = Field(default=None, description='Overlay color for masked elements')
    style: str | None = Field(default=None, description='Stylesheet text to apply while screenshotting')
    path: str | None = Field(default=None, description='Path to save the screenshot in the runtime filesystem')

    def playwright_options(self) -> dict[str, Any]:
        """Return kwargs for Playwright page.screenshot()."""
        explicit_options = self.model_dump(exclude={'session_id', 'options'}, exclude_none=True, by_alias=True)
        merged_options = dict(self.options)
        merged_options.update(explicit_options)
        return merged_options


# ============================================================================
# High-level Browser Primitives
# ============================================================================


class GetUrlResponse(BaseModel):
    url: str = Field(..., description='Current page URL')


class GetUrlRequest(ExecutorRequest[GetUrlResponse]):
    executor_name: ClassVar[str] = 'browser_get_url'
    response_type: ClassVar[type[BaseModel]] = GetUrlResponse

    session_id: str = Field(..., description='Browser session ID')


class EvalJsResponse(BaseModel):
    value: Any = Field(default=None, description='JavaScript evaluation result (JSON-serializable)')


class EvalJsRequest(ExecutorRequest[EvalJsResponse]):
    executor_name: ClassVar[str] = 'browser_eval_js'
    response_type: ClassVar[type[BaseModel]] = EvalJsResponse

    session_id: str = Field(..., description='Browser session ID')
    expression: str = Field(..., description='JavaScript expression to evaluate via page.evaluate()')


class WaitForSelectorResponse(BaseModel):
    found: bool = Field(..., description='Whether the selector was found within the timeout')


class WaitForSelectorRequest(ExecutorRequest[WaitForSelectorResponse]):
    executor_name: ClassVar[str] = 'browser_wait_for_selector'
    response_type: ClassVar[type[BaseModel]] = WaitForSelectorResponse

    session_id: str = Field(..., description='Browser session ID')
    selector: str = Field(..., description='CSS selector to wait for')
    timeout: int = Field(default=30000, description='Timeout in milliseconds')
    state: Literal['attached', 'detached', 'visible', 'hidden'] = Field(
        default='visible', description='Target state for the selector'
    )


class SetGeolocationResponse(BaseModel):
    success: bool = Field(..., description='Whether geolocation was set successfully')


class SetGeolocationRequest(ExecutorRequest[SetGeolocationResponse]):
    executor_name: ClassVar[str] = 'browser_set_geolocation'
    response_type: ClassVar[type[BaseModel]] = SetGeolocationResponse

    session_id: str = Field(..., description='Browser session ID')
    latitude: float = Field(..., description='Latitude')
    longitude: float = Field(..., description='Longitude')
    accuracy: float = Field(default=0, description='Accuracy in meters')


class WaitForLoadStateResponse(BaseModel):
    success: bool = Field(..., description='Whether the load state was reached')


class WaitForLoadStateRequest(ExecutorRequest[WaitForLoadStateResponse]):
    executor_name: ClassVar[str] = 'browser_wait_for_load_state'
    response_type: ClassVar[type[BaseModel]] = WaitForLoadStateResponse

    session_id: str = Field(..., description='Browser session ID')
    state: Literal['load', 'domcontentloaded', 'networkidle'] = Field(
        default='load', description='Load state to wait for'
    )
    timeout: int = Field(default=30000, description='Timeout in milliseconds')


class MouseDragResponse(BaseModel):
    success: bool = Field(..., description='Whether drag succeeded')


class MouseDragRequest(ExecutorRequest[MouseDragResponse]):
    executor_name: ClassVar[str] = 'browser_mouse_drag'
    response_type: ClassVar[type[BaseModel]] = MouseDragResponse

    session_id: str = Field(..., description='Browser session ID')
    start_x: float = Field(..., description='Start X coordinate in pixels')
    start_y: float = Field(..., description='Start Y coordinate in pixels')
    end_x: float = Field(..., description='End X coordinate in pixels')
    end_y: float = Field(..., description='End Y coordinate in pixels')
    steps: int = Field(default=10, description='Number of intermediate mouse move steps')


class GetContentResponse(BaseModel):
    content: str = Field(..., description='Page HTML content')
    url: str = Field(..., description='Current page URL')


class GetContentRequest(ExecutorRequest[GetContentResponse]):
    executor_name: ClassVar[str] = 'browser_get_content'
    response_type: ClassVar[type[BaseModel]] = GetContentResponse

    session_id: str = Field(..., description='Browser session ID')
    url: str | None = Field(default=None, description='Optional URL to navigate to first')
