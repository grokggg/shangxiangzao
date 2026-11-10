"""Browser executor for web automation using Playwright.

This module provides browser automation executors following the same pattern as jupyter.py.
Browser sessions are managed globally in BROWSER_SESSIONS dictionary.
"""

# /// script
# dependencies = []
#
# [project]
# name = "browser"
# authors = [
#     { name = "tarschen" },
# ]
#
# [optional-dependencies]
# runtime = ["playwright>=1.57.0"]
#
# [tool.nexus.executors]
# create_browser_session = "nexus.runtime.executors.browser"
# destroy_browser_session = "nexus.runtime.executors.browser"
# browser_goto = "nexus.runtime.executors.browser"
# browser_go_back = "nexus.runtime.executors.browser"
# browser_go_forward = "nexus.runtime.executors.browser"
# browser_click = "nexus.runtime.executors.browser"
# browser_fill = "nexus.runtime.executors.browser"
# browser_scroll = "nexus.runtime.executors.browser"
# browser_press_key = "nexus.runtime.executors.browser"
# browser_hover = "nexus.runtime.executors.browser"
# browser_select_option = "nexus.runtime.executors.browser"
# browser_keyboard_type = "nexus.runtime.executors.browser"
# browser_new_tab = "nexus.runtime.executors.browser"
# browser_close_tab = "nexus.runtime.executors.browser"
# browser_focus_tab = "nexus.runtime.executors.browser"
# browser_list_tabs = "nexus.runtime.executors.browser"
# browser_get_observation = "nexus.runtime.executors.browser"
# browser_screenshot = "nexus.runtime.executors.browser"
# browser_get_url = "nexus.runtime.executors.browser"
# browser_eval_js = "nexus.runtime.executors.browser"
# browser_wait_for_selector = "nexus.runtime.executors.browser"
# browser_set_geolocation = "nexus.runtime.executors.browser"
# browser_wait_for_load_state = "nexus.runtime.executors.browser"
# browser_get_content = "nexus.runtime.executors.browser"
# browser_mouse_drag = "nexus.runtime.executors.browser"
# ///

import asyncio
import base64
import contextlib
import logging
import uuid

from typing import Any, Literal

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from nexus.core.executor import executor_registry
from nexus.runtime.executors.models.browser import *  # noqa: F401,F403
from nexus.runtime.executors.models.browser import (
    BidLocator,
    ClickRequest,
    ClickResponse,
    CloseTabRequest,
    CloseTabResponse,
    CreateBrowserSessionRequest,
    CreateBrowserSessionResponse,
    DestroyBrowserSessionRequest,
    DestroyBrowserSessionResponse,
    ElementLocator,
    EvalJsRequest,
    EvalJsResponse,
    FillRequest,
    FillResponse,
    FocusTabRequest,
    FocusTabResponse,
    GetContentRequest,
    GetContentResponse,
    GetObservationRequest,
    GetObservationResponse,
    GetUrlRequest,
    GetUrlResponse,
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
    MouseDragRequest,
    MouseDragResponse,
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
    SelectorLocator,
    SetGeolocationRequest,
    SetGeolocationResponse,
    TabInfo,
    WaitForLoadStateRequest,
    WaitForLoadStateResponse,
    WaitForSelectorRequest,
    WaitForSelectorResponse,
    XYLocator,
)


logger = logging.getLogger(__name__)

# Global browser sessions storage (similar to IPY_SESSIONS in jupyter.py)
BROWSER_SESSIONS: dict[str, 'AsyncBrowserSession'] = {}

# Playwright instance (singleton)
_PLAYWRIGHT: Playwright | None = None
_BROWSER: Browser | None = None

# ============================================================================
# AsyncBrowserSession - Manages Playwright browser session
# ============================================================================

# Demo mode type
DemoMode = Literal['off', 'default', 'all_blue', 'only_visible_elements']


class AsyncBrowserSession:
    """Manages an async Playwright browser session."""

    # Constants aligned with BrowserGym
    BID_ATTR = 'bid'  # Playwright's default test ID attribute
    VIS_ATTR = 'browsergym_visibility_ratio'
    SOM_ATTR = 'browsergym_set_of_marks'

    def __init__(
        self,
        context: BrowserContext,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout: int = 10000,
        demo_mode: DemoMode = 'off',
        har_path: str | None = None,
    ):
        """Initialize browser session."""
        self.context = context
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout = timeout
        self.demo_mode = demo_mode
        self.har_path = har_path
        self.pages: list[Page] = []
        self.current_page_index = 0

    async def initialize(self):
        """Initialize the session with a first page."""
        page = await self.context.new_page()
        self.pages.append(page)

    @property
    def current_page(self) -> Page:
        """Get the currently active page."""
        if not self.pages:
            raise RuntimeError('No pages available in session')
        return self.pages[self.current_page_index]

    async def _get_elem_by_bid(self, bid: str, scroll_into_view: bool = False):
        """Get element locator by bid (aligned with BrowserGym's get_elem_by_bid).

        Parse the given bid to sequentially locate every nested frame leading to the bid,
        then locate the bid element. Bids are expected to take the form "abDb123", which means
        the element abDb123 is located inside frame abDAb, which is located inside frame abDA,
        which is located inside frame a, which is located inside the page's main frame.

        Args:
            bid: the browsergym id (playwright testid) of the page element.
            scroll_into_view: try to scroll element into view, unless it is completely visible.

        Returns:
            Playwright Locator for the element.
        """
        if not isinstance(bid, str):
            raise ValueError(f'expected a string, got {repr(bid)}')

        page = self.current_page
        current_frame = page

        # Dive into each nested frame, to the frame where the element is located
        i = 0
        while bid[i:] and not bid[i:].isnumeric():
            i += 1
            # Allow multi-character frame ids such as aA, bCD etc.
            while bid[i:] and bid[i].isalpha() and bid[i].isupper():
                i += 1
            frame_bid = bid[:i]  # bid of the next frame to select
            frame_elem = current_frame.get_by_test_id(frame_bid)
            if not await frame_elem.count():
                raise ValueError(f'Could not find element with bid "{bid}"')
            if scroll_into_view:
                await frame_elem.scroll_into_view_if_needed(timeout=500)
            current_frame = frame_elem.frame_locator(':scope')

        # Finally, we should have selected the frame where the target element is
        elem = current_frame.get_by_test_id(bid)
        if not await elem.count():
            raise ValueError(f'Could not find element with bid "{bid}"')
        if scroll_into_view:
            await elem.scroll_into_view_if_needed(timeout=500)
        return elem

    async def _get_elem_by_selector(self, selector: str, index: int = 0, scroll_into_view: bool = False):
        """Get element locator by CSS selector.

        Args:
            selector: CSS selector string.
            index: Element index if selector matches multiple elements.
            scroll_into_view: try to scroll element into view, unless it is completely visible.

        Returns:
            Playwright Locator for the element.
        """
        page = self.current_page
        elem = page.locator(selector).nth(index)

        if not await elem.count():
            raise ValueError(f'Could not find element with selector "{selector}" at index {index}')

        if scroll_into_view:
            await elem.scroll_into_view_if_needed(timeout=500)

        return elem

    async def _resolve_locator(
        self,
        locator: 'BidLocator | SelectorLocator',
        scroll_into_view: bool = False,
    ) -> tuple[Any, str | None]:
        """Resolve locator to element and optional bid.

        Args:
            locator: Element locator (BidLocator or SelectorLocator).
            scroll_into_view: try to scroll element into view.

        Returns:
            (element_locator, bid) - bid may be None for selector locator.
        """
        if isinstance(locator, BidLocator):
            elem = await self._get_elem_by_bid(locator.bid, scroll_into_view)
            return elem, locator.bid

        if isinstance(locator, SelectorLocator):
            elem = await self._get_elem_by_selector(locator.selector, locator.index, scroll_into_view)
            # Try to get bid for demo mode effects
            bid = await elem.get_attribute(self.BID_ATTR)
            return elem, bid

        raise ValueError(f'Unknown locator type: {type(locator)}')

    # =========================================================================
    # Demo Mode Methods (aligned with BrowserGym)
    # =========================================================================

    async def _highlight_by_box(self, page: Page, box: dict, color: str = 'blue'):
        """Highlight element by bounding box with wave animation effect."""
        if not box or color not in ('blue', 'red'):
            return

        left, top, width, height = box['x'], box['y'], box['width'], box['height']
        await page.evaluate(
            f"""
const overlay = document.createElement('div');
document.body.appendChild(overlay);
overlay.setAttribute('style', `
    all: initial;
    position: fixed;
    border: 2px solid transparent;
    borderRadius: 10px;
    boxShadow: 0 0 0px {color};
    left: {left - 2}px;
    top: {top - 2}px;
    width: {width}px;
    height: {height}px;
    z-index: 2147483646;
    pointerEvents: none;
`);

let spread = 0;
const waveInterval = setInterval(() => {{
    spread += 10;
    overlay.style.boxShadow = `0 0 40px ${{spread}}px {color}`;
    overlay.style.opacity = 1 - spread / 38;
    if (spread >= 38) {{
        clearInterval(waveInterval);
        document.body.removeChild(overlay);
    }}
}}, 200);
"""
        )
        # Wait for animation
        await page.wait_for_timeout(1000)

    async def _smooth_move_visual_cursor_to(self, page: Page, x: float, y: float, speed: float = 400):
        """Smoothly move visual cursor to target position."""
        movement_time = await page.evaluate(
            """
([targetX, targetY, speed]) => {
    if (!("browsergym_visual_cursor" in window)) {
        if (window.trustedTypes && window.trustedTypes.createPolicy) {
            window.trustedTypes.createPolicy('default', {
                createHTML: (string, sink) => string
            });
        }
        let cursor = document.createElement('div');
        cursor.setAttribute('id', 'browsergym-visual-cursor');
        cursor.innerHTML = `
            <svg width="50px" height="50px" viewBox="213 106 713 706" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M213.333 106.667L426.667 853.333 512 512 853.333 426.667 213.333 106.667z" fill="blue"/>
            </svg>
        `;
        cursor.setAttribute('style', `
            all: initial;
            position: fixed;
            opacity: 0.7;
            z-index: 2147483647;
            pointer-events: none;
        `);

        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        cursor.style.left = `${centerX}px`;
        cursor.style.top = `${centerY}px`;

        window.browsergym_visual_cursor = cursor;
        window.browsergym_visual_cursor_n_owners = 0;
    }

    let cursor = window.browsergym_visual_cursor;
    document.body.appendChild(cursor);
    window.browsergym_visual_cursor_n_owners += 1;

    let x = parseFloat(cursor.style.left);
    let y = parseFloat(cursor.style.top);

    let dx = targetX - x;
    let dy = targetY - y;
    let dist = Math.hypot(dx, dy);
    let movement_time = (dist / speed) * 1000;
    let still_wait_time = 1000;

    let steps = Math.max(1, Math.trunc(dist / 10));
    let step_dx = dx / steps;
    let step_dy = dy / steps;
    let step_wait_time = Math.max(10, movement_time / steps);

    let step = 0;
    let time_still = 0;
    const cursorInterval = setInterval(() => {
        if (step < steps) {
            x += step_dx;
            y += step_dy;
            cursor.style.left = `${x}px`;
            cursor.style.top = `${y}px`;
        } else if (time_still < still_wait_time) {
            time_still += step_wait_time;
        } else {
            clearInterval(cursorInterval);
            window.browsergym_visual_cursor_n_owners -= 1;
            if (window.browsergym_visual_cursor_n_owners <= 0) {
                document.body.removeChild(cursor);
            }
        }
        step += 1;
    }, step_wait_time);

    return movement_time;
}""",
            [x, y, speed],
        )
        await page.wait_for_timeout(movement_time)

    async def _check_for_overlay(self, page: Page, bid: str, elem, box: dict) -> bool:
        """Check if element is visible (not covered by other elements)."""
        if not elem:
            return False

        # First check visibility ratio attribute
        try:
            visibility = await elem.get_attribute(self.VIS_ATTR)
            if visibility is not None:
                return float(visibility) >= 0.5
        except Exception:
            pass

        # Fallback: check corners
        if box:
            points_to_check = [
                (box['x'], box['y']),
                (box['x'] + box['width'], box['y']),
                (box['x'], box['y'] + box['height']),
                (box['x'] + box['width'], box['y'] + box['height']),
            ]

            for px, py in points_to_check:
                top_element = await page.evaluate(
                    f"""() => {{
                        const el = document.elementFromPoint({px}, {py});
                        return el ? el.outerHTML : '';
                    }}"""
                )
                if top_element and bid in top_element:
                    return True

        return False

    async def _add_demo_mode_effects(
        self,
        page: Page,
        elem,
        bid: str,
        move_cursor: bool = True,
        highlight_box: bool = True,
    ):
        """Add visual effects for demo mode (cursor movement and highlight)."""
        if self.demo_mode == 'off':
            return

        try:
            box = await elem.bounding_box()
        except Exception:
            return

        if not box:
            return

        center_x = box['x'] + box['width'] / 2
        center_y = box['y'] + box['height'] / 2
        is_top_element = await self._check_for_overlay(page, bid, elem, box)

        if self.demo_mode == 'only_visible_elements':
            if not is_top_element:
                return
            color = 'blue'
        elif self.demo_mode == 'default':
            color = 'blue' if is_top_element else 'red'
        elif self.demo_mode == 'all_blue':
            color = 'blue'
        else:
            return

        if move_cursor:
            await self._smooth_move_visual_cursor_to(page, center_x, center_y)

        if highlight_box:
            await self._highlight_by_box(page, box, color=color)

    # =========================================================================
    # Navigation and Action Methods
    # =========================================================================

    async def goto(self, url: str, timeout: int | None = None) -> str:
        """Navigate to URL."""
        timeout = timeout or self.timeout
        await self.current_page.goto(url, timeout=timeout, wait_until='domcontentloaded')
        # Wait a bit for JS to execute
        await asyncio.sleep(0.5)
        return self.current_page.url

    async def click(
        self,
        locator: 'ElementLocator',
        timeout: int | None = None,
        button: Literal['left', 'right', 'middle'] = 'left',
    ):
        """Click element by locator (supports bid, xy, selector).

        Args:
            locator: Element locator (BidLocator, XYLocator, or SelectorLocator).
            timeout: Click timeout in milliseconds.
            button: Mouse button to click.
        """
        timeout = timeout or self.timeout

        # Handle XY locator - click at coordinates
        if isinstance(locator, XYLocator):
            if self.demo_mode != 'off':
                await self._smooth_move_visual_cursor_to(self.current_page, locator.x, locator.y)
                await self._highlight_by_box(
                    self.current_page,
                    {'x': locator.x - 5, 'y': locator.y - 5, 'width': 10, 'height': 10},
                    color='blue',
                )
            await self.current_page.mouse.click(locator.x, locator.y, button=button)
            await asyncio.sleep(0.2)
            return

        # Handle BidLocator or SelectorLocator
        elem, bid = await self._resolve_locator(locator, scroll_into_view=(self.demo_mode != 'off'))
        if self.demo_mode != 'off' and bid:
            await self._add_demo_mode_effects(self.current_page, elem, bid, move_cursor=True)
        await elem.click(timeout=timeout, button=button)
        await asyncio.sleep(0.2)  # Wait for potential page updates

    async def fill(self, locator: 'BidLocator | SelectorLocator', value: str, timeout: int | None = None):
        """Fill input element by locator (supports bid, selector).

        Args:
            locator: Element locator (BidLocator or SelectorLocator). XY not supported.
            value: Value to fill.
            timeout: Fill timeout in milliseconds.
        """
        timeout = timeout or self.timeout
        elem, bid = await self._resolve_locator(locator, scroll_into_view=(self.demo_mode != 'off'))
        # Demo mode effects (no cursor movement for fill)
        if self.demo_mode != 'off' and bid:
            await self._add_demo_mode_effects(self.current_page, elem, bid, move_cursor=False)
        await elem.fill(value, timeout=timeout)
        await asyncio.sleep(0.1)

    async def scroll(self, direction: str, amount: int = 500):
        """Scroll the page."""
        delta_y = amount if direction == 'down' else -amount
        await self.current_page.evaluate(f'window.scrollBy(0, {delta_y})')
        await asyncio.sleep(0.2)

    async def press_key(self, key: str, locator: 'BidLocator | SelectorLocator | None' = None):
        """Press a keyboard key, optionally focusing an element first.

        Args:
            key: Key to press (e.g., 'Enter', 'Escape', 'Tab').
            locator: Optional element locator to focus first (BidLocator or SelectorLocator).
        """
        if locator is not None:
            elem, bid = await self._resolve_locator(locator, scroll_into_view=(self.demo_mode != 'off'))
            # Demo mode effects
            if self.demo_mode != 'off' and bid:
                await self._add_demo_mode_effects(self.current_page, elem, bid, move_cursor=True)
            await elem.focus()
        await self.current_page.keyboard.press(key)
        await asyncio.sleep(0.1)

    async def hover(self, locator: 'ElementLocator', timeout: int | None = None):
        """Hover over element or position by locator (supports bid, xy, selector).

        Args:
            locator: Element locator (BidLocator, XYLocator, or SelectorLocator).
            timeout: Hover timeout in milliseconds.
        """
        timeout = timeout or self.timeout

        # Handle XY locator - move mouse to coordinates
        if isinstance(locator, XYLocator):
            if self.demo_mode != 'off':
                await self._smooth_move_visual_cursor_to(self.current_page, locator.x, locator.y)
            await self.current_page.mouse.move(locator.x, locator.y)
            await asyncio.sleep(0.1)
            return

        # Handle BidLocator or SelectorLocator
        elem, bid = await self._resolve_locator(locator, scroll_into_view=(self.demo_mode != 'off'))
        if self.demo_mode != 'off' and bid:
            await self._add_demo_mode_effects(self.current_page, elem, bid, move_cursor=True)
        await elem.hover(timeout=timeout)
        await asyncio.sleep(0.1)

    async def select_option(self, locator: 'BidLocator | SelectorLocator', value: str, timeout: int | None = None):
        """Select option from dropdown by locator (supports bid, selector).

        Args:
            locator: Element locator (BidLocator or SelectorLocator). XY not supported.
            value: Option value to select.
            timeout: Select timeout in milliseconds.
        """
        timeout = timeout or self.timeout
        elem, bid = await self._resolve_locator(locator, scroll_into_view=(self.demo_mode != 'off'))
        # Demo mode effects
        if self.demo_mode != 'off' and bid:
            await self._add_demo_mode_effects(self.current_page, elem, bid, move_cursor=True)
        await elem.select_option(value, timeout=timeout)
        await asyncio.sleep(0.1)

    async def keyboard_type(self, text: str, delay: int = 0):
        """Type text using keyboard (useful after clicking with xy coordinates).

        This method types text character by character using the keyboard,
        which is useful when you need to type into a focused element
        (e.g., after clicking with xy coordinates).

        Args:
            text: Text to type.
            delay: Delay between key presses in milliseconds.
        """
        await self.current_page.keyboard.type(text, delay=delay)
        await asyncio.sleep(0.1)

    async def new_tab(self, url: str | None = None) -> int:
        """Open a new tab."""
        page = await self.context.new_page()
        self.pages.append(page)
        new_index = len(self.pages) - 1
        self.current_page_index = new_index

        if url:
            await self.goto(url)

        return new_index

    async def close_tab(self, tab_index: int | None = None):
        """Close a tab."""
        if tab_index is None:
            tab_index = self.current_page_index

        if tab_index < 0 or tab_index >= len(self.pages):
            raise ValueError(f'Invalid tab index: {tab_index}')

        page = self.pages[tab_index]
        await page.close()
        self.pages.pop(tab_index)

        # Adjust current page index if needed
        if self.current_page_index >= len(self.pages):
            self.current_page_index = max(0, len(self.pages) - 1)

    async def focus_tab(self, tab_index: int):
        """Focus a specific tab."""
        if tab_index < 0 or tab_index >= len(self.pages):
            raise ValueError(f'Invalid tab index: {tab_index}')
        self.current_page_index = tab_index
        await self.current_page.bring_to_front()

    async def list_tabs(self) -> list[dict[str, Any]]:
        """List all tabs with their information.

        Returns:
            List of dicts containing tab info: index, url, title, is_active
        """
        tabs = []
        for i, page in enumerate(self.pages):
            try:
                url = page.url
                title = await page.title()
            except Exception:
                url = ''
                title = ''
            tabs.append({
                'index': i,
                'url': url,
                'title': title,
                'is_active': i == self.current_page_index,
            })
        return tabs

    async def go_back(self) -> str:
        """Go back in browser history."""
        await self.current_page.go_back(wait_until='domcontentloaded')
        await asyncio.sleep(0.3)
        return self.current_page.url

    async def go_forward(self) -> str:
        """Go forward in browser history."""
        await self.current_page.go_forward(wait_until='domcontentloaded')
        await asyncio.sleep(0.3)
        return self.current_page.url

    # Maximum retries for observation extraction (aligned with BrowserGym)
    EXTRACT_OBS_MAX_TRIES = 5

    async def get_observation(
        self,
        include_screenshot: bool = False,
        screenshot_scale_factor: float = 1.0,
        viewport_only: bool = False,
    ) -> dict[str, Any]:
        """Get current page observation with accessibility tree (aligned with BrowserGym's _get_obs).

        Args:
            include_screenshot: Whether to include a screenshot.
            screenshot_scale_factor: Scale factor for higher resolution screenshots (default 1.0).
            viewport_only: If True, only include elements visible in the current viewport.
        """
        page = self.current_page

        # Wait for DOM to be loaded on all pages and frames
        await self._wait_dom_loaded()

        # Get URL and title
        url = page.url
        title = await page.title()

        # Retry mechanism for robustness against async events (detached frames, etc.)
        axtree = ''
        focused_element_bid = ''
        for retries_left in reversed(range(self.EXTRACT_OBS_MAX_TRIES)):
            try:
                # Pre-extraction: mark dom elements (set bid, set dynamic attributes)
                await self._mark_page_elements(page, lenient=(retries_left == 0))

                # Extract visibility ratios if viewport_only is requested
                visibility_map: dict[str, float] = {}
                if viewport_only:
                    visibility_map = await self._extract_visibility_map(page)

                # Extract accessibility tree
                axtree = await self._get_accessibility_tree(page, visibility_map if viewport_only else None)

                # Extract focused element bid
                focused_element_bid = await self._extract_focused_element_bid(page)

            except Exception as e:
                err_msg = str(e)
                # Try to add robustness to async events (detached / deleted frames)
                if retries_left > 0 and any(
                    msg in err_msg
                    for msg in (
                        'Frame was detached',
                        'Frame with the given frameId is not found',
                        'Execution context was destroyed',
                        'Frame has been detached',
                        'Cannot mark a child frame without a bid',
                        'Cannot read properties of undefined',
                    )
                ):
                    logger.warning(
                        f'Error extracting observation. Retrying ({retries_left}/{self.EXTRACT_OBS_MAX_TRIES} tries left). {repr(e)}'
                    )
                    # Post-extract cleanup (ARIA attributes)
                    await self._unmark_page_elements(page)
                    await asyncio.sleep(0.5)
                    continue
                raise e
            break

        # Post-extraction cleanup of temporary info in dom
        await self._unmark_page_elements(page)

        # Optionally get screenshot using CDP
        screenshot = None
        if include_screenshot:
            try:
                screenshot = await self._extract_screenshot_cdp(page, scale_factor=screenshot_scale_factor)
            except Exception as e:
                logger.warning(f'CDP screenshot failed, falling back to Playwright screenshot: {e}')
                screenshot_bytes = await page.screenshot(type='png')
                import base64

                screenshot = base64.b64encode(screenshot_bytes).decode('utf-8')

        return {
            'url': url,
            'title': title,
            'axtree': axtree,
            'focused_element_bid': focused_element_bid,
            'screenshot': screenshot,
        }

    async def screenshot(self, options: dict[str, Any]) -> dict[str, Any]:
        """Capture current page screenshot via Playwright page.screenshot()."""
        page = self.current_page
        screenshot_bytes = await page.screenshot(**options)

        image_type = options.get('type')
        if image_type is None:
            path = str(options.get('path') or '').lower()
            image_type = 'jpeg' if path.endswith(('.jpg', '.jpeg')) else 'png'
        media_type = 'image/jpeg' if image_type == 'jpeg' else 'image/png'

        try:
            title = await page.title()
        except Exception:
            title = ''

        return {
            'screenshot': base64.b64encode(screenshot_bytes).decode('utf-8'),
            'media_type': media_type,
            'url': page.url,
            'title': title,
            'path': options.get('path'),
        }

    async def _extract_focused_element_bid(self, page: Page) -> str:
        """Extract the bid of the currently focused element (aligned with BrowserGym).

        This traverses through iframes and shadow DOMs to find the active element.
        """
        extract_focused_element_script = """
() => {
    // This recursive function traverses shadow DOMs
    function getActiveElement(root) {
        const active_element = root.activeElement;

        if (!active_element) {
            return null;
        }

        if (active_element.shadowRoot) {
            return getActiveElement(active_element.shadowRoot);
        } else {
            return active_element;
        }
    }
    return getActiveElement(document);
}"""
        # Traverse through iFrames
        frame = page
        focused_bid = ''
        try:
            while frame:
                focused_element = await frame.evaluate_handle(extract_focused_element_script)
                element = focused_element.as_element()
                if element:
                    content_frame = await element.content_frame()
                    focused_bid = await element.get_attribute(self.BID_ATTR) or ''
                    frame = content_frame
                else:
                    frame = None
        except Exception:
            focused_bid = ''

        return focused_bid

    async def _wait_dom_loaded(self):
        """Wait for DOM to be loaded on all pages and frames (aligned with BrowserGym).

        This ensures the page is stable before extracting observations.
        """
        for page in self.pages:
            with contextlib.suppress(Exception):
                await page.wait_for_load_state('domcontentloaded', timeout=3000)
            for frame in page.frames:
                with contextlib.suppress(Exception):
                    await frame.wait_for_load_state('domcontentloaded', timeout=3000)

    async def _extract_screenshot_cdp(self, page: Page, scale_factor: float = 1.0) -> str:
        """Extract screenshot using CDP for better control (aligned with BrowserGym).

        Args:
            page: The Playwright page to capture.
            scale_factor: Scale factor for higher resolution screenshots (default 1.0).

        Returns:
            Base64 encoded PNG screenshot.
        """
        cdp_session = await self.context.new_cdp_session(page)

        viewport = page.viewport_size
        if viewport is None:
            dimensions = await page.evaluate(
                """() => ({
                    width: window.innerWidth,
                    height: window.innerHeight,
                    devicePixelRatio: window.devicePixelRatio
                })"""
            )
        else:
            dimensions = {
                'width': viewport['width'],
                'height': viewport['height'],
                'devicePixelRatio': 1.0,
            }

        # Capture original metrics before changing
        original_metrics = {
            'width': dimensions['width'],
            'height': dimensions['height'],
            'deviceScaleFactor': 1.0,
            'mobile': False,
        }

        # Apply scale factor to device metrics for higher resolution capture
        await cdp_session.send(
            'Emulation.setDeviceMetricsOverride',
            {
                'width': dimensions['width'],
                'height': dimensions['height'],
                'deviceScaleFactor': scale_factor,
                'mobile': False,
            },
        )

        # Capture screenshot
        cdp_answer = await cdp_session.send(
            'Page.captureScreenshot',
            {
                'format': 'png',
            },
        )

        # Restore original metrics
        await cdp_session.send('Emulation.setDeviceMetricsOverride', original_metrics)

        await cdp_session.detach()

        # Return base64 encoded PNG
        return cdp_answer['data']

    # JavaScript for marking elements (aligned with BrowserGym's frame_mark_elements.js)
    FRAME_MARK_ELEMENTS_JS = """
async ([parent_bid, bid_attr_name, tags_to_mark]) => {
    // standard html tags
    const html_tags = new Set([
        "a", "abbr", "acronym", "address", "applet", "area", "article", "aside", "audio",
        "b", "base", "basefont", "bdi", "bdo", "big", "blockquote", "body", "br", "button",
        "canvas", "caption", "center", "cite", "code", "col", "colgroup", "data", "datalist",
        "dd", "del", "details", "dfn", "dialog", "dir", "div", "dl", "dt", "em", "embed",
        "fieldset", "figcaption", "figure", "font", "footer", "form", "frame", "frameset",
        "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "i",
        "iframe", "img", "input", "ins", "kbd", "label", "legend", "li", "link", "main",
        "map", "mark", "menu", "meta", "meter", "nav", "noframes", "noscript", "object",
        "ol", "optgroup", "option", "output", "p", "param", "picture", "pre", "progress",
        "q", "rp", "rt", "ruby", "s", "samp", "script", "search", "section", "select",
        "small", "source", "span", "strike", "strong", "style", "sub", "summary", "sup",
        "svg", "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead",
        "time", "title", "tr", "track", "tt", "u", "ul", "var", "video", "wbr"
    ]);
    const set_of_marks_tags = new Set([
        "input", "textarea", "select", "button", "a", "iframe", "video", "li", "td", "option"
    ]);

    let browsergym_first_visit = false;
    if (!("browsergym_elem_counter" in window)) {
        window.browsergym_elem_counter = 0;
        window.browsergym_frame_id_generator = new IFrameIdGenerator();
        browsergym_first_visit = true;
    }

    // Mechanism for computing all element's visibility
    let elems_to_be_visited = new Set();
    let intersection_observer = new IntersectionObserver(
        entries => {
            entries.forEach(entry => {
                let elem = entry.target;
                elem.setAttribute('browsergym_visibility_ratio', Math.round(entry.intersectionRatio * 100) / 100);
                if (elems_to_be_visited.has(elem)) {
                    elems_to_be_visited.delete(elem);
                }
            })
        },
        {
            threshold: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    );

    let all_bids = new Set();
    let elements = Array.from(document.querySelectorAll('*'));
    let som_buttons = [];
    let i = 0;

    while (i < elements.length) {
        const elem = elements[i];
        // Add shadowDOM elements to the elements array
        if (elem.shadowRoot !== null) {
            elements = new Array(
                ...Array.prototype.slice.call(elements, 0, i + 1),
                ...Array.from(elem.shadowRoot.querySelectorAll("*")),
                ...Array.prototype.slice.call(elements, i + 1)
            );
        }
        i++;

        // Decide if the current element should be marked
        switch (tags_to_mark) {
            case "all":
                break;
            case "standard_html":
                if (!elem.tagName || !html_tags.has(elem.tagName.toLowerCase())) {
                    continue;
                }
                break;
            default:
                throw new Error(`Invalid value for parameter "tags_to_mark": ${JSON.stringify(tags_to_mark)}`);
        }

        // Register intersection callback on element
        elem.setAttribute('browsergym_visibility_ratio', 0);
        elems_to_be_visited.add(elem);
        intersection_observer.observe(elem);

        // Write dynamic element values to the DOM
        if (typeof elem.value !== 'undefined') {
            elem.setAttribute("value", elem.value);
        }
        // Write dynamic checked properties to the DOM
        if (typeof elem.checked !== 'undefined') {
            if (elem.checked === true) {
                elem.setAttribute("checked", "");
            } else {
                elem.removeAttribute("checked");
            }
        }

        // Add the element global id (browsergym id)
        let elem_global_bid = null;
        if (elem.hasAttribute(bid_attr_name)) {
            if (browsergym_first_visit) {
                throw new Error(`Attribute ${bid_attr_name} already used in element ${elem.outerHTML}`);
            }
            elem_global_bid = elem.getAttribute(bid_attr_name);
            if (all_bids.has(elem_global_bid)) {
                console.log(`BrowserGym: duplicate bid ${elem_global_bid} detected, generating a new one`);
                elem_global_bid = null;
            }
        }
        if (elem_global_bid === null) {
            let elem_local_id = null;
            if (['iframe', 'frame'].includes(elem.tagName.toLowerCase())) {
                elem_local_id = `${window.browsergym_frame_id_generator.next()}`;
            } else {
                elem_local_id = `${window.browsergym_elem_counter++}`;
            }
            if (parent_bid == "") {
                elem_global_bid = `${elem_local_id}`;
            } else {
                elem_global_bid = `${parent_bid}${elem_local_id}`;
            }
            elem.setAttribute(bid_attr_name, `${elem_global_bid}`);
        }
        all_bids.add(elem_global_bid);

        // Store custom data inside ARIA attributes (will be available in DOM and AXTree)
        push_bid_to_attribute(elem_global_bid, elem, "aria-roledescription");
        push_bid_to_attribute(elem_global_bid, elem, "aria-description");

        // Set-of-marks flag
        elem.setAttribute("browsergym_set_of_marks", "0");
        if (["self", "child"].includes(whoCapturesCenterClick(elem))) {
            if (set_of_marks_tags.has(elem.tagName.toLowerCase()) || (elem.onclick != null) || (window.getComputedStyle(elem).cursor == "pointer")) {
                let rect = elem.getBoundingClientRect();
                let area = (rect.right - rect.left) * (rect.bottom - rect.top);
                if (area >= 20) {
                    if (som_buttons.every(button => !button.contains(elem))) {
                        let parent = elem.parentElement;
                        if (!(parent && parent.tagName.toLowerCase() == "span" && parent.children.length === 1 && parent.getAttribute("role") && parent.getAttribute("browsergym_set_of_marks") === "1")) {
                            elem.setAttribute("browsergym_set_of_marks", "1");
                            if (elem.matches('button, a, input[type="button"], div[role="button"]')) {
                                som_buttons.push(elem);
                            }
                            while (parent) {
                                if (parent.getAttribute("browsergym_set_of_marks") === "1") {
                                    parent.setAttribute("browsergym_set_of_marks", "0");
                                }
                                parent = parent.parentElement;
                            }
                        }
                    }
                }
            }
        }
    }

    let warning_msgs = [];
    let visibility_marking_timeout = 1000;
    try {
        await until(() => elems_to_be_visited.size == 0, visibility_marking_timeout);
    } catch {
        warning_msgs.push(`Frame marking: not all elements have been visited by the intersection_observer after ${visibility_marking_timeout} ms`);
    }
    intersection_observer.disconnect();
    return warning_msgs;
}

async function until(f, timeout, interval=40) {
    return new Promise((resolve, reject) => {
        const start_time = Date.now();
        if (f()) {
            resolve();
        }
        const wait = setInterval(() => {
            if (f()) {
                clearInterval(wait);
                resolve();
            } else if (Date.now() - start_time > timeout) {
                clearInterval(wait);
                reject();
            }
        }, interval);
    });
}

function whoCapturesCenterClick(element) {
    var rect = element.getBoundingClientRect();
    var x = (rect.left + rect.right) / 2;
    var y = (rect.top + rect.bottom) / 2;
    var element_at_center = elementFromPoint(x, y);
    if (!element_at_center) {
        return "nobody";
    } else if (element_at_center === element) {
        return "self";
    } else if (element.contains(element_at_center)) {
        return "child";
    } else {
        return "non-descendant";
    }
}

function push_bid_to_attribute(bid, elem, attr) {
    let original_content = "";
    if (elem.hasAttribute(attr)) {
        original_content = elem.getAttribute(attr);
    }
    let new_content = `browsergym_id_${bid} ${original_content}`;
    elem.setAttribute(attr, new_content);
}

function elementFromPoint(x, y) {
    let dom = document;
    let last_elem = null;
    let elem = null;
    do {
        last_elem = elem;
        elem = dom.elementFromPoint(x, y);
        dom = elem?.shadowRoot;
    } while(dom && elem !== last_elem);
    return elem;
}

class IFrameIdGenerator {
    constructor(chars = 'abcdefghijklmnopqrstuvwxyz') {
        this._chars = chars;
        this._nextId = [0];
    }
    next() {
        const r = [];
        for (let i = 0; i < this._nextId.length; i++) {
            let char = this._chars[this._nextId[i]];
            if (i < this._nextId.length - 1) {
                char = char.toUpperCase();
            }
            r.unshift(char);
        }
        this._increment();
        return r.join('');
    }
    _increment() {
        for (let i = 0; i < this._nextId.length; i++) {
            const val = ++this._nextId[i];
            if (val < this._chars.length) {
                return;
            }
            this._nextId[i] = 0;
        }
        this._nextId.push(0);
    }
    *[Symbol.iterator]() {
        while (true) {
            yield this.next();
        }
    }
}
"""

    # JavaScript for unmarking elements (aligned with BrowserGym's frame_unmark_elements.js)
    FRAME_UNMARK_ELEMENTS_JS = """
() => {
    let elements = Array.from(document.querySelectorAll('*'));
    let i = 0;
    while (i < elements.length) {
        const elem = elements[i];
        if (elem.shadowRoot !== null) {
            elements = new Array(
                ...Array.prototype.slice.call(elements, 0, i + 1),
                ...Array.from(elem.shadowRoot.querySelectorAll("*")),
                ...Array.prototype.slice.call(elements, i + 1)
            );
        }
        i++;
        pop_bid_from_attribute(elem, "aria-description");
        pop_bid_from_attribute(elem, "aria-roledescription");
    }
}

function pop_bid_from_attribute(elem, attr) {
    let bid_regex = /^browsergym_id[^\\s]*\\s/;
    if (elem.hasAttribute(attr)) {
        let content = elem.getAttribute(attr);
        let original_content = content.replace(bid_regex, '');
        if (original_content) {
            elem.setAttribute(attr, original_content);
        } else {
            elem.removeAttribute(attr);
        }
    }
}
"""

    async def _mark_page_elements(
        self,
        page: Page,
        tags_to_mark: str = 'standard_html',
        lenient: bool = False,
    ):
        """Mark all elements with unique bid (aligned with BrowserGym's _pre_extract).

        Recursively marks DOM elements in all frames (including iframes).
        Key: We embed the bid into ARIA attributes (aria-roledescription and aria-description)
        so that CDP's Accessibility.getFullAXTree can extract them.

        Args:
            page: The Playwright page to mark.
            tags_to_mark: "all" or "standard_html" (default).
            lenient: If True, skip frames without bid instead of raising error.
        """
        import re

        async def mark_frames_recursive(frame, frame_bid: str):
            """Recursively mark elements in frame and its child frames."""
            assert frame_bid == '' or re.match(r'^[a-z][a-zA-Z]*$', frame_bid)
            logger.debug(f'Marking frame {repr(frame_bid)}')

            try:
                # Mark all DOM elements in the frame
                warning_msgs = await frame.evaluate(
                    self.FRAME_MARK_ELEMENTS_JS,
                    [frame_bid, self.BID_ATTR, tags_to_mark],
                )
                for msg in warning_msgs:
                    logger.warning(msg)
            except Exception as e:
                logger.warning(f'Failed to mark frame {repr(frame_bid)}: {e}')
                return

            # Recursively mark all descendant frames
            for child_frame in frame.child_frames:
                # Deal with detached frames
                if child_frame.is_detached:
                    continue
                try:
                    # Deal with weird frames (pdf viewer in <embed>)
                    child_frame_elem = await child_frame.frame_element()
                    if await child_frame_elem.content_frame() != child_frame:
                        logger.warning(f"Skipping frame '{child_frame.name}' for marking, seems problematic.")
                        continue
                    # Deal with sandboxed frames with blocked script execution
                    sandbox_attr = await child_frame_elem.get_attribute('sandbox')
                    if sandbox_attr is not None and 'allow-scripts' not in sandbox_attr.split():
                        continue
                    child_frame_bid = await child_frame_elem.get_attribute(self.BID_ATTR)
                    if child_frame_bid is None:
                        if lenient:
                            logger.warning('Cannot mark a child frame without a bid. Skipping frame.')
                            continue
                        raise RuntimeError('Cannot mark a child frame without a bid.')
                    await mark_frames_recursive(child_frame, frame_bid=child_frame_bid)
                except Exception as e:
                    logger.warning(f'Error processing child frame: {e}')
                    continue

        # Mark all frames recursively starting from main frame
        await mark_frames_recursive(page.main_frame, frame_bid='')

    async def _unmark_page_elements(self, page: Page):
        """Remove ARIA embedded data from elements (aligned with BrowserGym's _post_extract).

        Note: This only removes temporary ARIA data, NOT the bid attribute itself.
        The bid attribute is kept for subsequent action execution.
        """
        for frame in page.frames:
            try:
                if frame != page.main_frame:
                    # Deal with weird frames (pdf viewer in <embed>)
                    frame_elem = await frame.frame_element()
                    if await frame_elem.content_frame() != frame:
                        logger.warning(f"Skipping frame '{frame.name}' for unmarking, seems problematic.")
                        continue
                    # Deal with sandboxed frames with blocked script execution
                    sandbox_attr = await frame_elem.get_attribute('sandbox')
                    if sandbox_attr is not None and 'allow-scripts' not in sandbox_attr.split():
                        continue
                    # Deal with frames without a BID
                    bid = await frame_elem.get_attribute(self.BID_ATTR)
                    if bid is None:
                        continue

                await frame.evaluate(self.FRAME_UNMARK_ELEMENTS_JS)
            except Exception as e:
                error_msg = str(e)
                if 'Frame was detached' in error_msg or 'Frame has been detached' in error_msg:
                    pass
                else:
                    logger.warning(f'Failed to unmark frame: {e}')

    # JavaScript to extract visibility map (bid -> visibility_ratio)
    EXTRACT_VISIBILITY_MAP_JS = """
() => {
    const visibility_map = {};
    const bid_attr = 'bid';
    const vis_attr = 'browsergym_visibility_ratio';

    function collectFromRoot(root) {
        const elements = root.querySelectorAll('*');
        for (const elem of elements) {
            if (elem.shadowRoot) {
                collectFromRoot(elem.shadowRoot);
            }
            const bid = elem.getAttribute(bid_attr);
            const vis = elem.getAttribute(vis_attr);
            if (bid && vis !== null) {
                visibility_map[bid] = parseFloat(vis);
            }
        }
    }

    collectFromRoot(document);
    return visibility_map;
}
"""

    async def _extract_visibility_map(self, page: Page) -> dict[str, float]:
        """Extract a map of bid to visibility_ratio for all elements.

        Returns:
            Dict mapping bid to visibility_ratio (0.0 to 1.0).
        """
        visibility_map: dict[str, float] = {}

        for frame in page.frames:
            try:
                if frame != page.main_frame:
                    frame_elem = await frame.frame_element()
                    if await frame_elem.content_frame() != frame:
                        continue
                    sandbox_attr = await frame_elem.get_attribute('sandbox')
                    if sandbox_attr is not None and 'allow-scripts' not in sandbox_attr.split():
                        continue

                frame_map = await frame.evaluate(self.EXTRACT_VISIBILITY_MAP_JS)
                visibility_map.update(frame_map)
            except Exception as e:
                error_msg = str(e)
                if 'Frame was detached' not in error_msg and 'Frame has been detached' not in error_msg:
                    logger.debug(f'Failed to extract visibility map from frame: {e}')

        return visibility_map

    async def _get_accessibility_tree(self, page: Page, visibility_map: dict[str, float] | None = None) -> str:
        """Extract merged accessibility tree using CDP (aligned with BrowserGym's extract_merged_axtree).

        Extracts AXTree from all frames and merges them together.

        Args:
            page: The Playwright page.
            visibility_map: Optional map of bid to visibility_ratio for filtering viewport-only elements.
        """
        try:
            merged_axtree = await self._extract_merged_axtree(page)
            return self._format_axtree_to_str(merged_axtree, visibility_map)
        except Exception as e:
            logger.warning(f'Failed to get accessibility tree via CDP: {e}')
            return ''

    async def _extract_all_frame_axtrees(self, page: Page) -> dict[str, dict]:
        """Extract AXTree of all frames (aligned with BrowserGym's extract_all_frame_axtrees)."""
        cdp_session = await self.context.new_cdp_session(page)

        # Extract the frame tree
        frame_tree = await cdp_session.send('Page.getFrameTree', {})

        # Extract all frame IDs into a list (BFS through the frame tree)
        frame_ids = []
        root_frame = frame_tree['frameTree']
        frames_to_process = [root_frame]
        while frames_to_process:
            frame = frames_to_process.pop()
            frames_to_process.extend(frame.get('childFrames', []))
            frame_id = frame['frame']['id']
            frame_ids.append(frame_id)

        # Extract the AXTree of each frame
        frame_axtrees = {}
        for frame_id in frame_ids:
            try:
                ax_tree = await cdp_session.send('Accessibility.getFullAXTree', {'frameId': frame_id})
                frame_axtrees[frame_id] = ax_tree
            except Exception as e:
                logger.debug(f'Failed to get AXTree for frame {frame_id}: {e}')

        await cdp_session.detach()

        # Extract browsergym data from ARIA attributes
        for ax_tree in frame_axtrees.values():
            for node in ax_tree.get('nodes', []):
                data_items = self._extract_data_items_from_aria_node(node)
                if data_items:
                    node['browsergym_id'] = data_items[0]

        return frame_axtrees

    def _extract_data_items_from_aria_node(self, node: dict) -> list:
        """Extract browsergym data from ARIA attributes of an AXTree node."""
        import re

        data_items = []
        bid_regex = re.compile(r'^browsergym_id_([a-zA-Z0-9]+)\s?(.*)')

        # Look for data in the node's "roledescription" property
        if 'properties' in node:
            for i, prop in enumerate(node['properties']):
                if prop.get('name') == 'roledescription':
                    value = prop.get('value', {}).get('value', '')
                    match = bid_regex.fullmatch(value)
                    if match:
                        data_items = [match.group(1)]
                        new_value = match.group(2)
                        prop['value']['value'] = new_value
                        if new_value == '':
                            del node['properties'][i]
                    break

        # Look for data in the node's "description" (fallback)
        if 'description' in node:
            value = node['description'].get('value', '')
            match = bid_regex.fullmatch(value)
            if match:
                data_items_bis = [match.group(1)]
                new_value = match.group(2)
                node['description']['value'] = new_value
                if new_value == '':
                    del node['description']
                if not data_items:
                    data_items = data_items_bis

        return data_items

    async def _extract_merged_axtree(self, page: Page) -> dict:
        """Extract merged AXTree (aligned with BrowserGym's extract_merged_axtree)."""
        frame_axtrees = await self._extract_all_frame_axtrees(page)

        cdp_session = await self.context.new_cdp_session(page)

        # Merge all AXTrees into one
        merged_axtree = {'nodes': []}
        for ax_tree in frame_axtrees.values():
            merged_axtree['nodes'].extend(ax_tree.get('nodes', []))
            # Connect each iframe node to the corresponding AXTree root node
            for node in ax_tree.get('nodes', []):
                if node.get('role', {}).get('value') == 'Iframe':
                    try:
                        result = await cdp_session.send('DOM.describeNode', {'backendNodeId': node['backendDOMNodeId']})
                        frame_id = result.get('node', {}).get('frameId')
                        if not frame_id:
                            logger.warning(
                                f'AXTree merging: unable to recover frameId of node with '
                                f'backendDOMNodeId {repr(node["backendDOMNodeId"])}, skipping'
                            )
                        elif frame_id in frame_axtrees:
                            # Root node should always be the first node in the AXTree
                            frame_root_node = frame_axtrees[frame_id]['nodes'][0]
                            assert frame_root_node.get('frameId') == frame_id
                            node.setdefault('childIds', []).append(frame_root_node['nodeId'])
                        else:
                            logger.warning(
                                f"AXTree merging: extracted AXTree does not contain frameId '{frame_id}', skipping"
                            )
                    except Exception as e:
                        logger.debug(f'Failed to describe node for iframe merging: {e}')

        await cdp_session.detach()

        return merged_axtree

    def _format_axtree_to_str(self, ax_tree: dict[str, Any], visibility_map: dict[str, float] | None = None) -> str:
        """Format AXTree to string (aligned with BrowserGym's flatten_axtree_to_str).

        Args:
            ax_tree: The merged accessibility tree.
            visibility_map: Optional map of bid to visibility_ratio. If provided, only elements
                with visibility_ratio > 0 will be included.
        """
        if 'nodes' not in ax_tree or not ax_tree['nodes']:
            return ''

        # Build node index
        node_id_to_idx = {}
        for idx, node in enumerate(ax_tree['nodes']):
            node_id_to_idx[node['nodeId']] = idx

        lines = []

        def get_visibility(bid: str | None) -> float | None:
            """Get visibility ratio for an element. Returns None if no bid."""
            if visibility_map is None:
                return 1.0  # No filtering, all visible
            if bid is None:
                return None  # No bid, inherit from parent
            return visibility_map.get(bid, 0)

        def has_visible_descendants(node_idx: int) -> bool:
            """Check if a node or any of its descendants has a visible element (bid with ratio > 0)."""
            node = ax_tree['nodes'][node_idx]
            bid = node.get('browsergym_id')

            # If this node has a bid and is visible, return True
            vis = get_visibility(bid)
            if vis is not None and vis > 0:
                return True

            # Check children
            for child_id in node.get('childIds', []):
                if (
                    child_id in node_id_to_idx
                    and child_id != node['nodeId']
                    and has_visible_descendants(node_id_to_idx[child_id])
                ):
                    return True

            return False

        def dfs(node_idx: int, depth: int, inherited_visible: bool) -> None:
            """DFS traversal with inherited visibility.

            Args:
                node_idx: Current node index
                depth: Current depth for indentation
                inherited_visible: Whether the nearest ancestor with a bid was visible
            """
            node = ax_tree['nodes'][node_idx]
            indent = '\t' * depth

            # Get role
            role = node.get('role', {}).get('value', '')
            if not role or role == 'LineBreak':
                # Skip line breaks and nodes without role
                # But still process children with same inherited visibility
                for child_id in node.get('childIds', []):
                    if child_id in node_id_to_idx:
                        dfs(node_id_to_idx[child_id], depth, inherited_visible)
                return

            # Get bid (already extracted)
            bid = node.get('browsergym_id')

            # Determine current visibility: if node has bid, check visibility; otherwise inherit from parent
            vis = get_visibility(bid)
            current_visible = vis > 0 if vis is not None else inherited_visible

            # If viewport filtering is enabled and not visible, skip unless has visible descendants
            if visibility_map is not None and not current_visible and not has_visible_descendants(node_idx):
                return

            # Get name
            name = node.get('name', {}).get('value', '')

            # Get value
            value = None
            if 'value' in node and 'value' in node['value']:
                value = node['value']['value']

            # Build node string
            node_str = ''
            if bid:
                node_str = f'[{bid}] '

            if role == 'generic' and not name:
                node_str += f'{role}'
            else:
                node_str += f'{role} {repr(name.strip())}' if name else f'{role}'

            if value is not None:
                node_str += f' value={repr(value)}'

            # Add properties (skip some common ones)
            ignored_props = ('editable', 'readonly', 'level', 'settable', 'multiline', 'invalid', 'focusable')
            for prop in node.get('properties', []):
                prop_name = prop.get('name', '')
                if prop_name in ignored_props or not prop_name:
                    continue
                if 'value' not in prop or 'value' not in prop['value']:
                    continue
                prop_value = prop['value']['value']
                if prop_name in ('required', 'focused', 'atomic'):
                    if prop_value:
                        node_str += f', {prop_name}'
                else:
                    node_str += f', {prop_name}={repr(prop_value)}'

            lines.append(f'{indent}{node_str}')

            # Process children, passing down current visibility
            for child_id in node.get('childIds', []):
                if child_id in node_id_to_idx and child_id != node['nodeId']:
                    dfs(node_id_to_idx[child_id], depth + 1, current_visible)

        # Start from root (root is always considered visible for inheritance purposes)
        if ax_tree['nodes']:
            dfs(0, 0, True)

        return '\n'.join(lines)

    async def close(self):
        """Close all pages and context."""
        for page in self.pages:
            await page.close()
        await self.context.close()


# ============================================================================
# Session Management Executors
# ============================================================================


@executor_registry.register('create_browser_session')
async def create_browser_session(request: CreateBrowserSessionRequest) -> CreateBrowserSessionResponse:
    """Create a new browser session."""
    global _PLAYWRIGHT, _BROWSER

    # Initialize Playwright if needed
    if _PLAYWRIGHT is None:
        _PLAYWRIGHT = await async_playwright().start()
        # Important: change playwright's test id attribute from "data-testid" to "bid"
        # This is required for get_by_test_id() to work with our bid attribute
        _PLAYWRIGHT.selectors.set_test_id_attribute(AsyncBrowserSession.BID_ATTR)
        _BROWSER = await _PLAYWRIGHT.chromium.launch(headless=request.headless, executable_path=request.executable_path)

    # Create context
    if _BROWSER is None:
        raise RuntimeError('Browser not initialized')

    context_options: dict[str, Any] = {
        'viewport': {'width': request.viewport_width, 'height': request.viewport_height},
        'ignore_https_errors': True,
        'user_agent': request.user_agent,
    }

    har_path = None
    if request.record_har:
        if not request.har_path:
            raise ValueError('har_path is required when record_har is True')
        har_path = request.har_path
        context_options['record_har_path'] = har_path
        context_options['record_har_content'] = request.har_content
        if request.har_url_filter:
            context_options['record_har_url_filter'] = request.har_url_filter
        logger.info(f'HAR recording enabled, will save to: {har_path}')

    context = await _BROWSER.new_context(**context_options)
    context.set_default_timeout(request.timeout)

    # Create session
    session_id = str(uuid.uuid4())
    session = AsyncBrowserSession(
        context=context,
        viewport_width=request.viewport_width,
        viewport_height=request.viewport_height,
        timeout=request.timeout,
        demo_mode=request.demo_mode,  # type: ignore[arg-type]
        har_path=har_path,
    )
    await session.initialize()

    BROWSER_SESSIONS[session_id] = session
    logger.info(f'Created browser session: {session_id}')

    return CreateBrowserSessionResponse(session_id=session_id)


@executor_registry.register('destroy_browser_session')
async def destroy_browser_session(request: DestroyBrowserSessionRequest) -> DestroyBrowserSessionResponse:
    """Destroy a browser session."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        logger.warning(f'Session not found: {request.session_id}')
        return DestroyBrowserSessionResponse(success=False, har_path=None)

    har_path = session.har_path
    await session.close()
    del BROWSER_SESSIONS[request.session_id]
    logger.info(f'Destroyed browser session: {request.session_id}')

    return DestroyBrowserSessionResponse(success=True, har_path=har_path)


# ============================================================================
# Navigation Executors
# ============================================================================


@executor_registry.register('browser_goto')
async def browser_goto(request: GotoRequest) -> GotoResponse:
    """Navigate to a URL."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        final_url = await session.goto(request.url, request.timeout)
        return GotoResponse(success=True, url=final_url)
    except Exception as e:
        logger.error(f'Navigation failed: {e}')
        return GotoResponse(success=False, url=session.current_page.url)


@executor_registry.register('browser_go_back')
async def browser_go_back(request: GoBackRequest) -> GoBackResponse:
    """Go back in browser history."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        url = await session.go_back()
        return GoBackResponse(success=True, url=url)
    except Exception as e:
        logger.error(f'Go back failed: {e}')
        return GoBackResponse(success=False, url=session.current_page.url)


@executor_registry.register('browser_go_forward')
async def browser_go_forward(request: GoForwardRequest) -> GoForwardResponse:
    """Go forward in browser history."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        url = await session.go_forward()
        return GoForwardResponse(success=True, url=url)
    except Exception as e:
        logger.error(f'Go forward failed: {e}')
        return GoForwardResponse(success=False, url=session.current_page.url)


# ============================================================================
# Interaction Executors
# ============================================================================


@executor_registry.register('browser_click')
async def browser_click(request: ClickRequest) -> ClickResponse:
    """Click an element or position."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.click(request.locator, request.timeout, request.button)  # type: ignore[arg-type]
        return ClickResponse(success=True)
    except Exception as e:
        logger.error(f'Click failed: {e}')
        return ClickResponse(success=False)


@executor_registry.register('browser_fill')
async def browser_fill(request: FillRequest) -> FillResponse:
    """Fill an input element."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.fill(request.locator, request.value, request.timeout)  # type: ignore[arg-type]
        return FillResponse(success=True)
    except Exception as e:
        logger.error(f'Fill failed: {e}')
        return FillResponse(success=False)


@executor_registry.register('browser_scroll')
async def browser_scroll(request: ScrollRequest) -> ScrollResponse:
    """Scroll the page."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.scroll(request.direction, request.amount)
        return ScrollResponse(success=True)
    except Exception as e:
        logger.error(f'Scroll failed: {e}')
        return ScrollResponse(success=False)


@executor_registry.register('browser_press_key')
async def browser_press_key(request: PressKeyRequest) -> PressKeyResponse:
    """Press a keyboard key."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.press_key(request.key, request.locator)
        return PressKeyResponse(success=True)
    except Exception as e:
        logger.error(f'Press key failed: {e}')
        return PressKeyResponse(success=False)


@executor_registry.register('browser_hover')
async def browser_hover(request: HoverRequest) -> HoverResponse:
    """Hover over an element or position."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.hover(request.locator, request.timeout)  # type: ignore[arg-type]
        return HoverResponse(success=True)
    except Exception as e:
        logger.error(f'Hover failed: {e}')
        return HoverResponse(success=False)


@executor_registry.register('browser_select_option')
async def browser_select_option(request: SelectOptionRequest) -> SelectOptionResponse:
    """Select an option from a dropdown."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.select_option(request.locator, request.value, request.timeout)  # type: ignore[arg-type]
        return SelectOptionResponse(success=True)
    except Exception as e:
        logger.error(f'Select option failed: {e}')
        return SelectOptionResponse(success=False)


@executor_registry.register('browser_keyboard_type')
async def browser_keyboard_type(request: KeyboardTypeRequest) -> KeyboardTypeResponse:
    """Type text using keyboard (useful with xy coordinates)."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.keyboard_type(request.text, request.delay)
        return KeyboardTypeResponse(success=True)
    except Exception as e:
        logger.error(f'Keyboard type failed: {e}')
        return KeyboardTypeResponse(success=False)


@executor_registry.register('browser_mouse_drag')
async def browser_mouse_drag(request: MouseDragRequest) -> MouseDragResponse:
    """Drag from one position to another using mouse."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        page = session.current_page
        await page.mouse.move(request.start_x, request.start_y)
        await page.mouse.down()
        await page.mouse.move(request.end_x, request.end_y, steps=request.steps)
        await page.mouse.up()
        return MouseDragResponse(success=True)
    except Exception as e:
        logger.error(f'Mouse drag failed: {e}')
        return MouseDragResponse(success=False)


# ============================================================================
# Tab Management Executors
# ============================================================================


@executor_registry.register('browser_new_tab')
async def browser_new_tab(request: NewTabRequest) -> NewTabResponse:
    """Open a new tab."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        tab_index = await session.new_tab(request.url)
        return NewTabResponse(success=True, tab_index=tab_index)
    except Exception as e:
        logger.error(f'New tab failed: {e}')
        return NewTabResponse(success=False, tab_index=-1)


@executor_registry.register('browser_close_tab')
async def browser_close_tab(request: CloseTabRequest) -> CloseTabResponse:
    """Close a tab."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.close_tab(request.tab_index)
        return CloseTabResponse(success=True)
    except Exception as e:
        logger.error(f'Close tab failed: {e}')
        return CloseTabResponse(success=False)


@executor_registry.register('browser_focus_tab')
async def browser_focus_tab(request: FocusTabRequest) -> FocusTabResponse:
    """Focus a specific tab."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        await session.focus_tab(request.tab_index)
        return FocusTabResponse(success=True)
    except Exception as e:
        logger.error(f'Focus tab failed: {e}')
        return FocusTabResponse(success=False)


@executor_registry.register('browser_list_tabs')
async def browser_list_tabs(request: ListTabsRequest) -> ListTabsResponse:
    """List all tabs with their information."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        tabs_data = await session.list_tabs()
        tabs = [TabInfo(**tab) for tab in tabs_data]
        return ListTabsResponse(
            success=True,
            tabs=tabs,
            active_tab_index=session.current_page_index,
        )
    except Exception as e:
        logger.error(f'List tabs failed: {e}')
        return ListTabsResponse(success=False, tabs=[], active_tab_index=-1)


# ============================================================================
# Observation Executor
# ============================================================================


@executor_registry.register('browser_get_observation')
async def browser_get_observation(request: GetObservationRequest) -> GetObservationResponse:
    """Get current page observation."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    obs = await session.get_observation(
        include_screenshot=request.include_screenshot,
        screenshot_scale_factor=request.screenshot_scale_factor,
        viewport_only=request.viewport_only,
    )
    return GetObservationResponse(**obs)


@executor_registry.register('browser_screenshot')
async def browser_screenshot(request: ScreenshotRequest) -> ScreenshotResponse:
    """Capture a screenshot via Playwright page.screenshot()."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')

    try:
        result = await session.screenshot(request.playwright_options())
        return ScreenshotResponse(success=True, **result)
    except Exception as e:
        logger.error(f'Screenshot failed: {e}')
        return ScreenshotResponse(success=False, error=str(e), url=session.current_page.url)


# ============================================================================
# High-level Browser Primitives
# ============================================================================


@executor_registry.register('browser_get_url')
async def browser_get_url(request: GetUrlRequest) -> GetUrlResponse:
    """Get current page URL."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')
    return GetUrlResponse(url=session.current_page.url)


@executor_registry.register('browser_eval_js')
async def browser_eval_js(request: EvalJsRequest) -> EvalJsResponse:
    """Evaluate JavaScript expression on current page."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')
    result = await session.current_page.evaluate(request.expression)
    return EvalJsResponse(value=result)


@executor_registry.register('browser_wait_for_selector')
async def browser_wait_for_selector(request: WaitForSelectorRequest) -> WaitForSelectorResponse:
    """Wait for a CSS selector to reach the target state."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')
    try:
        await session.current_page.wait_for_selector(request.selector, timeout=request.timeout, state=request.state)
        return WaitForSelectorResponse(found=True)
    except TimeoutError:
        return WaitForSelectorResponse(found=False)


@executor_registry.register('browser_set_geolocation')
async def browser_set_geolocation(request: SetGeolocationRequest) -> SetGeolocationResponse:
    """Set geolocation on the browser context."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')
    await session.context.grant_permissions(['geolocation'])
    await session.context.set_geolocation({
        'latitude': request.latitude,
        'longitude': request.longitude,
        'accuracy': request.accuracy,
    })
    return SetGeolocationResponse(success=True)


@executor_registry.register('browser_wait_for_load_state')
async def browser_wait_for_load_state(request: WaitForLoadStateRequest) -> WaitForLoadStateResponse:
    """Wait for the page to reach the specified load state."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')
    await session.current_page.wait_for_load_state(request.state, timeout=request.timeout)
    return WaitForLoadStateResponse(success=True)


@executor_registry.register('browser_get_content')
async def browser_get_content(request: GetContentRequest) -> GetContentResponse:
    """Get page HTML content, optionally navigating first."""
    session = BROWSER_SESSIONS.get(request.session_id)
    if not session:
        raise ValueError(f'Session not found: {request.session_id}')
    if request.url:
        await session.goto(request.url)
    content = await session.current_page.content()
    return GetContentResponse(content=content, url=session.current_page.url)
