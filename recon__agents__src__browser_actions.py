"""Browser action tools - thin wrappers over browser executors.

These tools are used by agents to interact with web pages.
They directly call browser executors and use session_id from tool_state.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.runtime.executors import (
    ClickRequest,
    CloseTabRequest,
    EvalJsRequest,
    FillRequest,
    FocusTabRequest,
    GetObservationRequest,
    GoBackRequest,
    GoForwardRequest,
    GotoRequest,
    HoverRequest,
    ListTabsRequest,
    MouseDragRequest,
    NewTabRequest,
    PressKeyRequest,
    ScreenshotClip,
    ScreenshotRequest,
    ScrollRequest,
    SelectOptionRequest,
    XYLocator,
)


# ============================================================================
# Input Models
# ============================================================================


class GotoInput(BaseModel):
    """Navigate to a URL."""

    url: str = Field(..., description='URL to navigate to')
    timeout: int | None = Field(default=None, description='Navigation timeout in milliseconds')


class ClickInput(BaseModel):
    """Click an element."""

    bid: str = Field(..., description='Browser element ID to click (from accessibility tree)')
    timeout: int | None = Field(default=None, description='Click timeout in milliseconds')
    button: Literal['left', 'right', 'middle'] = Field(default='left', description='Mouse button to click')


class FillInput(BaseModel):
    """Fill an input element."""

    bid: str = Field(..., description='Browser element ID to fill (from accessibility tree)')
    value: str = Field(..., description='Value to fill in the input')
    timeout: int | None = Field(default=None, description='Fill timeout in milliseconds')


class ScrollInput(BaseModel):
    """Scroll the page."""

    direction: Literal['up', 'down'] = Field(..., description='Scroll direction')
    amount: int = Field(default=500, description='Scroll amount in pixels')


class PressKeyInput(BaseModel):
    """Press a keyboard key."""

    key: str = Field(..., description="Key to press (e.g., 'Enter', 'Escape', 'Tab', 'Backspace')")
    bid: str | None = Field(default=None, description='Optional element to focus before pressing key')


class HoverInput(BaseModel):
    """Hover over an element."""

    bid: str = Field(..., description='Browser element ID to hover over')
    timeout: int | None = Field(default=None, description='Hover timeout in milliseconds')


class SelectOptionInput(BaseModel):
    """Select an option from a dropdown."""

    bid: str = Field(..., description='Browser element ID of the select element')
    value: str = Field(..., description='Option value to select')
    timeout: int | None = Field(default=None, description='Select timeout in milliseconds')


class NewTabInput(BaseModel):
    """Open a new tab."""

    url: str | None = Field(default=None, description='Optional URL to navigate to in new tab')


class CloseTabInput(BaseModel):
    """Close a tab."""

    tab_index: int | None = Field(default=None, description='Tab index to close (default: current tab)')


class FocusTabInput(BaseModel):
    """Focus a specific tab."""

    tab_index: int = Field(..., description='Tab index to focus (0-based)')


class ListTabsInput(BaseModel):
    """List all open tabs."""

    pass


class GoBackInput(BaseModel):
    """Go back in browser history."""

    pass


class GoForwardInput(BaseModel):
    """Go forward in browser history."""

    pass


class GetObservationInput(BaseModel):
    """Get current page observation."""

    include_screenshot: bool = Field(default=False, description='Whether to include a screenshot')


class ScreenshotInput(BaseModel):
    """Take a screenshot of the current page."""

    full_page: bool | None = Field(default=None, description='Whether to take a full-page screenshot')
    type: Literal['png', 'jpeg'] | None = Field(default=None, description='Screenshot image type')
    quality: int | None = Field(default=None, ge=0, le=100, description='JPEG quality from 0 to 100')
    clip: ScreenshotClip | None = Field(default=None, description='Optional clip rectangle')
    omit_background: bool | None = Field(default=None, description='Whether to hide the default white background')
    timeout: int | None = Field(default=None, description='Screenshot timeout in milliseconds')
    animations: Literal['allow', 'disabled'] | None = Field(default=None, description='Animation handling')
    caret: Literal['hide', 'initial'] | None = Field(default=None, description='Caret handling')
    scale: Literal['css', 'device'] | None = Field(default=None, description='Screenshot scaling mode')
    mask_color: str | None = Field(default=None, description='Overlay color for masked elements')
    style: str | None = Field(default=None, description='Stylesheet text to apply while screenshotting')
    options: dict[str, Any] = Field(
        default_factory=dict,
        description='Additional Playwright page.screenshot kwargs. Explicit fields take precedence.',
    )


class ClickAtInput(BaseModel):
    """Click at pixel coordinates."""

    x: float = Field(..., description='X coordinate in pixels')
    y: float = Field(..., description='Y coordinate in pixels')
    button: Literal['left', 'right', 'middle'] = Field(default='left', description='Mouse button to click')


class DoubleClickAtInput(BaseModel):
    """Double-click at pixel coordinates."""

    x: float = Field(..., description='X coordinate in pixels')
    y: float = Field(..., description='Y coordinate in pixels')


class DragInput(BaseModel):
    """Drag from one position to another."""

    start_x: float = Field(..., description='Start X coordinate in pixels')
    start_y: float = Field(..., description='Start Y coordinate in pixels')
    end_x: float = Field(..., description='End X coordinate in pixels')
    end_y: float = Field(..., description='End Y coordinate in pixels')
    steps: int = Field(default=10, description='Number of intermediate mouse move steps (higher = smoother)')


class EvalJsInput(BaseModel):
    """Execute JavaScript on the current page."""

    expression: str = Field(..., description='JavaScript expression to evaluate (use arrow function for complex logic)')


# ============================================================================
# Tool Implementations
# ============================================================================


@tool_registry.register(
    'browser/goto',
    meta={
        'name': 'goto',
        'description': 'Navigate to a URL in the browser',
        'tool_type': ToolType.RUNTIME,
    },
)
async def goto_tool(params: GotoInput, context: ToolContext) -> ToolResponse:
    """Navigate to a URL."""
    # Get session_id from tool_state
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_goto

    result = await browser_goto(GotoRequest(session_id=session_id, url=params.url, timeout=params.timeout))

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Navigated to: {result.url}'}],
            meta={'success': True, 'url': result.url},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Navigation failed to: {params.url}'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/click',
    meta={
        'name': 'click',
        'description': 'Click an element on the page using its browser ID (bid)',
        'tool_type': ToolType.RUNTIME,
    },
)
async def click_tool(params: ClickInput, context: ToolContext) -> ToolResponse:
    """Click an element."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_click

    result = await browser_click(
        ClickRequest(session_id=session_id, bid=params.bid, timeout=params.timeout, button=params.button)
    )

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Clicked element: {params.bid}'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Click failed on element: {params.bid}'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/fill',
    meta={
        'name': 'fill',
        'description': 'Fill an input element with text using its browser ID (bid)',
        'tool_type': ToolType.RUNTIME,
    },
)
async def fill_tool(params: FillInput, context: ToolContext) -> ToolResponse:
    """Fill an input element."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_fill

    result = await browser_fill(
        FillRequest(session_id=session_id, bid=params.bid, value=params.value, timeout=params.timeout)
    )

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Filled element {params.bid} with: {params.value}'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Fill failed on element: {params.bid}'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/scroll',
    meta={
        'name': 'scroll',
        'description': 'Scroll the page up or down',
        'tool_type': ToolType.RUNTIME,
    },
)
async def scroll_tool(params: ScrollInput, context: ToolContext) -> ToolResponse:
    """Scroll the page."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_scroll

    result = await browser_scroll(
        ScrollRequest(session_id=session_id, direction=params.direction, amount=params.amount)
    )

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Scrolled {params.direction} by {params.amount}px'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Scroll failed'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/press_key',
    meta={
        'name': 'press_key',
        'description': 'Press a keyboard key (e.g., Enter, Escape, Tab). Optionally focus an element first.',
        'tool_type': ToolType.RUNTIME,
    },
)
async def press_key_tool(params: PressKeyInput, context: ToolContext) -> ToolResponse:
    """Press a keyboard key."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_press_key

    result = await browser_press_key(PressKeyRequest(session_id=session_id, key=params.key, bid=params.bid))

    if result.success:
        focus_msg = f' (focused {params.bid})' if params.bid else ''
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Pressed key: {params.key}{focus_msg}'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Press key failed: {params.key}'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/hover',
    meta={
        'name': 'hover',
        'description': 'Hover over an element using its browser ID (bid)',
        'tool_type': ToolType.RUNTIME,
    },
)
async def hover_tool(params: HoverInput, context: ToolContext) -> ToolResponse:
    """Hover over an element."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_hover

    result = await browser_hover(HoverRequest(session_id=session_id, bid=params.bid, timeout=params.timeout))

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Hovered over element: {params.bid}'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Hover failed on element: {params.bid}'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/select_option',
    meta={
        'name': 'select_option',
        'description': 'Select an option from a dropdown using the select element bid',
        'tool_type': ToolType.RUNTIME,
    },
)
async def select_option_tool(params: SelectOptionInput, context: ToolContext) -> ToolResponse:
    """Select an option from a dropdown."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_select_option

    result = await browser_select_option(
        SelectOptionRequest(session_id=session_id, bid=params.bid, value=params.value, timeout=params.timeout)
    )

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f"Selected option '{params.value}' in element: {params.bid}"}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Select option failed on element: {params.bid}'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/new_tab',
    meta={
        'name': 'new_tab',
        'description': 'Open a new browser tab, optionally navigating to a URL',
        'tool_type': ToolType.RUNTIME,
    },
)
async def new_tab_tool(params: NewTabInput, context: ToolContext) -> ToolResponse:
    """Open a new tab."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_new_tab

    result = await browser_new_tab(NewTabRequest(session_id=session_id, url=params.url))

    if result.success:
        msg = f'Opened new tab (index {result.tab_index})'
        if params.url:
            msg += f' at {params.url}'
        return ToolResponse(
            output=[{'type': 'text', 'text': msg}],
            meta={'success': True, 'tab_index': result.tab_index},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Failed to open new tab'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/close_tab',
    meta={
        'name': 'close_tab',
        'description': 'Close a browser tab (default: current tab)',
        'tool_type': ToolType.RUNTIME,
    },
)
async def close_tab_tool(params: CloseTabInput, context: ToolContext) -> ToolResponse:
    """Close a tab."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_close_tab

    result = await browser_close_tab(CloseTabRequest(session_id=session_id, tab_index=params.tab_index))

    if result.success:
        tab_msg = f'tab {params.tab_index}' if params.tab_index is not None else 'current tab'
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Closed {tab_msg}'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Failed to close tab'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/focus_tab',
    meta={
        'name': 'focus_tab',
        'description': 'Switch focus to a specific browser tab by index',
        'tool_type': ToolType.RUNTIME,
    },
)
async def focus_tab_tool(params: FocusTabInput, context: ToolContext) -> ToolResponse:
    """Focus a specific tab."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_focus_tab

    result = await browser_focus_tab(FocusTabRequest(session_id=session_id, tab_index=params.tab_index))

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Focused tab {params.tab_index}'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Failed to focus tab {params.tab_index}'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/list_tabs',
    meta={
        'name': 'list_tabs',
        'description': 'List all open browser tabs with their index, URL, title, and active status',
        'tool_type': ToolType.RUNTIME,
    },
)
async def list_tabs_tool(params: ListTabsInput, context: ToolContext) -> ToolResponse:
    """List all open tabs."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_list_tabs

    result = await browser_list_tabs(ListTabsRequest(session_id=session_id))

    if result.success:
        lines = [f'Open tabs ({len(result.tabs)} total):']
        for tab in result.tabs:
            active_marker = ' [ACTIVE]' if tab.is_active else ''
            lines.append(f'  [{tab.index}] {tab.title or "(no title)"} - {tab.url}{active_marker}')
        return ToolResponse(
            output=[{'type': 'text', 'text': '\n'.join(lines)}],
            meta={
                'success': True,
                'tabs': [t.model_dump() for t in result.tabs],
                'active_tab_index': result.active_tab_index,
            },
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Failed to list tabs'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/go_back',
    meta={
        'name': 'go_back',
        'description': 'Go back in browser history',
        'tool_type': ToolType.RUNTIME,
    },
)
async def go_back_tool(params: GoBackInput, context: ToolContext) -> ToolResponse:
    """Go back in browser history."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_go_back

    result = await browser_go_back(GoBackRequest(session_id=session_id))

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Went back to: {result.url}'}],
            meta={'success': True, 'url': result.url},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Failed to go back'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/go_forward',
    meta={
        'name': 'go_forward',
        'description': 'Go forward in browser history',
        'tool_type': ToolType.RUNTIME,
    },
)
async def go_forward_tool(params: GoForwardInput, context: ToolContext) -> ToolResponse:
    """Go forward in browser history."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_go_forward

    result = await browser_go_forward(GoForwardRequest(session_id=session_id))

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Went forward to: {result.url}'}],
            meta={'success': True, 'url': result.url},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Failed to go forward'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/get_observation',
    meta={
        'name': 'get_observation',
        'description': 'Get current page state including URL, title, and accessibility tree with element IDs',
        'tool_type': ToolType.RUNTIME,
    },
)
async def get_observation_tool(params: GetObservationInput, context: ToolContext) -> ToolResponse:
    """Get current page observation."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_get_observation

    result = await browser_get_observation(
        GetObservationRequest(session_id=session_id, include_screenshot=params.include_screenshot)
    )

    # Format output
    output_parts = [
        f'URL: {result.url}',
        f'Title: {result.title}',
        '',
        'Accessibility Tree:',
        result.axtree,
    ]

    output_content = [{'type': 'text', 'text': '\n'.join(output_parts)}]

    # Add screenshot if included
    if params.include_screenshot and result.screenshot:
        output_content.append({
            'type': 'image',
            'source': {'type': 'base64', 'media_type': 'image/png', 'data': result.screenshot},
        })

    return ToolResponse(
        output=output_content,
        meta={'success': True, 'url': result.url, 'title': result.title},
    )


# ============================================================================
# Coordinate-based tools (for canvas/game scenarios without accessibility tree)
# ============================================================================


@tool_registry.register(
    'browser/screenshot',
    meta={
        'name': 'screenshot',
        'description': 'Take a screenshot of the current page and return the image',
        'tool_type': ToolType.RUNTIME,
    },
)
async def screenshot_tool(params: ScreenshotInput, context: ToolContext) -> ToolResponse:
    """Take a screenshot of the current page."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_screenshot

    result = await browser_screenshot(
        ScreenshotRequest(
            session_id=session_id,
            options=params.options,
            full_page=params.full_page,
            image_type=params.type,
            quality=params.quality,
            clip=params.clip,
            omit_background=params.omit_background,
            timeout=params.timeout,
            animations=params.animations,
            caret=params.caret,
            scale=params.scale,
            mask_color=params.mask_color,
            style=params.style,
        )
    )

    output_content = []
    if result.screenshot:
        output_content.append({
            'type': 'image',
            'source': {'type': 'base64', 'media_type': result.media_type, 'data': result.screenshot},
        })
        output_content.append({'type': 'text', 'text': f'Screenshot taken. Page: {result.title} ({result.url})'})
    else:
        output_content.append({
            'type': 'text',
            'text': f'Failed to capture screenshot: {result.error or "unknown error"}',
        })

    return ToolResponse(
        output=output_content,
        meta={
            'success': result.success,
            'url': result.url,
            'title': result.title,
            'path': result.path,
            'error': result.error,
        },
    )


@tool_registry.register(
    'browser/click_at',
    meta={
        'name': 'click_at',
        'description': 'Click at specific pixel coordinates (x, y) on the page',
        'tool_type': ToolType.RUNTIME,
    },
)
async def click_at_tool(params: ClickAtInput, context: ToolContext) -> ToolResponse:
    """Click at pixel coordinates."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_click

    result = await browser_click(
        ClickRequest(
            session_id=session_id,
            locator=XYLocator(x=params.x, y=params.y),
            button=params.button,
        )
    )

    if result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Clicked at ({params.x}, {params.y})'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Click failed at ({params.x}, {params.y})'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/double_click_at',
    meta={
        'name': 'double_click_at',
        'description': 'Double-click at specific pixel coordinates (x, y) on the page',
        'tool_type': ToolType.RUNTIME,
    },
)
async def double_click_at_tool(params: DoubleClickAtInput, context: ToolContext) -> ToolResponse:
    """Double-click at pixel coordinates."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_click

    # Two rapid clicks at the same position
    result1 = await browser_click(ClickRequest(session_id=session_id, locator=XYLocator(x=params.x, y=params.y)))
    result2 = await browser_click(ClickRequest(session_id=session_id, locator=XYLocator(x=params.x, y=params.y)))

    if result1.success and result2.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Double-clicked at ({params.x}, {params.y})'}],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Double-click failed at ({params.x}, {params.y})'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/drag',
    meta={
        'name': 'drag',
        'description': 'Drag from one position to another. Useful for drag-and-drop or swipe gestures in games.',
        'tool_type': ToolType.RUNTIME,
    },
)
async def drag_tool(params: DragInput, context: ToolContext) -> ToolResponse:
    """Drag from one position to another."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_mouse_drag

    result = await browser_mouse_drag(
        MouseDragRequest(
            session_id=session_id,
            start_x=params.start_x,
            start_y=params.start_y,
            end_x=params.end_x,
            end_y=params.end_y,
            steps=params.steps,
        )
    )

    if result.success:
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': f'Dragged from ({params.start_x}, {params.start_y}) to ({params.end_x}, {params.end_y})',
                }
            ],
            meta={'success': True},
        )
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Drag failed'}],
        meta={'success': False},
    )


@tool_registry.register(
    'browser/eval_js',
    meta={
        'name': 'eval_js',
        'description': 'Execute JavaScript on the current page and return the result. Useful for reading game state.',
        'tool_type': ToolType.RUNTIME,
    },
)
async def eval_js_tool(params: EvalJsInput, context: ToolContext) -> ToolResponse:
    """Execute JavaScript on the current page."""
    session_id = context.tool_state.get('browser_session_id')
    if not session_id:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No browser session available'}],
            meta={'success': False, 'error': 'No session ID in tool_state'},
        )

    from nexus.runtime.executors.browser import browser_eval_js

    try:
        result = await browser_eval_js(EvalJsRequest(session_id=session_id, expression=params.expression))
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Result: {result.value}'}],
            meta={'success': True, 'value': result.value},
        )
    except Exception as e:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'JavaScript execution failed: {e}'}],
            meta={'success': False, 'error': str(e)},
        )
