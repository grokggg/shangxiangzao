"""Shared OSWorld desktop action-space descriptions."""

COMPUTER_13_ACTION_TYPES = (
    'MOVE_TO',
    'CLICK',
    'MOUSE_DOWN',
    'MOUSE_UP',
    'RIGHT_CLICK',
    'DOUBLE_CLICK',
    'DRAG_TO',
    'SCROLL',
    'TYPING',
    'PRESS',
    'KEY_DOWN',
    'KEY_UP',
    'HOTKEY',
)

FIXED_MARKERS = ('WAIT', 'FAIL', 'DONE')

COMPUTER_13_ACTION_SPACE = """\
Official OSWorld computer_13 action space:
- MOVE_TO: move the cursor to x/y.
- CLICK: click a mouse button at optional x/y; supports button and num_clicks.
- MOUSE_DOWN: press a mouse button.
- MOUSE_UP: release a mouse button.
- RIGHT_CLICK: right click at optional x/y.
- DOUBLE_CLICK: double click at optional x/y.
- DRAG_TO: drag the cursor to x/y.
- SCROLL: scroll by dx and/or dy.
- TYPING: type text.
- PRESS: press and release one key.
- KEY_DOWN: press and hold one key.
- KEY_UP: release one key.
- HOTKEY: press a key combination.
"""

COMPUTER_13_PARAMETER_GUIDE = """\
Use these parameters. Coordinate fields are x/y screen positions; scroll fields are dx/dy deltas:
- MOVE_TO: {"x": number, "y": number}
- CLICK: optional {"x": number, "y": number, "button": "left"|"right"|"middle", "num_clicks": number}
- MOUSE_DOWN or MOUSE_UP: optional {"button": "left"|"right"|"middle"}
- RIGHT_CLICK or DOUBLE_CLICK: optional {"x": number, "y": number}
- DRAG_TO: {"x": number, "y": number}
- SCROLL: {"dx": number, "dy": number}
- TYPING: {"text": string}
- PRESS, KEY_DOWN, or KEY_UP: {"key": string}
- HOTKEY: {"keys": [string, ...]}
"""

COMPUTER_13_FIXED_MARKER_GUIDANCE = (
    'For OSWorld-style tasks where desktop/fixed_marker is available, WAIT, DONE, and FAIL are fixed actions '
    'represented by desktop/fixed_marker, not by desktop/computer_13.'
)

OSWORLD_SCREEN_BOUNDS = (
    'The left top corner of the screen is (0, 0), and the right bottom corner of the screen is (1920, 1080).'
)

PYAUTOGUI_RESTRICTED_METHODS = (
    'Do not call pyautogui.screenshot(). Do not use pyautogui.locate* methods such as pyautogui.locateCenterOnScreen.'
)

PYAUTOGUI_SAFETY_POLICY = (
    'Only direct pyautogui.* calls, time.sleep(...), and read-only platform helpers are available. '
    'Do not import modules. Do not read or write files. '
    'Do not call shell commands, network APIs, or mock-service APIs. '
    'Do not directly modify application state outside the visible GUI.\n'
    'Do not use hidden window-inspection helpers such as pyautogui.getActiveWindow() or '
    'pyautogui.getAllWindows(); use the screenshot observation instead.\n'
    'Do not type javascript: or data:text/html URLs.'
)

PYAUTOGUI_CODE_SHAPE_GUIDANCE = (
    'For multiple lines, include small intervals such as time.sleep(0.5). '
    'Each call must be complete code; no variables or function can be shared from history. '
    'You must specify coordinates yourself based on the observation.'
)


__all__ = [
    'COMPUTER_13_ACTION_SPACE',
    'COMPUTER_13_ACTION_TYPES',
    'COMPUTER_13_FIXED_MARKER_GUIDANCE',
    'COMPUTER_13_PARAMETER_GUIDE',
    'FIXED_MARKERS',
    'OSWORLD_SCREEN_BOUNDS',
    'PYAUTOGUI_CODE_SHAPE_GUIDANCE',
    'PYAUTOGUI_RESTRICTED_METHODS',
    'PYAUTOGUI_SAFETY_POLICY',
]
