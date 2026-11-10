"""Validation for the restricted OSWorld PyAutoGUI action subset."""

from __future__ import annotations

import ast

from typing import NoReturn


class PyAutoGUICodeValidationError(ValueError):
    """Raised when desktop/pyautogui code uses unsupported Python syntax or APIs."""


_BANNED_PYAUTOGUI_METHODS = {
    'getActiveWindow',
    'getActiveWindowTitle',
    'getAllWindows',
    'getWindowsAt',
    'getWindowsWithTitle',
    'screenshot',
    'locate',
    'locateAll',
    'locateAllOnScreen',
    'locateCenterOnScreen',
    'locateOnScreen',
}

_BANNED_PYAUTOGUI_PREFIXES = ('locate',)

_DEFAULT_BANNED_STRING_FRAGMENTS = ('javascript:', 'data:text/html')

_ALLOWED_PLATFORM_METHODS = {
    'architecture',
    'machine',
    'node',
    'platform',
    'processor',
    'python_version',
    'release',
    'system',
    'uname',
    'version',
}


def validate_pyautogui_code(code: str) -> None:
    """Validate code against the pyautogui/time.sleep/platform-only action subset."""
    try:
        tree = ast.parse(code, mode='exec')
    except SyntaxError as exc:
        raise PyAutoGUICodeValidationError(f'pyautogui code is not valid Python: {exc.msg}') from exc

    if not tree.body:
        raise PyAutoGUICodeValidationError('pyautogui code is empty')

    validator = _PyAutoGUIValidator()
    for statement in tree.body:
        validator.validate_statement(statement)


class _PyAutoGUIValidator:
    def validate_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Expr):
            self.validate_call(statement.value, context='statement')
            return
        if isinstance(statement, ast.If):
            self.validate_safe_expression(statement.test, context='if condition')
            if not statement.body:
                self._reject(statement, 'empty if body is not allowed')
            for nested in statement.body:
                self.validate_statement(nested)
            for nested in statement.orelse:
                self.validate_statement(nested)
            return
        self._reject(
            statement,
            'only direct pyautogui.* calls, time.sleep(...), platform helpers, and simple platform-based if blocks '
            'are allowed',
        )

    def validate_call(self, node: ast.AST, *, context: str) -> None:
        if isinstance(node, ast.Call):
            call = node
        else:
            self._reject(node, f'{context} must be a direct function call')

        module_name, method_name = self._call_target(call.func)
        if module_name == 'pyautogui':
            self._validate_pyautogui_method(call, method_name)
        elif (
            module_name == 'time'
            and method_name == 'sleep'
            or module_name == 'platform'
            and method_name in _ALLOWED_PLATFORM_METHODS
        ):
            self._validate_call_arguments(call)
        else:
            self._reject(call, f'{module_name}.{method_name} is not allowed')

    def validate_safe_expression(self, node: ast.AST, *, context: str) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                lowered = node.value.lower()
                for fragment in _DEFAULT_BANNED_STRING_FRAGMENTS:
                    if fragment in lowered:
                        self._reject(node, f'{context} contains forbidden text fragment: {fragment}')
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            self.validate_safe_expression(node.operand, context=context)
            return
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
        ):
            self.validate_safe_expression(node.left, context=context)
            self.validate_safe_expression(node.right, context=context)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                self.validate_safe_expression(value, context=context)
            return
        if isinstance(node, ast.Compare):
            self.validate_safe_expression(node.left, context=context)
            for comparator in node.comparators:
                self.validate_safe_expression(comparator, context=context)
            return
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for item in node.elts:
                self.validate_safe_expression(item, context=context)
            return
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if key is not None:
                    self.validate_safe_expression(key, context=context)
            for value in node.values:
                self.validate_safe_expression(value, context=context)
            return
        if isinstance(node, ast.Starred):
            self.validate_safe_expression(node.value, context=context)
            return
        if isinstance(node, ast.Call):
            self.validate_call(node, context=context)
            return
        self._reject(node, f'{context} contains unsupported expression syntax')

    def _validate_pyautogui_method(self, node: ast.Call, method_name: str) -> None:
        if method_name.startswith('_') or method_name in _BANNED_PYAUTOGUI_METHODS:
            self._reject(node, f'pyautogui.{method_name} is not allowed')
        if any(method_name.startswith(prefix) for prefix in _BANNED_PYAUTOGUI_PREFIXES):
            self._reject(node, f'pyautogui.{method_name} is not allowed')
        self._validate_call_arguments(node)

    def _validate_call_arguments(self, node: ast.Call) -> None:
        for arg in node.args:
            self.validate_safe_expression(arg, context='call argument')
        for keyword in node.keywords:
            if keyword.arg is None:
                self._reject(keyword, 'keyword unpacking is not allowed')
            self.validate_safe_expression(keyword.value, context='call argument')

    def _call_target(self, func: ast.AST) -> tuple[str, str]:
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return func.value.id, func.attr
        line = getattr(func, 'lineno', None)
        suffix = f' at line {line}' if line else ''
        raise PyAutoGUICodeValidationError(
            f'pyautogui code is not allowed: only module.method(...) calls are allowed{suffix}'
        )

    def _reject(self, node: ast.AST, message: str) -> NoReturn:
        line = getattr(node, 'lineno', None)
        suffix = f' at line {line}' if line else ''
        raise PyAutoGUICodeValidationError(f'pyautogui code is not allowed: {message}{suffix}')


__all__ = ['PyAutoGUICodeValidationError', 'validate_pyautogui_code']
