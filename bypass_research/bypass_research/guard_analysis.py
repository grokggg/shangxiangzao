"""pyautogui_guard 绕过向量测试"""
from nexus.runtime.pyautogui_guard import validate_pyautogui_code, PyAutoGUICodeValidationError

CASES = [
    ("A1 直接调用 screenshot", "pyautogui.screenshot('/tmp/x.png')\n"),
    ("A2 getattr 字符串", "getattr(pyautogui, 'screenshot')()\n"),
    ("A3 动态拼接 getattr", "getattr(pyautogui, 'screen'+'shot')()\n"),
    ("A4 __getattribute__", "pyautogui.__getattribute__('screenshot')()\n"),
    ("A5 __dict__ 下标", "pyautogui.__dict__['screenshot']()\n"),
    ("A6 模块别名赋值", "pg = pyautogui\npg.screenshot()\n"),
    ("B1 import 后调用", "import pyautogui\nm='screen'+'shot'\ngetattr(pyautogui,m)()\n"),
    ("B2 hex exec", "import pyautogui\nexec(bytes.fromhex('70796175746f6775692e73637265656e73686f742829').decode())\n"),
    ("B3 base64 exec", "import base64,pyautogui\nexec(base64.b64decode('cHlhdXRvZ3VpLnNjcmVlbmNob3QoKQ==').decode())\n"),
    ("C1 import as", "import pyautogui as pg\npg.screenshot()\n"),
    ("C2 from import", "from pyautogui import screenshot\nscreenshot()\n"),
    ("D1 time.sleep 后系统调用", "time.sleep(0.1)\nos.system('id')\n"),
    ("D2 platform 后系统调用", "platform.system()\nimport os\nos.system('id')\n"),
    ("E1 if+getattr", "import platform\nif platform.system():\n    getattr(pyautogui,'screenshot')()\n"),
    ("E2 if+禁用方法", "import platform\nif platform.system()=='Linux':\n    pyautogui.screenshot('/tmp/x.png')\n"),
]

def main():
    results = []
    for name, code in CASES:
        try:
            validate_pyautogui_code(code)
            results.append((name, 'BYPASS', ''))
            print(f"[BYPASS] {name}")
        except PyAutoGUICodeValidationError as e:
            results.append((name, 'BLOCKED', str(e)[:80]))
            print(f"[BLOCKED] {name:30s} {str(e)[:80]}")
    blocked = sum(1 for r in results if r[1]=='BLOCKED')
    print(f"\n总计: {len(results)} 测试, {blocked} 拦截, {len(results)-blocked} 绕过")
    return 0 if blocked == len(results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
