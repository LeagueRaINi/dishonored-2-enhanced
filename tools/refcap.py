"""Drive the original game for reference captures (Windows only, stdlib only).

Each invocation does one step so a capture session can be scripted or run
interactively and inspected between steps:

  refcap.py launch --game <ROOT> [-- +engine args...]   start the game, print PID
  refcap.py shot <out.png> [--scale N]                    capture the game window
  refcap.py key <NAME> [--hold-ms N]                      press a key or chord (F2, RETURN, A..Z, SHIFT+W, ...)
  refcap.py type <text>                                   type text as Unicode characters
  refcap.py click <x> <y>                                 left-click at window client coords
  refcap.py look <dx> <dy>                                relative mouse motion (camera look)
  refcap.py cmd <command line>                            run an engine command via its hidden console
  refcap.py clip                                          print clipboard text
  refcap.py focus                                         bring the game window to the front
  refcap.py alive                                         exit 0 while the game runs

The game is found by process name Dishonored2.exe. Nothing is written except
the requested PNG. Run with `python -I`.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import os
import struct
import subprocess
import sys
import time
import zlib

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
try:
    ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
except OSError:
    user32.SetProcessDPIAware()

PROCESS_NAME = "Dishonored2.exe"


# ---------------------------------------------------------------- processes / windows
class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def find_pids(name=PROCESS_NAME):
    snap = kernel32.CreateToolhelp32Snapshot(2, 0)
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    pids = []
    ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
    while ok:
        if entry.szExeFile.lower() == name.lower():
            pids.append(entry.th32ProcessID)
        ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    kernel32.CloseHandle(snap)
    return pids


def find_window():
    pids = set(find_pids())
    found = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    def cb(hwnd, _):
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd):
            r = wt.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(r))
            found.append((r.right * r.bottom, hwnd))
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    if not found:
        sys.exit("game window not found")
    return max(found)[1]


def client_origin_and_size(hwnd):
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, r.right, r.bottom


def focus(hwnd):
    if user32.GetForegroundWindow() == hwnd:
        return  # an Alt tap while in game opens the pause menu (observed)
    # Synthetic Alt press lets SetForegroundWindow succeed from a background process.
    send_keys([(0x12, 0)], hold_ms=0)
    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.2)


# ---------------------------------------------------------------- input
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _U(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _U)]


KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, KEYEVENTF_SCANCODE = 0x2, 0x4, 0x8


def _send(inputs):
    arr = (INPUT * len(inputs))(*inputs)
    if user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT)) != len(inputs):
        raise OSError(ctypes.get_last_error(), "SendInput failed")


def _key(vk, up):
    scan = user32.MapVirtualKeyW(vk, 0)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if vk in (0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22):
        flags |= 0x1  # extended
    i = INPUT(type=1)
    i.u.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
    return i


def send_chord(vks, hold_ms):
    """Press all keys (in order), hold, release in reverse order."""
    for vk in vks:
        _send([_key(vk, False)])
    time.sleep(hold_ms / 1000)
    for vk in reversed(vks):
        _send([_key(vk, True)])
    time.sleep(0.03)


def send_keys(keys, hold_ms=40):
    for vk, _ in keys:
        _send([_key(vk, False)])
        time.sleep(hold_ms / 1000)
        _send([_key(vk, True)])
        time.sleep(0.03)


VK = {"RETURN": 0x0D, "ENTER": 0x0D, "ESCAPE": 0x1B, "ESC": 0x1B, "TAB": 0x09, "SPACE": 0x20, "BACK": 0x08,
      "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27, "SHIFT": 0x10, "CTRL": 0x11, "ALT": 0x12,
      "HOME": 0x24, "END": 0x23, "DELETE": 0x2E}
VK.update({f"F{n}": 0x6F + n for n in range(1, 13)})
VK.update({chr(c): c for c in range(ord("A"), ord("Z") + 1)})
VK.update({str(d): 0x30 + d for d in range(10)})


def type_text(text):
    for ch in text:
        down = INPUT(type=1)
        down.u.ki = KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE, 0, 0)
        up = INPUT(type=1)
        up.u.ki = KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)
        _send([down, up])
        time.sleep(0.015)


def click(hwnd, x, y):
    ox, oy, _, _ = client_origin_and_size(hwnd)
    user32.SetCursorPos(ox + x, oy + y)
    time.sleep(0.05)
    for flag in (0x2, 0x4):  # left down, left up
        i = INPUT(type=0)
        i.u.mi = MOUSEINPUT(0, 0, 0, flag, 0, 0)
        _send([i])
        time.sleep(0.05)


def look(dx, dy, steps=20):
    """Relative mouse motion in small steps (raw-input mouse look)."""
    for k in range(steps):
        sx = dx * (k + 1) // steps - dx * k // steps
        sy = dy * (k + 1) // steps - dy * k // steps
        i = INPUT(type=0)
        i.u.mi = MOUSEINPUT(sx, sy, 0, 0x1, 0, 0)  # MOUSEEVENTF_MOVE
        _send([i])
        time.sleep(0.01)


def drag(hwnd, x0, y0, x1, y1, steps=25):
    ox, oy, _, _ = client_origin_and_size(hwnd)
    user32.SetCursorPos(ox + x0, oy + y0)
    time.sleep(0.05)
    for flag in (0x2,):
        i = INPUT(type=0)
        i.u.mi = MOUSEINPUT(0, 0, 0, flag, 0, 0)
        _send([i])
    for k in range(1, steps + 1):
        user32.SetCursorPos(ox + x0 + (x1 - x0) * k // steps, oy + y0 + (y1 - y0) * k // steps)
        time.sleep(0.02)
    i = INPUT(type=0)
    i.u.mi = MOUSEINPUT(0, 0, 0, 0x4, 0, 0)
    _send([i])


# ---------------------------------------------------------------- capture
class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


def grab(hwnd):
    x, y, w, h = client_origin_and_size(hwnd)
    screen = user32.GetDC(0)
    mem = gdi32.CreateCompatibleDC(screen)
    bmp = gdi32.CreateCompatibleBitmap(screen, w, h)
    gdi32.SelectObject(mem, bmp)
    gdi32.BitBlt(mem, 0, 0, w, h, screen, x, y, 0x00CC0020)
    info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(0, screen)
    return w, h, buf.raw  # BGRA rows, top-down


def write_png(path, w, h, bgra, scale):
    ow, oh = w // scale, h // scale
    rows = bytearray()
    for oy in range(oh):
        rows.append(0)
        base = oy * scale * w * 4
        for ox in range(ow):
            i = base + ox * scale * 4
            rows += bytes((bgra[i + 2], bgra[i + 1], bgra[i]))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", ow, oh, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b"")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)
    return ow, oh


def clipboard_text():
    CF_UNICODETEXT = 13
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    if not user32.OpenClipboard(0):
        sys.exit("cannot open clipboard")
    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return ""
        p = kernel32.GlobalLock(h)
        try:
            return ctypes.wstring_at(p)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


# ---------------------------------------------------------------- engine console
CONSOLE_INPUT_ID = 101  # edit control of the hidden "VoidEngine ... Console" window (class "... WinConsole")


def console_input():
    pids = set(find_pids())
    found = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    def cb(hwnd, _):
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, 64)
        if pid.value in pids and cls.value.endswith("WinConsole"):
            found.append(hwnd)
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    if not found:
        sys.exit("engine console window not found")
    edit = user32.GetDlgItem(found[0], CONSOLE_INPUT_ID)
    if not edit:
        sys.exit("engine console input control not found")
    return edit


def console_command(text):
    """Queues one command line through the engine's own console input."""
    user32.SendMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    edit = console_input()
    buf = ctypes.create_unicode_buffer(text)
    user32.SendMessageW(edit, 0x000C, 0, ctypes.addressof(buf))  # WM_SETTEXT
    user32.SendMessageW(edit, 0x0102, 13, 0)  # WM_CHAR '\r'


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--game", required=True)
    p.add_argument("--app-id", default="403640")
    p.add_argument("engine_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("shot")
    p.add_argument("out")
    p.add_argument("--scale", type=int, default=4)
    p = sub.add_parser("key")
    p.add_argument("name")
    p.add_argument("--hold-ms", type=int, default=40)
    p = sub.add_parser("type")
    p.add_argument("text")
    p = sub.add_parser("click")
    p.add_argument("x", type=int)
    p.add_argument("y", type=int)
    p = sub.add_parser("drag")
    for n in ("x0", "y0", "x1", "y1"):
        p.add_argument(n, type=int)
    p = sub.add_parser("look")
    p.add_argument("dx", type=int)
    p.add_argument("dy", type=int)
    p = sub.add_parser("cmd")
    p.add_argument("text")
    sub.add_parser("clip")
    sub.add_parser("focus")
    sub.add_parser("alive")
    a = ap.parse_args()

    if a.cmd == "launch":
        if find_pids():
            sys.exit("game already running")
        exe = os.path.join(a.game, PROCESS_NAME)
        args = [x for x in a.engine_args if x != "--"]
        env = dict(os.environ, SteamAppId=a.app_id, SteamGameId=a.app_id)
        proc = subprocess.Popen([exe, *args], cwd=a.game, env=env)
        print(proc.pid)
        return
    if a.cmd == "alive":
        sys.exit(0 if find_pids() else 1)
    if a.cmd == "clip":
        print(clipboard_text())
        return
    if a.cmd == "cmd":
        console_command(a.text)
        return
    hwnd = find_window()
    if a.cmd == "focus":
        focus(hwnd)
    elif a.cmd == "shot":
        w, h, data = grab(hwnd)
        print("%dx%d -> %dx%d %s" % ((w, h) + write_png(a.out, w, h, data, a.scale) + (a.out,)))
    elif a.cmd == "key":
        focus(hwnd)
        names = a.name.upper().split("+")
        if len(names) == 1:
            send_keys([(VK[names[0]], 0)], a.hold_ms)
        else:
            send_chord([VK[n] for n in names], a.hold_ms)
    elif a.cmd == "type":
        focus(hwnd)
        type_text(a.text)
    elif a.cmd == "drag":
        focus(hwnd)
        drag(hwnd, a.x0, a.y0, a.x1, a.y1)
    elif a.cmd == "look":
        focus(hwnd)
        look(a.dx, a.dy)
    elif a.cmd == "click":
        focus(hwnd)
        click(hwnd, a.x, a.y)


if __name__ == "__main__":
    main()
