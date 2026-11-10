"""Map tmux-style key names to raw bytes for writing into the PTY."""

_NAMED: dict[str, bytes] = {
    'Enter': b'\r',
    'Tab': b'\t',
    'Space': b' ',
    'Escape': b'\x1b',
    'BSpace': b'\x7f',
    'Up': b'\x1b[A',
    'Down': b'\x1b[B',
    'Right': b'\x1b[C',
    'Left': b'\x1b[D',
    'Home': b'\x1b[H',
    'End': b'\x1b[F',
    'PageUp': b'\x1b[5~',
    'PageDown': b'\x1b[6~',
}


def _control(letter: str) -> bytes:
    """Map 'a'..'z' (or '@', '[', etc.) to its control byte: C-a -> 0x01."""
    code = ord(letter.upper()) & 0x1F
    return bytes([code])


def _token_to_bytes(token: str) -> bytes:
    if token in _NAMED:
        return _NAMED[token]
    if len(token) == 3 and token.startswith('C-'):
        return _control(token[2])
    return token.encode('utf-8')


def keys_to_bytes(keys: str | list[str]) -> bytes:
    """Convert a key spec (single token, literal string, or list) to bytes.

    A bare string that is not a recognized key name is sent literally
    (e.g. 'hello' -> b'hello'). A list maps each element and concatenates.
    """
    if isinstance(keys, list):
        return b''.join(_token_to_bytes(k) for k in keys)
    return _token_to_bytes(keys)
