import json
import os
import sys
from typing import Any


def print_console(text: str = "") -> None:
    payload = f"{text}{os.linesep}"
    try:
        sys.stdout.write(payload)
        sys.stdout.flush()
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        buffer = getattr(sys.stdout, "buffer", None)
        encoded = payload.encode(encoding, errors="backslashreplace")
        if buffer is not None:
            buffer.write(encoded)
            buffer.flush()
            return
        safe_text = encoded.decode(encoding, errors="ignore")
        sys.stdout.write(safe_text)
        sys.stdout.flush()


def dump_json_console(payload: Any, ensure_ascii: bool = False, indent: int = 2) -> None:
    print_console(json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent))
