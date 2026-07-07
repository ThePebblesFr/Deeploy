from typing import List
from pathlib import Path
import re

# -----------------------------------------------------------------------------------------#
#                    Functions for fusing the variables of multiple Network.h               #
# -----------------------------------------------------------------------------------------#

INCLUDE_RE = re.compile(r"^\s*#\s*include\b")
IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\b")
DEFINE_GUARD_RE = re.compile(r"^\s*#\s*define\s+\w+\s*$")
ENDIF_RE = re.compile(r"^\s*#\s*endif\b")
FUNC_PROTO_RE = re.compile(r"^\s*[\w\*\s]+\(\s*[^;{}]*\)\s*;\s*$")


def is_variable_line(line: str) -> bool:
    """
    Returns False for #include statements, the header guard
    (#ifndef/#define/#endif) and function prototypes, True otherwise.
    """
    stripped = line.strip()
    if not stripped:
        return True
    if INCLUDE_RE.match(stripped):
        return False
    if IFNDEF_RE.match(stripped) or DEFINE_GUARD_RE.match(stripped) or ENDIF_RE.match(stripped):
        return False
    if FUNC_PROTO_RE.match(stripped):
        return False
    return True


def fuse_variables(files: List[Path]) -> str:
    """
    Fuses the variable declarations of multiple Network.h files into a single string.
    Args:
        files: List of paths to Network.h files whose variables should be fused.
    Returns:
        The fused variable declarations as a single string.
    """
    fused_lines: List[str] = []

    for path in files:
        text = path.read_text()
        lines = [line for line in text.splitlines(keepends=True) if is_variable_line(line)]

        if fused_lines and fused_lines[-1].strip() != "":
            fused_lines.append("\n")
        fused_lines.extend(lines)

    return "".join(fused_lines)

# -----------------------------------------------------------------------------------------#
#                    Functions for stripping the core-count suffix of variables             #
# -----------------------------------------------------------------------------------------#

CORE_COUNT_SUFFIXES = ("2", "4", "6")

def strip_core_suffix(text: str, model_names: List[str]) -> str:
    """
    Removes the core-count suffix (_2, _4 or _6) that follows a model name
    in every variable occurrence within text.
    Args:
        text: The source text (e.g. a fused Network.h/Network.c) to rewrite.
        model_names: Model names whose core-count suffix should be stripped.
    Returns:
        The text with all `<model_name>_<2|4|6>_` occurrences replaced by `<model_name>_`.
    """
    for model_name in model_names:
        pattern = re.compile(rf"\b{re.escape(model_name)}_(?:{'|'.join(CORE_COUNT_SUFFIXES)})_")
        text = pattern.sub(f"{model_name}_", text)

    return text
