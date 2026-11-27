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
#              Functions for stripping the core-count suffix of variant tags                #
# -----------------------------------------------------------------------------------------#

def strip_variant_core_suffix(text: str, variant_tags: List[str]) -> str:
    """
    Removes the trailing core-count suffix (an arbitrary integer, e.g. _2, _4, _6, _8)
    that follows a variant tag in every variable occurrence within text.

    A variant tag identifies one (task, variant) pair, e.g. "AnomalyDetection_light"
    or "AnomalyDetection_heavy" -- it is the exact per-task model directory name used
    to generate that config's per-model MultiDeeployTest, before testRunner_2_models.py
    appends the core count (e.g. "AnomalyDetection_light_4"). Unlike the old
    core-count-only suffix stripping (CORE_COUNT_SUFFIXES), this is keyed per variant
    tag rather than per base model name: two configs that reuse the SAME variant at
    different core counts (e.g. "AnomalyDetection_light_4" and
    "AnomalyDetection_light_2") canonicalize to the same "AnomalyDetection_light_" tag
    and therefore dedupe, while two configs using DIFFERENT variants of the same task
    (e.g. "AnomalyDetection_light_4" and "AnomalyDetection_heavy_4") keep their own
    distinct tags and are never conflated.
    Args:
        text: The source text (e.g. a fused Network.h/Network.c) to rewrite.
        variant_tags: Per-(task, variant) tags whose core-count suffix should be stripped.
    Returns:
        The text with all `<variant_tag>_<cores>_` occurrences replaced by `<variant_tag>_`.
    """
    for tag in variant_tags:
        pattern = re.compile(rf"\b{re.escape(tag)}_\d+_")
        text = pattern.sub(f"{tag}_", text)

    return text
