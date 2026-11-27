import re
from pathlib import Path
from typing import List, Tuple

"""
Combines N independently-generated, full-cluster single-model deployments
(testRunner_tiled_siracusa.py --cores 8, one Network.c/.h per model, each already
using a model-specific global-variable prefix once --name is passed) into one
Round-Robin/Priority-Sequential binary that alternates full-cluster inferences in
time rather than co-residing on disjoint core subsets (unlike DynMultiDeeploy's
SourceFusion.py, which is specifically for the latter).

No tile-level interleaving, weight dedup, or core-split disambiguation is needed
here (each model already keeps every symbol distinct via its own --name prefix,
confirmed empirically: 0 residual "DeeployNetwork_*" occurrences once --name is
passed) -- the ONLY collision across independently-generated files is Deeploy's
always-unqualified `RunNetwork`/`InitNetwork` entry-point names, so this is just
rename-then-concatenate.
"""

_ENTRY_POINT_RE = re.compile(r"\b(RunNetwork|InitNetwork)\b")

_HEADER_GUARD_RE = re.compile(
    r"^#ifndef __DEEPLOY_HEADER__\n#define __DEEPLOY_HEADER__\n(?P<includes>(?:#include.*\n)+)",
    re.MULTILINE,
)


def rename_entry_points(text: str, model_tag: str) -> str:
    """RunNetwork -> RunNetwork_<model_tag>, InitNetwork -> InitNetwork_<model_tag>."""
    return _ENTRY_POINT_RE.sub(lambda m: f"{m.group(1)}_{model_tag}", text)


def fuse_roundrobin_network_c(files: List[Path], model_tags: List[str]) -> str:
    """Renames each file's entry points, then concatenates them (see module docstring)."""
    parts = []
    for path, tag in zip(files, model_tags):
        parts.append(rename_entry_points(path.read_text(), tag))
    return "\n".join(parts)


def fuse_roundrobin_network_h(files: List[Path], model_tags: List[str]) -> str:
    """
    Keeps the first file's #ifndef/#define/#include block and trailing #endif;
    every file's declarations (renamed entry points + already-model-prefixed
    globals) are concatenated in between.
    """
    bodies = []
    includes_block = None
    for path, tag in zip(files, model_tags):
        text = rename_entry_points(path.read_text(), tag).lstrip()
        m = _HEADER_GUARD_RE.match(text)
        if not m:
            raise ValueError(f"{path}: expected file to start with the standard __DEEPLOY_HEADER__ guard")
        if includes_block is None:
            includes_block = m.group("includes")
        body = text[m.end():]
        if body.rstrip().endswith("#endif"):
            body = body.rstrip()[: -len("#endif")]
        bodies.append(body.strip("\n"))

    return (
        "#ifndef __DEEPLOY_HEADER__\n"
        "#define __DEEPLOY_HEADER__\n"
        f"{includes_block}"
        + "\n\n".join(bodies)
        + "\n#endif\n"
    )
