import sys
import re
from pathlib import Path
from typing import Dict, List

from testUtils.ModeSwitchBodyGen import inject_generated_sections

def update_deeploy_test_main(model_name: str):
    """
    Update the Platforms/Siracusa/src/deeploytest_template.c file to replace occurrences of DeeployNetwork with the given model_name.
    """

    main_c_path_template = Path("/app/Deeploy/DeeployTest/Platforms/Siracusa/src/deeploytest_template.c")
    main_c_path = Path("/app/Deeploy/DeeployTest/Platforms/Siracusa/src/deeploytest.c")

    with main_c_path_template.open("r") as f:
        content = f.read()

    pattern = re.compile(r"DeeployNetwork(?P<suffix>\w*)")

    new_code = pattern.sub(lambda m: f"{model_name}{m.group('suffix')}", content)

    with main_c_path.open("w") as f:
        f.write(new_code)

    print(f"[MultiDeeploy] Replaced {len(pattern.findall(content))} occurrences of 'DeeployNetwork' with '{model_name}' in {main_c_path} template.")


def update_multideeploy_test_main(model_names: list[str]):
    """
    Update multideeploytest_template.c by replacing:
      - Model_i  -> model_names[i]
      - MODEL_i  -> model_names[i].upper()
    """

    template_path = Path(
        "/app/Deeploy/DynMultiDeeployTest/Platforms/Siracusa/src/multideeploytest_template.c"
    )
    output_path = Path(
        "/app/Deeploy/DynMultiDeeployTest/Platforms/Siracusa/src/multideeploytest.c"
    )

    content = template_path.read_text()

    for i, model_name in enumerate(model_names):
        idx = i + 1

        # 1) Replace MODEL_i → UPPERCASE
        pattern_upper = re.compile(rf"\bMODEL_{idx}(\w*)\b")
        matches_upper = pattern_upper.findall(content)
        content = pattern_upper.sub(
            lambda m: f"{model_name.upper()}{m.group(1)}",
            content
        )

        # 2) Replace Model_i → original case
        pattern_normal = re.compile(rf"\bModel_{idx}(\w*)\b")
        matches_normal = pattern_normal.findall(content)
        content = pattern_normal.sub(
            lambda m: f"{model_name}{m.group(1)}",
            content
        )

        print(
            f"[MultiDeeploy] Replaced "
            f"{len(matches_normal)} × Model_{idx} → {model_name}, "
            f"{len(matches_upper)} × MODEL_{idx} → {model_name.upper()}"
        )

    output_path.write_text(content)


def update_dynmultideeploy_test_main(model_names: list[str], configs: List[Dict]):
    """
    Reads dynmultideeploytest_template.c and writes dynmultideeploytest.c (created if
    it doesn't exist yet, overwritten otherwise) with:
      - Model_i  -> model_names[i]  (vestigial: only matches this pair of words inside
        a comment now that the mode-dispatch bodies use literal per-config names, not
        Model_1/Model_2 placeholders -- kept for backward compatibility, harmless)
      - MODEL_i  -> model_names[i].upper()
      - the 3 generated regions (TASK1_HEAVY_LOOKUP, ARENA_FREE_CASES,
        MODE_DISPATCH_CASES) filled in from `configs` -- see ModeSwitchBodyGen.py.
    The template file itself is left untouched.
    """

    template_path = Path(
        "/app/Deeploy/DynMultiDeeployTest/Platforms/DynSiracusa/src/dynmultideeploytest_template.c"
    )
    output_path = Path(
        "/app/Deeploy/DynMultiDeeployTest/Platforms/DynSiracusa/src/dynmultideeploytest.c"
    )

    content = template_path.read_text()
    content = inject_generated_sections(content, configs)

    for i, model_name in enumerate(model_names):
        idx = i + 1

        # 1) Replace MODEL_i → UPPERCASE
        pattern_upper = re.compile(rf"\bMODEL_{idx}(\w*)\b")
        matches_upper = pattern_upper.findall(content)
        content = pattern_upper.sub(
            lambda m: f"{model_name.upper()}{m.group(1)}",
            content
        )

        # 2) Replace Model_i → original case
        pattern_normal = re.compile(rf"\bModel_{idx}(\w*)\b")
        matches_normal = pattern_normal.findall(content)
        content = pattern_normal.sub(
            lambda m: f"{model_name}{m.group(1)}",
            content
        )

        print(
            f"[DynMultiDeeploy] Replaced "
            f"{len(matches_normal)} × Model_{idx} → {model_name}, "
            f"{len(matches_upper)} × MODEL_{idx} → {model_name.upper()}"
        )

    output_path.write_text(content)


def update_dynmultideeploy_singleshot_test_main(model_names: list[str], configs: List[Dict]):
    """
    Sibling of update_dynmultideeploy_test_main(), targeting
    Platforms/DynSiracusaSingleShot/src/dynmultideeploytest_singleshot_template.c
    instead of DynSiracusa's own template -- the per-tick-relaunch workaround for
    the DB repeated-invocation hang (see the plan file's "Step 2 follow-up"
    section). Reuses inject_generated_sections() unmodified: that function only
    requires the same 3 "// ===== GENERATED: ... START/END =====" marker pairs to
    be present, which the single-shot template carries verbatim, so no parallel
    code-generation path is needed -- the exact same per-config Init/Run dispatch
    case bodies apply to both templates.
    """

    template_path = Path(
        "/app/Deeploy/DynMultiDeeployTest/Platforms/DynSiracusaSingleShot/src/dynmultideeploytest_singleshot_template.c"
    )
    output_path = Path(
        "/app/Deeploy/DynMultiDeeployTest/Platforms/DynSiracusaSingleShot/src/dynmultideeploytest_singleshot.c"
    )

    content = template_path.read_text()
    content = inject_generated_sections(content, configs)

    for i, model_name in enumerate(model_names):
        idx = i + 1

        pattern_upper = re.compile(rf"\bMODEL_{idx}(\w*)\b")
        matches_upper = pattern_upper.findall(content)
        content = pattern_upper.sub(
            lambda m: f"{model_name.upper()}{m.group(1)}",
            content
        )

        pattern_normal = re.compile(rf"\bModel_{idx}(\w*)\b")
        matches_normal = pattern_normal.findall(content)
        content = pattern_normal.sub(
            lambda m: f"{model_name}{m.group(1)}",
            content
        )

        print(
            f"[DynMultiDeeploy SingleShot] Replaced "
            f"{len(matches_normal)} × Model_{idx} → {model_name}, "
            f"{len(matches_upper)} × MODEL_{idx} → {model_name.upper()}"
        )

    output_path.write_text(content)
