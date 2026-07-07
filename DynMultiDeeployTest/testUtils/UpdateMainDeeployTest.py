import sys
import re
from pathlib import Path

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


def update_dynmultideeploy_test_main(model_names: list[str]):
    """
    Reads dynmultideeploytest_template.c and writes dynmultideeploytest.c (created if
    it doesn't exist yet, overwritten otherwise) with:
      - Model_i  -> model_names[i]
      - MODEL_i  -> model_names[i].upper()
    The template file itself is left untouched.
    """

    template_path = Path(
        "/app/Deeploy/DynMultiDeeployTest/Platforms/DynSiracusa/src/dynmultideeploytest_template.c"
    )
    output_path = Path(
        "/app/Deeploy/DynMultiDeeployTest/Platforms/DynSiracusa/src/dynmultideeploytest.c"
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
            f"[DynMultiDeeploy] Replaced "
            f"{len(matches_normal)} × Model_{idx} → {model_name}, "
            f"{len(matches_upper)} × MODEL_{idx} → {model_name.upper()}"
        )

    output_path.write_text(content)
