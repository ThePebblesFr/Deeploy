from typing import List, Tuple
from pathlib import Path
import re

# -----------------------------------------------------------------------------------------#
#                         Functions for fusing multiple Network.h                          #
# -----------------------------------------------------------------------------------------#

def split_Network_h(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a header file into 3 parts:
      1) from start to and including the line with InitNetwork
      2) from after InitNetwork line to before the last #endif
      3) from the last #endif to end of file
    """
    lines = text.splitlines(keepends=True)

    init_idx = None
    endif_idx = None

    for i, line in enumerate(lines):
        if init_idx is None and "InitNetwork" in line:
            init_idx = i
        if "#endif" in line:
            endif_idx = i

    first = lines[: init_idx + 1]
    second = lines[init_idx + 1 : endif_idx]
    third = lines[endif_idx:]

    return first, second, third

def Network_h_fusion(files: List[Path], output_path: Path) -> None:
    """
    Fuses multiple Network.h files into a single Network.h file.
    Args:
        files: List of paths to Network.h files to be fused.
        output_path: Path to the output fused Network.h file.
    Returns:
        The content of the fused Network.h file as a string.
    """
    merged_first: List[str] = []
    seen_first = set()

    all_second: List[str] = []
    endif_line: str | None = None

    for idx, path in enumerate(files):
        text = path.read_text()
        first, second, third = split_Network_h(text)

        for line in first:
            if line not in seen_first:
                merged_first.append(line)
                seen_first.add(line)

        if second:
            if all_second and all_second[-1].strip() != "":
                all_second.append("\n")
            all_second.extend(second)

        if endif_line is None:
            for line in third:
                if "#endif" in line:
                    endif_line = line
                    break
    
    out_lines: List[str] = []
    out_lines.extend(merged_first)
    if out_lines and not out_lines[-1].endswith("\n"):
        out_lines[-1] += "\n"

    if all_second:
        if not out_lines[-1].strip() == "":
            out_lines.append("\n")

    out_lines.extend(all_second)
    if out_lines and not out_lines[-1].endswith("\n"):
        out_lines[-1] += "\n"

    if not endif_line.endswith("\n"):
        endif_line += "\n"
    out_lines.append(endif_line)

    with open(output_path, 'w') as f:
        f.write("".join(out_lines))

# -----------------------------------------------------------------------------------------#
#                       Functions for fusing multiple testinputs.h                         #
# -----------------------------------------------------------------------------------------#

def process_testinputs_file(path: Path, model_name: str) -> str:
    """
    Takes one testInputs.h file and a model name,
    returns rewritten content with renamed variables.
    """

    RE_ARRAY = re.compile(
        r"("                       # group 1: full type prefix (kept)
        r"(?:const\s+)?"
        r"(?:u?int(?:8|16|32|64)_t)\s+"
        r")"
        r"(\w+)"                   # group 2: var name
        r"(\s*\[\s*\]\s*=\s*\{.*?\}\s*;)",  # group 3: [] = { ... };
        re.DOTALL
    )

    RE_PTR = re.compile(
        r"(void\s*\*\s*)"          # group 1: 'void *'
        r"(\w+)"                   # group 2: var name
        r"\s*(\[[^\]]*\])"         # group 3: [N] or []
        r"\s*=\s*"
        r"(\{.*?\})"               # group 4: initializer {...}
        r"\s*;",
        re.DOTALL
    )

    text = path.read_text()
    out = []

    for prefix, varname, suffix in RE_ARRAY.findall(text):
        out.append(f"{prefix}{model_name}_{varname}{suffix}\n")
    for prefix, varname, arr_size, initializer in RE_PTR.findall(text):
        new_var = f"{model_name}_{varname}"
        new_initializer = re.sub(
            r"\b(testInputVector\w*)\b",
            lambda m: f"{model_name}_{m.group(1)}",
            initializer
        )
        out.append(f"{prefix}{new_var}{arr_size} = {new_initializer};\n")

    return "".join(out)

def testinputs_h_fusion(files: List[Path], model_names: List[str], output_path: Path):
    """
    Fuses multiple testInputs.h files into one.
    Each model_name corresponds to each file in order.
    """
    assert len(files) == len(model_names), "One model name per testInputs file is required."

    fused = []

    for file, model_name in zip(files, model_names):
        fused.append(process_testinputs_file(file, model_name))

    output_path.write_text("\n".join(fused))

# -----------------------------------------------------------------------------------------#
#                       Functions for fusing multiple testoutputs.h                        #
# -----------------------------------------------------------------------------------------#

def process_testoutputs_file(path: Path, model_name: str) -> str:
    RE_DEFINE = re.compile(
        r"#define\s+(\w+)\s+(.+)$",
        re.MULTILINE
    )
    RE_ARRAY = re.compile(
        r"^\s*(?!#)((?:\w+\s+)+)(\w*testOutputVector\w*)\s*(\[\s*\]\s*=\s*\{.*?\};)",
        re.DOTALL | re.MULTILINE
    )
    RE_PTR = re.compile(
        r"(void\s*\*\s*)(\w+)\s*(\[[^\]]*\])\s*=\s*(\{.*?\});",
        re.DOTALL
    )

    text = path.read_text()
    out = []

    UPPER = model_name.upper()

    for name, value in RE_DEFINE.findall(text):
        out.append(f"#define {UPPER}_{name} {value}\n")

    for prefix, varname, suffix in RE_ARRAY.findall(text):
        out.append(f"{prefix}{model_name}_{varname}{suffix}\n\n")

    for prefix, varname, arr_size, initializer in RE_PTR.findall(text):

        new_var = f"{model_name}_{varname}"
        new_initializer = re.sub(
            r"\b(testOutputVector\w*)\b",
            lambda m: f"{model_name}_{m.group(1)}",
            initializer
        )

        out.append(f"{prefix}{new_var}{arr_size} = {new_initializer};\n\n")

    return "".join(out)

def testoutputs_h_fusion(files: List[Path], model_names: List[str], output_path: Path):
    """
    Fuses multiple testOutputs.h files into one.
    Each model_name corresponds to each file in order.
    """
    assert len(files) == len(model_names), "One model name per testOutputs file is required."

    fused = []

    for file, model_name in zip(files, model_names):
        fused.append(process_testoutputs_file(file, model_name))

    output_path.write_text("\n".join(fused))

