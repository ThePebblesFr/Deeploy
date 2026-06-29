#!/usr/bin/env python3
import re
import csv
import argparse
from pathlib import Path

FILENAME_RE = re.compile(r"cores(?P<cores>\d+)_l1(?P<l1>\d+)", re.IGNORECASE)

DELTA_RE = re.compile(r"Runtime\s+RunNetwork:\s+(\d+)\s+cycles")
PHI_RE = re.compile(r"@power\.measure_0@([0-9]*\.?[0-9]+)@")
ERRORS_RE = re.compile(r"Errors:\s*(\d+)\s*out of\s*(\d+)", re.IGNORECASE)

DIR_MODEL_RE = re.compile(r"^gvsoc_log_(.+)$", re.IGNORECASE)

FREQUENCY_HZ = 50_000_000


def infer_model_name(log_dir: Path, user_model):
    if user_model:
        return user_model

    match = DIR_MODEL_RE.match(log_dir.name)
    if match:
        return match.group(1)

    return "unknown_model"


def compute_energy_j(delta_cycles, avg_phi_w, frequency_hz):
    if delta_cycles in ("", None) or avg_phi_w in ("", None):
        return ""

    try:
        delta_cycles = float(delta_cycles)
        avg_phi_w = float(avg_phi_w)
        return avg_phi_w * (delta_cycles / frequency_hz)
    except (ValueError, TypeError):
        return ""


def parse_log_file(path: Path, model: str):
    match = FILENAME_RE.search(path.stem)
    if not match:
        return None

    cores = int(match.group("cores"))
    l1_bytes = int(match.group("l1"))
    l1_kb = l1_bytes // 1024

    text = path.read_text(errors="ignore").strip()

    if text == "gvsoc failed":
        return {
            "Model": model,
            "N_c": cores,
            "s_L1 (kB)": l1_kb,
            "delta (cycles)": "",
            "avg phi (W)": "",
            "Frequency (Hz)": FREQUENCY_HZ,
            "Energy Consumption (J)": "",
            "result correctness": "0",
        }

    delta_match = DELTA_RE.search(text)
    phi_match = PHI_RE.search(text)
    errors_match = ERRORS_RE.search(text)

    delta = delta_match.group(1) if delta_match else ""
    phi = phi_match.group(1) if phi_match else ""

    correctness = "0"
    if errors_match:
        n_errors = int(errors_match.group(1))
        total = int(errors_match.group(2))
        correctness = "1" if n_errors == 0 and total > 0 else "0"

    energy_j = compute_energy_j(delta, phi, FREQUENCY_HZ)

    return {
        "Model": model,
        "N_c": cores,
        "s_L1 (kB)": l1_kb,
        "delta (cycles)": delta,
        "avg phi (W)": phi,
        "Frequency (Hz)": FREQUENCY_HZ,
        "Energy Consumption (J)": energy_j,
        "result correctness": correctness,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", nargs="?", default=".")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    model = infer_model_name(log_dir, args.model)

    output_path = (
        log_dir / f"results_{model}.csv"
        if args.output is None
        else Path(args.output)
    )

    rows = []

    for log_file in sorted(log_dir.glob("*.log")):
        row = parse_log_file(log_file, model)
        if row:
            rows.append(row)

    rows.sort(key=lambda r: (-r["s_L1 (kB)"], -r["N_c"]))

    fieldnames = [
        "Model",
        "N_c",
        "s_L1 (kB)",
        "delta (cycles)",
        "avg phi (W)",
        "Frequency (Hz)",
        "Energy Consumption (J)",
        "result correctness",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()