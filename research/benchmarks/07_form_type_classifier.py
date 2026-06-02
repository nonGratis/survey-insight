"""07_form_type_classifier.py — keyword-based form-type classifier.

Парсимо `data/Form Catalog.tsv`, класифікуємо кожну форму у одну з категорій.

Keyword-rules живуть у `core/forecast/form_type.py` (single source — прод-UI
класифікує там же). Цей скрипт лише застосовує їх до каталогу й пише CSV.

Output: `research/reports/figures/07_form_types.csv` (form_id, form_title, form_type).

Запуск:
    .venv/Scripts/python.exe research/benchmarks/07_form_type_classifier.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast.form_type import classify_form_type  # noqa: E402


def main(catalog_path: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(catalog_path, sep="\t", dtype=str).fillna("")
    df = df[["form_id", "form_title", "short_name", "description"]].copy()
    df = df[df["form_id"].str.strip() != ""].reset_index(drop=True)

    results = []
    for _, row in df.iterrows():
        cat = classify_form_type(row["form_title"], row["short_name"], row["description"])
        results.append(
            {
                "form_id": row["form_id"],
                "form_title": row["form_title"],
                "form_type": cat,
            }
        )
    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"Classified {len(out)} forms -> {output_csv}")
    print("\nDistribution:")
    print(out["form_type"].value_counts().to_string())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument("--catalog", type=Path, default=repo_root / "data" / "Form Catalog.tsv")
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "figures" / "07_form_types.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.catalog, args.output)
