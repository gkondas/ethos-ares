"""Standalone ICD-to-ETHOS translation tool.

Translates ICD-9 and ICD-10 codes into ETHOS vocabulary tokens. ICD-9 codes are
converted to ICD-10 first. Uses exact code match, then falls back to the 3-digit
ICD prefix for codes without an exact entry in the mapping.

Usage:
    python scripts/icd_to_ethos.py codes.yaml

Where codes.yaml contains raw_path entries:

    codes:
      - DIAGNOSIS//ICD//10//I25118
      - DIAGNOSIS//ICD//9//5856
      - DIAGNOSIS//ICD//10//E109
"""

import sys
from pathlib import Path

import polars as pl
import yaml

from ethos.tokenize.mappings import get_icd_cm_9_to_10_mapping, get_icd_cm_code_to_name_mapping


def _normalize_name(col: pl.Expr) -> pl.Expr:
    """Apply the same normalization as unify_code_names: uppercase, strip commas/periods, spaces→underscores."""
    return col.str.to_uppercase().str.replace_all(r"[,.]", "").str.replace_all(" ", "_", literal=True)


def translate_icd_to_ethos(df: pl.DataFrame) -> pl.DataFrame:
    """Translate raw_path ICD codes to ETHOS tokens with 3-digit prefix fallback.

    Args:
        df: DataFrame with a ``raw_path`` column whose values look like
            ``DIAGNOSIS//ICD//10//I25118`` or ``DIAGNOSIS//ICD//9//5856``.
            ICD-9 codes are converted to ICD-10 first via the repo mapping.

    Returns:
        The input DataFrame with an added ``ethos_token`` column.  Rows whose
        ICD code cannot be resolved at either the exact or 3-digit level get a
        null ``ethos_token``.
    """
    icd9_to_10 = get_icd_cm_9_to_10_mapping()
    code_to_name = get_icd_cm_code_to_name_mapping()

    return (
        df
        # Extract ICD version and code from raw_path
        .with_columns(
            pl.col("raw_path").str.split("//").list[2].alias("_icd_ver"),
            pl.col("raw_path").str.split("//").list[3].alias("_icd_code"),
        )
        # Convert ICD-9 to ICD-10; leave ICD-10 codes as-is
        .with_columns(
            pl.when(pl.col("_icd_ver") == "9")
            .then(pl.col("_icd_code").replace_strict(icd9_to_10, default=None))
            .otherwise(pl.col("_icd_code"))
            .alias("_icd_code"),
        )
        .drop("_icd_ver")
        # Exact match: look up full code
        .with_columns(
            pl.col("_icd_code")
            .replace_strict(code_to_name, default=None)
            .alias("_exact_name"),
        )
        # 3-digit prefix fallback
        .with_columns(
            pl.col("_icd_code")
            .str.slice(0, 3)
            .replace_strict(code_to_name, default=None)
            .alias("_fallback_name"),
        )
        # Resolve: prefer exact, fall back to prefix
        .with_columns(
            pl.coalesce("_exact_name", "_fallback_name").alias("_resolved_name"),
            pl.when(pl.col("_exact_name").is_not_null())
            .then(pl.col("_icd_code"))
            .otherwise(pl.col("_icd_code").str.slice(0, 3))
            .alias("_resolved_prefix"),
        )
        # Build and normalize the ETHOS token
        .with_columns(
            pl.when(pl.col("_resolved_name").is_not_null())
            .then(
                _normalize_name(
                    pl.lit("ICD//CM//")
                    + pl.col("_resolved_prefix")
                    + pl.lit("//")
                    + pl.col("_resolved_name")
                )
            )
            .alias("ethos_token"),
        )
        .drop("_icd_code", "_exact_name", "_fallback_name", "_resolved_name", "_resolved_prefix")
    )


_DEFAULT_YAML = Path(__file__).parent / "icd_codes.yaml"


if __name__ == "__main__":
    yaml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_YAML

    with open(yaml_path) as f:
        codes = yaml.safe_load(f)["codes"]

    df = pl.DataFrame({"raw_path": codes})
    result = translate_icd_to_ethos(df)

    for row in result.iter_rows(named=True):
        code = row["raw_path"].split("//")[-1]
        token = row["ethos_token"]
        print(f"{code:15s} -> {token}")
