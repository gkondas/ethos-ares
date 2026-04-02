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
        The input DataFrame with an added ``ethos_tokens`` list column containing
        the multi-token representation (category, 3-6, suffix) matching the vocab
        format.  Rows whose 3-char prefix has no mapping get an empty list.
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
        # Part 1: 3-char prefix → description (matches pipeline process_icd10)
        .with_columns(
            pl.col("_icd_code")
            .str.slice(0, 3)
            .replace_strict(code_to_name, default=None)
            .alias("_part1"),
        )
        # Part 2: chars 3-6 of the code
        .with_columns(
            pl.col("_icd_code").str.slice(3, 3).alias("_part2"),
        )
        # Part 3: chars 6+ (suffix)
        .with_columns(
            pl.col("_icd_code").str.slice(6).alias("_part3"),
        )
        # Build the multi-token output
        .with_columns(
            pl.when(pl.col("_part1").is_not_null())
            .then(_normalize_name(pl.lit("ICD//CM//") + pl.col("_part1")))
            .alias("_tok1"),
            pl.when(pl.col("_part2") != "")
            .then(_normalize_name(pl.lit("ICD//CM//3-6//") + pl.col("_part2")))
            .alias("_tok2"),
            pl.when(pl.col("_part3") != "")
            .then(_normalize_name(pl.lit("ICD//CM//SFX//") + pl.col("_part3")))
            .alias("_tok3"),
        )
        .with_columns(
            pl.concat_list("_tok1", "_tok2", "_tok3").list.drop_nulls().alias("ethos_tokens"),
        )
        .drop("_icd_code", "_part1", "_part2", "_part3", "_tok1", "_tok2", "_tok3")
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
        tokens = row["ethos_tokens"]
        print(f"{code:15s} -> {tokens}")
