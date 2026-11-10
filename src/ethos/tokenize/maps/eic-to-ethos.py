import polars as pl
import gzip
import re

# 1. Load the Mapping Files
# -------------------------
print("Loading GEMs mapping...")
map_9_to_10 = pl.read_csv(
    "icd_cm_9_to_10_mapping.csv.gz",
    schema_overrides={"icd_9": pl.String, "icd_10": pl.String}
)

print("Loading ICD-10 descriptions (Flexible Parser)...")
# def load_icd10_descriptions(path):
#     data = []
#     # Regex to find: [5 digits] [space] [The Code] [space] [Header Flag] [Description]
#     # This is more robust than character slicing
#     pattern = re.compile(r"^\d{5}\s+([A-Z0-9]+)\s+[01]\s+(.*)")
#
#     with gzip.open(path, 'rt') as f:
#         for line in f:
#             match = pattern.match(line.strip())
#             if match:
#                 code = match.group(1).strip()
#                 # The description often has a "Short Name" then "Long Name"
#                 # We'll take the whole remaining string and clean it
#                 desc = match.group(2).strip()
#                 data.append({"icd_10": code, "long_desc": desc})
#
#     df = pl.DataFrame(data)
#     # Deduplicate in case a code appears twice (once as header, once as leaf)
#     return df.unique(subset=["icd_10"])

icd10_desc = pl.read_csv("icd10cm-order-Jan-2021.csv.gz")


# 2. Prepare the Target Vocabulary
# -------------------------------
print("Loading target vocabulary...")
vocab = pl.read_csv("/users/gbk2114/data/ethos-tokenize/test/vocab_t4523.csv", has_header=False, new_columns=["path"])

vocab = vocab.with_columns(
    match_key = pl.col("path")
               .str.split("//").list.last()
               .str.to_uppercase()
               .str.replace_all(r"[^A-Z0-9]", "")
)

# 3. Translation Function
# -----------------------
# def translate_codes(input_paths):
#     inputs = pl.DataFrame({"raw_path": input_paths})
#
#     inputs = inputs.with_columns([
#         pl.col("raw_path").str.split("//").list.get(2).alias("version"),
#         pl.col("raw_path").str.split("//").list.get(3).alias("code")
#     ])
#
#     v9 = inputs.filter(pl.col("version") == "9").join(
#         map_9_to_10, left_on="code", right_on="icd_9", how="left"
#     ).select(["raw_path", "icd_10"])
#
#     v10 = inputs.filter(pl.col("version") == "10").with_columns(
#         icd_10 = pl.col("code")
#     ).select(["raw_path", "icd_10"])
#
#     unified = pl.concat([v9, v10]).join(icd10_desc, left_on="icd_10",right_on="code", how="left")
#
#     # Final Match Logic
#     unified = unified.with_columns(
#         # We search the long_desc for the vocabulary string
#         clean_name_key = pl.col("long")
#                         .str.to_uppercase()
#                         .str.replace_all(r"[^A-Z0-9]", "")
#     )
#
#     final = unified.join(
#         vocab, left_on="clean_name_key", right_on="match_key", how="left"
#     )
#
#     return final.select(["raw_path", "icd_10", "long", "path"]).rename({"path":
#                                                                         "target_vocab_path","long":"long_desc"})

def translate_codes(input_paths):
    # ... (Extract version and code as before) ...

    # 1. Use the lab's official name mapping logic
    code_to_name = get_icd_cm_code_to_name_mapping()
    
    # 2. Map your input code to its clinical name
    # If I25118 is missing, we try the prefix I25
    unified = pl.concat([v9, v10]).with_columns(
        long_desc = pl.col("icd_10").replace(code_to_name, default=None)
    )

    # 3. Apply the 'Atherosclerosis' Fallback (The 3-digit rollup)
    unified = unified.with_columns(
        target_name = pl.when(pl.col("long_desc").is_null())
                      .then(pl.col("icd_10").str.slice(0, 3).replace(code_to_name, default=None))
                      .otherwise(pl.col("long_desc"))
    )

    # 4. Normalize for the Vocabulary join
    unified = unified.with_columns(
        match_key = pl.col("target_name").str.to_uppercase().str.replace_all(r"[^A-Z0-9]", "")
    )

    return unified.join(vocab, on="match_key", how="left")

# 4. Execute
# ---------
my_codes = [
    "DIAGNOSIS//ICD//9//3320", 
    "DIAGNOSIS//ICD//10//I25118"
]

results = translate_codes(my_codes)
print("\n--- Translation Results ---")
print(results)
