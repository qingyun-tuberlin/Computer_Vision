# valid images must satisfy the following constraints:
# 1) it has mgs score, at least one valid score.
# 2) among the mgs score, 
#   if more than 30% score is 9, then filter it away. because 9 is not valid.
# after filter the images
# for the left over images, replace the 9 with -

# store the final result:  usable_csv_path = DATASET_DIR / "usable_mgs_images.csv"
# if it already exists , you can overwrite it.

from pathlib import Path
import shutil
import pandas as pd

# -------------------------
# Paths
# -------------------------
DATASET_DIR = Path("mouse_dataset")

IMAGE_DIR = DATASET_DIR / "images"
MGS_CSV = DATASET_DIR / "MouseGrimaceFaces_mgs.csv"

OUTPUT_DIR = DATASET_DIR / "images_mgs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load MGS csv
# -------------------------
mgs_df = pd.read_csv(MGS_CSV)

print("Original MGS rows:", len(mgs_df))
print(mgs_df.head())

# -------------------------
# Score columns
# -------------------------
score_cols = [
    col for col in mgs_df.columns
    if col not in ["index", "subset"]
]

# Convert scores to numeric
# valid scores: 0, 1, 2
# invalid score: 9
# missing/no score: "-", NaN, empty cells
scores = mgs_df[score_cols].apply(
    pd.to_numeric,
    errors="coerce"
)

# -------------------------
# Filter usable images
# Constraints:
# 1) at least one valid MGS score: 0, 1, or 2
# 2) among existing MGS scores, if more than 30% are 9, remove the image
# -------------------------

has_valid_score = scores.isin([0, 1, 2]).any(axis=1)

has_score_or_9 = scores.isin([0, 1, 2, 9])
num_scored_entries = has_score_or_9.sum(axis=1)

num_9 = scores.eq(9).sum(axis=1)

ratio_9 = num_9 / num_scored_entries

has_acceptable_9_ratio = ratio_9 <= 0.30

usable_df = mgs_df[
    has_valid_score & has_acceptable_9_ratio
].copy()

print("Usable MGS rows:", len(usable_df))
print("Removed rows:", len(mgs_df) - len(usable_df))

# -------------------------
# Replace 9 with "-" in leftover images
# -------------------------
usable_df[score_cols] = usable_df[score_cols].replace(9, "-")
usable_df[score_cols] = usable_df[score_cols].replace("9", "-")

# -------------------------
# Copy usable images
# -------------------------
copied = 0
missing_files = []

for filename in usable_df["index"]:
    src = IMAGE_DIR / filename
    dst = OUTPUT_DIR / filename

    if src.exists():
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing_files.append(filename)

print("Copied images:", copied)
print("Missing image files:", len(missing_files))

if missing_files:
    print("First missing files:")
    print(missing_files[:10])

# -------------------------
# Save usable image list
# overwrite if already exists
# -------------------------
usable_csv_path = DATASET_DIR / "usable_mgs_images.csv"
usable_df.to_csv(usable_csv_path, index=False)

print("Saved usable image list to:", usable_csv_path)
print("Copied images stored under:", OUTPUT_DIR)
