import numpy as np
import pandas as pd
from pathlib import Path
import shutil

def build_labels(MGS_CSV,MAIN_CSV,IMG_DIR):
    """
    Input: 
        MGS_CSV: Path of MGS_CSV
        MAIN_CSV: Path of MAIN_CSV
        IMG_DIR: Path of IMG_DIR

        
    Build binary pain labels from Mouse Grimace Scale (MGS) annotations.

    This function loads MGS annotations and image metadata, cleans invalid
    scores, computes the mean MGS score for each image, and generates a
    binary classification label:

        label = 0 : non-impaired / normal well-being
        label = 1 : impaired / pain-like expression

    Label generation rule:
        mgs_mean < 1.0   -> label = 0
        mgs_mean >= 1.0  -> label = 1

    Processing steps:
        1. Load MGS and metadata CSV files.
        2. Merge image IDs into the MGS annotations.
        3. Convert score columns to numeric values.
        4. Treat '-' and score value 9 as missing values.
        5. Compute the mean MGS score across all available annotations.
        6. Remove samples without valid scores.
        7. Generate binary labels based on the mean MGS score.
        8. Construct image file paths.
        9. Remove entries whose image files do not exist.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:

        - index : image filename
        - subset : dataset subset identifier
        - id : mouse identifier
        - path : image file path
        - mgs_mean : average MGS score
        - label : binary classification target

    Notes
    -----
    MGS score interpretation:

        0 = absent
        1 = moderate
        2 = obvious
        9 = cannot be judged

    Images with only invalid or missing scores are discarded.
    """    

    df = pd.read_csv(MGS_CSV)
    main_df = pd.read_csv(MAIN_CSV)

    # merge id information
    df = df.merge(
        main_df[["index","id"]],
        on="index",
        how="left"
    )

    meta_cols = ["index", "subset", "id"]
    score_cols = [c for c in df.columns if c not in meta_cols]

    scores = df[score_cols].replace("-", np.nan)
    scores = scores.apply(pd.to_numeric, errors="coerce")

    # 9 means "cannot be judged"
    scores = scores.replace(9, np.nan)

    # Average all available MGS scores
    df["mgs_mean"] = scores.mean(axis=1)

    # Remove images without valid scores
    df = df.dropna(subset=["mgs_mean"])

    # Binary label:
    # 0 = well-being / not impaired
    # 1 = impaired / pain-like expression
    df["label"] = (df["mgs_mean"] >= 1.0).astype(int)

    # Build image path
    df["path"] = df["index"].apply(lambda x: IMG_DIR / str(x))
 
    before = len(df)

    # Keep only existing images, 过滤不存在的图片
    df = df[df["path"].apply(lambda p: p.exists())].copy()
    
    after = len(df)
    print(f"Rows before crop filtering: {before}")
    print(f"Rows after crop filtering:  {after}")
    print(f"Removed because crop image missing: {before - after}")

    return df[["index", "subset", "id", 
               "path", "mgs_mean", "label"]].reset_index(drop=True)


def copy_impaired_images(
    df,
    output_dir
):
    """
    Copy all images with label == 1 (impaired) into output_dir.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from build_labels()

    output_dir : str or Path
        Target folder
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    impaired_df = df[df["label"] == 1]

    copied = 0

    for _, row in impaired_df.iterrows():

        src = Path(row["path"])

        if src.exists():

            dst = output_dir / src.name

            shutil.copy2(src, dst)

            copied += 1

    print(f"Copied {copied} impaired images to:")
    print(output_dir)



