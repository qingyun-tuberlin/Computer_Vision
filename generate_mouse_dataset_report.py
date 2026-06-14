from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============================================================
# Config - 修改这里即可
# ============================================================
MAIN_CSV = Path("mouse_dataset/MouseGrimaceFaces_main.csv")
MGS_CSV = Path("mouse_dataset/MouseGrimaceFaces_mgs.csv")
OUTPUT_DIR = Path("mouse_dataset/report_output")
PDF_NAME = "mouse_dataset_exploration_report.pdf"

# If you run this script outside your project root, set absolute paths, e.g.:
# MAIN_CSV = Path("/content/drive/MyDrive/.../MouseGrimaceFaces_main.csv")
# MGS_CSV = Path("/content/drive/MyDrive/.../MouseGrimaceFaces_mgs.csv")

# ============================================================
# Helper functions
# ============================================================
def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def df_to_table_data(df: pd.DataFrame, index=True, max_rows=30):
    if index:
        tmp = df.copy()
        tmp.insert(0, tmp.index.name or "index", tmp.index)
    else:
        tmp = df.copy()
    if len(tmp) > max_rows:
        tmp = tmp.head(max_rows)
    return [list(tmp.columns)] + tmp.astype(str).values.tolist()


def add_table(story, df, title=None, index=True, max_rows=30, font_size=8):
    if title:
        story.append(Paragraph(title, STYLES["Heading3"]))
    data = df_to_table_data(df, index=index, max_rows=max_rows)
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.35 * cm))


def add_image(story, path: Path, width_cm=16, max_height_cm=11):
    img = Image(str(path))
    target_width = width_cm * cm
    target_height = img.imageHeight * target_width / img.imageWidth
    max_height = max_height_cm * cm
    if target_height > max_height:
        target_height = max_height
        target_width = img.imageWidth * target_height / img.imageHeight
    img.drawWidth = target_width
    img.drawHeight = target_height
    story.append(img)
    story.append(Spacer(1, 0.35 * cm))


# ============================================================
# Main analysis
# ============================================================
def main():
    ensure_output_dir(OUTPUT_DIR)
    chart_dir = OUTPUT_DIR / "charts"
    ensure_output_dir(chart_dir)

    main_df = pd.read_csv(MAIN_CSV)
    mgs_df = pd.read_csv(MGS_CSV)

    # Basic checks
    labeled_indices = set(mgs_df["index"])
    main_df["has_mgs_label"] = main_df["index"].isin(labeled_indices)

    subset_main = main_df["subset"].value_counts().sort_index()
    subset_mgs = mgs_df["subset"].value_counts().sort_index()

    subset_compare = pd.DataFrame({
        "main_total_images": subset_main,
        "mgs_labeled_images": subset_mgs,
    }).fillna(0).astype(int)
    subset_compare["label_coverage_ratio"] = (
        subset_compare["mgs_labeled_images"] / subset_compare["main_total_images"]
    )

    mouse_identity = main_df.groupby("subset")["id"].nunique().sort_index().to_frame("n_mice")

    images_per_mouse = (
        main_df.groupby(["subset", "id"])
        .size()
        .reset_index(name="n_images")
        .sort_values(["subset", "id"])
    )
    images_per_mouse_sorted = images_per_mouse.sort_values("n_images", ascending=False)
    images_per_mouse_desc = images_per_mouse["n_images"].describe().to_frame("n_images")

    label_cols = [c for c in mgs_df.columns if c not in ["index", "subset"]]
    raw_label_long = mgs_df[label_cols].stack().astype(str)
    raw_label_long = raw_label_long.replace({"nan": "NaN"})
    raw_label_counts = raw_label_long.value_counts().rename_axis("raw_value").to_frame("count")

    # For label columns, normalize scores to categories
    score_numeric = mgs_df[label_cols].apply(pd.to_numeric, errors="coerce")
    valid_score_mask = score_numeric.isin([0, 1, 2])
    invalid_9_mask = score_numeric.eq(9)
    normalized_label_summary = pd.DataFrame({
        "valid_0_1_2": valid_score_mask.sum(),
        "invalid_9": invalid_9_mask.sum(),
        "missing_or_dash_or_other": (~valid_score_mask & ~invalid_9_mask).sum(),
    })

    # Save detailed tables as CSV for inspection
    subset_compare.to_csv(OUTPUT_DIR / "subset_compare.csv")
    mouse_identity.to_csv(OUTPUT_DIR / "mouse_identity_by_subset.csv")
    images_per_mouse.to_csv(OUTPUT_DIR / "images_per_mouse.csv", index=False)
    images_per_mouse_sorted.to_csv(OUTPUT_DIR / "images_per_mouse_sorted.csv", index=False)
    raw_label_counts.to_csv(OUTPUT_DIR / "raw_label_value_counts.csv")
    normalized_label_summary.to_csv(OUTPUT_DIR / "label_column_summary.csv")

    # ========================================================
    # Charts
    # ========================================================
    chart_paths = {}

    # 1. subset imbalance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(subset_compare.index))
    width = 0.38
    ax.bar(x - width / 2, subset_compare["main_total_images"], width, label="main.csv: all images")
    ax.bar(x + width / 2, subset_compare["mgs_labeled_images"], width, label="mgs.csv: labeled images")
    ax.set_xticks(x)
    ax.set_xticklabels(subset_compare.index)
    ax.set_ylabel("Number of images")
    ax.set_title("Subset imbalance: all images vs labeled images")
    ax.legend()
    path = chart_dir / "subset_imbalance.png"
    save_fig(path)
    chart_paths["subset_imbalance"] = path

    # 2. coverage ratio
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(subset_compare.index, subset_compare["label_coverage_ratio"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("MGS label coverage")
    ax.set_title("Label coverage by subset")
    path = chart_dir / "label_coverage_by_subset.png"
    save_fig(path)
    chart_paths["coverage"] = path

    # 3. mouse identities by subset
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(mouse_identity.index, mouse_identity["n_mice"])
    ax.set_ylabel("Number of unique mice")
    ax.set_title("Mouse identity distribution by subset")
    path = chart_dir / "mouse_identity_distribution.png"
    save_fig(path)
    chart_paths["mouse_identity"] = path

    # 4. images per mouse histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(images_per_mouse["n_images"], bins=40)
    ax.set_xlabel("Images per mouse")
    ax.set_ylabel("Number of mice")
    ax.set_title("Distribution of images per mouse")
    path = chart_dir / "images_per_mouse_histogram.png"
    save_fig(path)
    chart_paths["mouse_hist"] = path

    # 5. top 30 mice
    top30 = images_per_mouse_sorted.head(30).copy()
    top30["mouse_key"] = top30["subset"].astype(str) + "_" + top30["id"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(top30["mouse_key"], top30["n_images"])
    ax.set_xticks(range(len(top30["mouse_key"])))
    ax.set_xticklabels(top30["mouse_key"], rotation=75, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Top 30 mice by number of images")
    path = chart_dir / "top30_mice_by_images.png"
    save_fig(path)
    chart_paths["top30"] = path

    # 6. has label counts
    label_count = main_df["has_mgs_label"].value_counts().rename(index={True: "Labeled", False: "Not labeled"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(label_count.index.astype(str), label_count.values)
    ax.set_ylabel("Number of images")
    ax.set_title("Overall MGS label availability")
    path = chart_dir / "overall_label_availability.png"
    save_fig(path)
    chart_paths["overall_label"] = path

    # 7. raw label values aggregated
    fig, ax = plt.subplots(figsize=(8, 4.5))
    raw_label_counts.head(10)["count"].plot(kind="bar", ax=ax)
    ax.set_xlabel("Raw value in MGS score cells")
    ax.set_ylabel("Count across all score columns")
    ax.set_title("Raw MGS score-cell values, aggregated")
    path = chart_dir / "raw_label_values.png"
    save_fig(path)
    chart_paths["raw_label_values"] = path

    # 8. valid / invalid / missing per first 10 label columns
    first10 = normalized_label_summary.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    first10.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Label column")
    ax.set_ylabel("Number of images")
    ax.set_title("Score status in first 10 MGS columns")
    ax.legend(loc="upper right")
    path = chart_dir / "first10_label_column_summary.png"
    save_fig(path)
    chart_paths["first10_label_summary"] = path

    # ========================================================
    # PDF report
    # ========================================================
    pdf_path = OUTPUT_DIR / PDF_NAME
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm,
    )

    story = []
    story.append(Paragraph("Mouse Grimace Faces - Dataset Exploration Report", STYLES["Title"]))
    story.append(Spacer(1, 0.35 * cm))

    intro = (
        f"This report summarizes main.csv and mgs.csv. "
        f"main.csv contains {len(main_df):,} image rows and {main_df.shape[1]} columns. "
        f"mgs.csv contains {len(mgs_df):,} labeled image rows and {mgs_df.shape[1]} columns. "
        f"A mouse identity should be interpreted as subset + id, because the same id value can occur in different subsets."
    )
    story.append(Paragraph(intro, STYLES["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    overview_df = pd.DataFrame({
        "metric": [
            "Rows in main.csv",
            "Rows in mgs.csv",
            "Columns in main.csv",
            "Columns in mgs.csv",
            "Unique subsets",
            "Unique mouse identities (subset + id)",
            "MGS-labeled images",
            "Unlabeled images",
            "Overall label coverage",
        ],
        "value": [
            f"{len(main_df):,}",
            f"{len(mgs_df):,}",
            f"{main_df.shape[1]}",
            f"{mgs_df.shape[1]}",
            f"{main_df['subset'].nunique()}",
            f"{len(images_per_mouse):,}",
            f"{main_df['has_mgs_label'].sum():,}",
            f"{(~main_df['has_mgs_label']).sum():,}",
            f"{main_df['has_mgs_label'].mean():.2%}",
        ]
    })
    add_table(story, overview_df, "1. Dataset overview", index=False, max_rows=20, font_size=8)

    add_table(story, subset_compare.assign(label_coverage_ratio=subset_compare["label_coverage_ratio"].map(lambda x: f"{x:.2%}")),
              "2. Subset imbalance and label coverage", index=True, max_rows=10, font_size=8)
    add_image(story, chart_paths["subset_imbalance"])
    add_image(story, chart_paths["coverage"])

    story.append(PageBreak())
    add_table(story, mouse_identity, "3. Mouse identity distribution", index=True, max_rows=20, font_size=8)
    add_image(story, chart_paths["mouse_identity"])

    desc_for_pdf = images_per_mouse_desc.copy()
    desc_for_pdf["n_images"] = desc_for_pdf["n_images"].map(lambda x: f"{x:.2f}")
    add_table(story, desc_for_pdf, "4. Images per mouse - summary", index=True, max_rows=20, font_size=8)
    add_image(story, chart_paths["mouse_hist"])
    add_image(story, chart_paths["top30"], width_cm=17, max_height_cm=10)

    story.append(PageBreak())
    story.append(Paragraph("5. Images per mouse - detailed table", STYLES["Heading2"]))
    story.append(Paragraph(
        "The full table is saved as images_per_mouse.csv. The PDF shows the first 60 rows sorted by subset and id.",
        STYLES["BodyText"]
    ))
    add_table(story, images_per_mouse.head(60), index=False, max_rows=60, font_size=7)

    story.append(PageBreak())
    story.append(Paragraph("6. MGS label values", STYLES["Heading2"]))
    add_image(story, chart_paths["overall_label"])
    add_table(story, raw_label_counts.head(12), "Raw label-cell values across all MGS score columns", index=True, max_rows=12, font_size=8)
    add_image(story, chart_paths["raw_label_values"])
    add_image(story, chart_paths["first10_label_summary"], width_cm=17, max_height_cm=10)

    story.append(PageBreak())
    story.append(Paragraph("7. Interpretation", STYLES["Heading2"]))
    bullets = [
        f"Only {main_df['has_mgs_label'].mean():.2%} of all images have MGS labels. For supervised pain classification, the effective labeled dataset is mgs.csv, not the full main.csv.",
        "KH dominates the full image pool, but its label coverage is low. MR and LW have 100% coverage.",
        "The number of images per mouse is highly imbalanced. This creates a risk that a model learns mouse identity or recording condition rather than pain-related visual cues.",
        "For train/validation splitting, use GroupShuffleSplit or GroupKFold with groups = subset + '_' + id to reduce identity leakage.",
        "For the final report, include subset distribution, label coverage, images-per-mouse distribution, and qualitative examples such as saliency maps or attention visualization.",
    ]
    for b in bullets:
        story.append(Paragraph("- " + b, STYLES["BodyText"]))
        story.append(Spacer(1, 0.18 * cm))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Generated files", STYLES["Heading3"]))
    generated = pd.DataFrame({
        "file": [
            "mouse_dataset_exploration_report.pdf",
            "images_per_mouse.csv",
            "images_per_mouse_sorted.csv",
            "subset_compare.csv",
            "mouse_identity_by_subset.csv",
            "raw_label_value_counts.csv",
            "label_column_summary.csv",
            "charts/*.png",
        ]
    })
    add_table(story, generated, index=False, max_rows=20, font_size=8)

    doc.build(story)
    print(f"Saved PDF report to: {pdf_path}")
    print(f"Saved tables and charts to: {OUTPUT_DIR}")


# Basic ReportLab styles
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(
    name="SmallText",
    parent=STYLES["BodyText"],
    fontSize=8,
    leading=10,
))


if __name__ == "__main__":
    main()
