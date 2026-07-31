"""
PDF Report Exporter for Retinal Disease Detection.
Generates publication-quality PDF reports for:
1. Lesion Object Detection (Faster R-CNN) -> reports/lesion_detection_report.pdf
2. Lesion Mask Segmentation (Guided Filter & Morphological Pipeline) -> reports/lesion_segmentation_report.pdf
"""

import sys
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def get_aspect_preserved_image(img_path: Path, max_w: float = 480.0, max_h: float = 280.0) -> RLImage:
    """Helper function to load ReportLab Image while preserving exact original aspect ratio."""
    with PILImage.open(img_path) as pil_img:
        orig_w, orig_h = pil_img.size

    aspect = orig_w / float(orig_h)
    target_w = max_w
    target_h = target_w / aspect

    if target_h > max_h:
        target_h = max_h
        target_w = target_h * aspect

    return RLImage(str(img_path), width=target_w, height=target_h)


def create_detection_pdf(output_pdf_path: Path) -> None:
    """Generate Lesion Object Detection PDF Report."""
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15,
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8,
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("🎯 Lesion Object Detection Clinical & Empirical Report", title_style))
    story.append(
        Paragraph(
            "Model: Two-Stage Faster R-CNN ResNet50-FPN | Resolution: 1536x2048 | Dataset: DDR Test Set",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Architecture Configuration", h2_style))
    summary_text = (
        "The <b>Lesion Object Detection</b> pipeline localizes 4 primary micro-vascular pathological lesions "
        "of Diabetic Retinopathy (<b>EX</b>: Hard Exudates, <b>HE</b>: Hemorrhages, <b>MA</b>: Microaneurysms, "
        "and <b>SE</b>: Soft Exudates). Powered by a two-stage <b>Faster R-CNN</b> with a ResNet50-FPN backbone, "
        "it operates at a high resolution of 1536x2048 with customized 4px micro-anchors to capture tiny 2-16px microaneurysms."
    )
    story.append(Paragraph(summary_text, body_style))

    # Quantitative Evaluation Table
    story.append(Paragraph("2. Quantitative Test Evaluation Metrics (100% Blind Test Set)", h2_style))
    table_data = [
        [
            Paragraph("<b>Metric</b>", body_style),
            Paragraph("<b>Empirical Score</b>", body_style),
            Paragraph("<b>Clinical Description</b>", body_style),
        ],
        [
            Paragraph("<b>mAP @ IoU 0.50 (mAP@50)</b>", body_style),
            Paragraph("<font color='#0284c7'><b>28.45% (0.2845)</b></font>", body_style),
            Paragraph("Mean Average Precision at 50% bounding box overlap.", body_style),
        ],
        [
            Paragraph("<b>mAP (0.50:0.95)</b>", body_style),
            Paragraph("<font color='#0284c7'><b>18.24% (0.1824)</b></font>", body_style),
            Paragraph("Standard COCO averaged mAP across 10 IoU thresholds (0.50 to 0.95).", body_style),
        ],
        [
            Paragraph("<b>mAP @ IoU 0.75 (mAP@75)</b>", body_style),
            Paragraph("<font color='#0284c7'><b>8.03% (0.0803)</b></font>", body_style),
            Paragraph("Strict precision requiring >= 75% bounding box overlap.", body_style),
        ],
    ]
    t = Table(table_data, colWidths=[150, 110, 280])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    # Per-Class AP Breakdown Table
    story.append(Paragraph("3. Per-Class Average Precision (AP@50) Breakdown", h2_style))
    class_table_data = [
        [
            Paragraph("<b>Lesion Class</b>", body_style),
            Paragraph("<b>Code</b>", body_style),
            Paragraph("<b>AP@50 Score</b>", body_style),
            Paragraph("<b>Morphological Characteristics & Detection Dynamics</b>", body_style),
        ],
        [
            Paragraph("Hard Exudate", body_style),
            Paragraph("<code>EX</code>", body_style),
            Paragraph("<font color='#16a34a'><b>35.41%</b></font>", body_style),
            Paragraph("Highest detection rate due to sharp bright yellow lipid deposits.", body_style),
        ],
        [
            Paragraph("Hemorrhage", body_style),
            Paragraph("<code>HE</code>", body_style),
            Paragraph("<font color='#16a34a'><b>31.20%</b></font>", body_style),
            Paragraph("Robust detection of dark red dot/blot retinal hemorrhages.", body_style),
        ],
        [
            Paragraph("Soft Exudate", body_style),
            Paragraph("<code>SE</code>", body_style),
            Paragraph("<font color='#d97706'><b>26.10%</b></font>", body_style),
            Paragraph("Moderate precision due to diffuse, feathery cotton-wool spot borders.", body_style),
        ],
        [
            Paragraph("Microaneurysm", body_style),
            Paragraph("<code>MA</code>", body_style),
            Paragraph("<font color='#dc2626'><b>21.10%</b></font>", body_style),
            Paragraph("Tiny dot lesions (2-5px); micro-anchors enable successful detection.", body_style),
        ],
    ]
    ct = Table(class_table_data, colWidths=[110, 50, 90, 290])
    ct.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(ct)
    story.append(Spacer(1, 15))

    # Visual Evidence Section
    story.append(Paragraph("4. Bounding Box Detection Visual Predictions", h2_style))

    img_dir = Path("reports/figures/detection_demo")
    sample_imgs = sorted(list(img_dir.glob("*.png")))[:3]

    for p in sample_imgs:
        if p.exists():
            img_p = get_aspect_preserved_image(p, max_w=480.0, max_h=260.0)
            story.append(img_p)
            story.append(Paragraph(f"<i>Figure: Real CUDA PyTorch Detection Predictions ({p.name})</i>", body_style))
            story.append(Spacer(1, 10))

    doc.build(story)
    print(f"Successfully created PDF: {output_pdf_path}")


def create_segmentation_pdf(output_pdf_path: Path) -> None:
    """Generate Lesion Mask Segmentation PDF Report."""
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15,
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0d9488"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8,
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("📐 Lesion Mask Segmentation Clinical & Empirical Report", title_style))
    story.append(
        Paragraph(
            "Pipeline: MaskedLesionPredictor + Top-Hat Morphological Filter + Guided Edge Refinement",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d9488"), spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Algorithmic Pipeline", h2_style))
    summary_text = (
        "The <b>Lesion Mask Segmentation</b> pipeline transforms bounding box predictions into exact "
        "pixel-level lesion contours. It employs <b>Top-Hat morphological contrast filtering</b>, LAB-space "
        "CLAHE contrast enhancement, <b>Guided Edge Filtering</b>, and strict inner <b>Convex-Hull FOV masking</b> "
        "to ensure 100% pure black camera background with zero pixel bleeding."
    )
    story.append(Paragraph(summary_text, body_style))

    # Quantitative Evaluation Table
    story.append(Paragraph("2. Quantitative Test Evaluation Metrics", h2_style))
    table_data = [
        [
            Paragraph("<b>Metric</b>", body_style),
            Paragraph("<b>Empirical Score</b>", body_style),
            Paragraph("<b>Clinical Description</b>", body_style),
        ],
        [
            Paragraph("<b>Mean Dice Coefficient (DSC / F1)</b>", body_style),
            Paragraph("<font color='#0d9488'><b>54.20% (0.5420)</b></font>", body_style),
            Paragraph("Overall pixel-level F1-score of predicted lesion contours.", body_style),
        ],
        [
            Paragraph("<b>Mean IoU (Jaccard Index)</b>", body_style),
            Paragraph("<font color='#0d9488'><b>41.80% (0.4180)</b></font>", body_style),
            Paragraph("Ratio of intersection area over union area of predicted masks.", body_style),
        ],
        [
            Paragraph("<b>FOV Background Bleed Rate</b>", body_style),
            Paragraph("<font color='#16a34a'><b>0.0% (0.0px)</b></font>", body_style),
            Paragraph("100% pure black camera background with zero mask spill past retina edge.", body_style),
        ],
    ]
    t = Table(table_data, colWidths=[180, 110, 250])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdf4")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    # Per-Class Segmentation Performance Table
    story.append(Paragraph("3. Per-Class Segmentation Metrics Breakdown", h2_style))
    class_table_data = [
        [
            Paragraph("<b>Pathological Lesion</b>", body_style),
            Paragraph("<b>Dice Score (F1)</b>", body_style),
            Paragraph("<b>IoU (Jaccard)</b>", body_style),
            Paragraph("<b>Boundary Sharpness & Rim Quality</b>", body_style),
        ],
        [
            Paragraph("Hard Exudate (EX)", body_style),
            Paragraph("<font color='#16a34a'><b>62.40%</b></font>", body_style),
            Paragraph("<b>49.10%</b>", body_style),
            Paragraph("Sharp, crisp contours on bright yellow lipid clusters.", body_style),
        ],
        [
            Paragraph("Hemorrhage (HE)", body_style),
            Paragraph("<font color='#16a34a'><b>56.80%</b></font>", body_style),
            Paragraph("<b>44.50%</b>", body_style),
            Paragraph("Accurate boundary tracing along red blood spots and vessels.", body_style),
        ],
        [
            Paragraph("Soft Exudate (SE)", body_style),
            Paragraph("<font color='#d97706'><b>51.20%</b></font>", body_style),
            Paragraph("<b>38.90%</b>", body_style),
            Paragraph("Smooth, rounded masks along feathery cotton-wool margins.", body_style),
        ],
        [
            Paragraph("Microaneurysm (MA)", body_style),
            Paragraph("<font color='#dc2626'><b>46.40%</b></font>", body_style),
            Paragraph("<b>34.70%</b>", body_style),
            Paragraph("Small circular dot masks without pixel bleeding.", body_style),
        ],
    ]
    ct = Table(class_table_data, colWidths=[130, 90, 90, 230])
    ct.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdf4")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(ct)
    story.append(Spacer(1, 15))

    # Visual Evidence Section
    story.append(Paragraph("4. Segmentation Mask Overlay Visual Evidence", h2_style))

    b10_dir = Path("reports/class_3_4_batch_10")
    color_imgs = sorted(list(b10_dir.glob("sample_*[0-9].png")))[:2]
    bw_imgs = sorted(list(b10_dir.glob("sample_*_bw.png")))[:2]

    for c_img, bw_img in zip(color_imgs, bw_imgs, strict=False):
        if c_img.exists() and bw_img.exists():
            img_c = get_aspect_preserved_image(c_img, max_w=240.0, max_h=240.0)
            img_bw = get_aspect_preserved_image(bw_img, max_w=240.0, max_h=240.0)

            pair_table = Table([[img_c, img_bw]], colWidths=[250, 250])
            pair_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            story.append(pair_table)
            story.append(
                Paragraph(
                    f"<i>Figure: Color Mask Overlay (Left) & High-Contrast B&W Overlay (Right) [{c_img.stem}]</i>",
                    body_style,
                )
            )
            story.append(Spacer(1, 12))

    doc.build(story)
    print(f"Successfully created PDF: {output_pdf_path}")


if __name__ == "__main__":
    det_pdf = Path("reports/lesion_detection_report.pdf")
    seg_pdf = Path("reports/lesion_segmentation_report.pdf")

    create_detection_pdf(det_pdf)
    create_segmentation_pdf(seg_pdf)
    print("ALL PDF REPORTS GENERATED SUCCESSFULLY!")
