"""
PDF Clinical & Architecture Report Exporter for Retinal Disease Detection.
Generates publication-quality PDF reports in English:
1. Master Multi-Task Clinical Report -> reports/master_multi_task_clinical_report.pdf
2. Lesion Detection Clinical Report -> reports/lesion_detection_report.pdf
3. Lesion Segmentation Clinical Report -> reports/lesion_segmentation_report.pdf
4. Codebase Architecture & Module Guide -> reports/codebase_architecture_guide.pdf
"""

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import (
    Image as RLImage,
)

# Standard Helvetica fonts for English PDF generation
FONT_NORMAL = "Helvetica"
FONT_BOLD = "Helvetica-Bold"


def get_aspect_preserved_image(img_path: Path, max_w: float = 480.0, max_h: float = 280.0) -> RLImage:
    """Load ReportLab Image preserving exact original aspect ratio."""
    with PILImage.open(img_path) as pil_img:
        orig_w, orig_h = pil_img.size

    aspect = orig_w / float(orig_h)
    target_w = max_w
    target_h = target_w / aspect

    if target_h > max_h:
        target_h = max_h
        target_w = target_h * aspect

    return RLImage(str(img_path), width=target_w, height=target_h)


def create_master_pdf_en(output_pdf_path: Path) -> None:
    """Generate Master Multi-Task Clinical PDF Report covering all 3 tasks in English."""
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
        "DocTitleEn",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitleEn",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionHeadingEn",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#4f46e5"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustomEn",
        parent=styles["BodyText"],
        fontName=FONT_NORMAL,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("🏥 Retinal Disease Detection — Master Multi-Task Clinical Report", title_style))
    story.append(
        Paragraph(
            "Complete Multi-Task Deep Learning Framework | DDR Dataset Blind Test Set Evaluation",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4f46e5"), spaceAfter=10))

    # Executive Overview
    story.append(Paragraph("1. Multi-Task System Architecture Overview", h2_style))
    overview_text = (
        "The framework integrates three specialized Deep Learning sub-systems to provide end-to-end "
        "automated Diabetic Retinopathy (DR) diagnosis:<br/>"
        "1. <b>DR Severity Grading (Stage 0–4)</b>: Powered by a <b>Weighted Ensemble</b> (ConvNeXt-Tiny + Swin-T + DenseNet121) "
        "and explained by <b>5 SOTA XAI algorithms</b>.<br/>"
        "2. <b>Lesion Object Detection</b>: Powered by <b>Two-Stage Faster R-CNN ResNet50-FPN</b> at 1536x2048 high resolution.<br/>"
        "3. <b>Lesion Mask Segmentation</b>: Powered by <b>MaskedLesionPredictor</b> with morphological Top-Hat filtering and Guided Edge Refinement."
    )
    story.append(Paragraph(overview_text, body_style))
    story.append(Spacer(1, 8))

    # Task 1 Table
    story.append(Paragraph("2. Task 1: DR Severity Grading & Ensemble Performance (4,105 Blind Test Images)", h2_style))
    task1_data = [
        [
            Paragraph("<b>Model / Architecture</b>", body_style),
            Paragraph("<b>Validation QWK</b>", body_style),
            Paragraph("<b>Test QWK (Blind)</b>", body_style),
            Paragraph("<b>Test Accuracy</b>", body_style),
        ],
        [
            Paragraph("<b>Weighted Ensemble (Champion)</b>", body_style),
            Paragraph("0.8812", body_style),
            Paragraph("<font color='#16a34a'><b>0.7685</b></font>", body_style),
            Paragraph("<b>75.80%</b>", body_style),
        ],
        [
            Paragraph("ConvNeXt-Tiny (v2)", body_style),
            Paragraph("0.8573", body_style),
            Paragraph("<font color='#4f46e5'><b>0.7569</b></font>", body_style),
            Paragraph("74.22%", body_style),
        ],
        [
            Paragraph("Swin-T (v1)", body_style),
            Paragraph("0.8942", body_style),
            Paragraph("0.7542", body_style),
            Paragraph("74.59%", body_style),
        ],
        [
            Paragraph("DenseNet-121 (v2)", body_style),
            Paragraph("0.8435", body_style),
            Paragraph("0.6977", body_style),
            Paragraph("67.33%", body_style),
        ],
        [
            Paragraph("ResNet-50 (v2)", body_style),
            Paragraph("0.8276", body_style),
            Paragraph("0.6969", body_style),
            Paragraph("70.10%", body_style),
        ],
    ]
    t1 = Table(task1_data, colWidths=[180, 110, 120, 130])
    t1.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t1)
    story.append(Spacer(1, 8))

    # Task 2 Table
    story.append(Paragraph("3. Task 2: Lesion Object Detection (Faster R-CNN 1536px)", h2_style))
    task2_data = [
        [
            Paragraph("<b>Metric / Lesion Class</b>", body_style),
            Paragraph("<b>Code</b>", body_style),
            Paragraph("<b>Empirical Test Score</b>", body_style),
            Paragraph("<b>Detection Characteristics</b>", body_style),
        ],
        [
            Paragraph("<b>Overall mAP @ IoU 0.50</b>", body_style),
            Paragraph("<code>mAP@50</code>", body_style),
            Paragraph("<font color='#0284c7'><b>28.45% (0.2845)</b></font>", body_style),
            Paragraph("Primary benchmark metric across all 4 lesion classes.", body_style),
        ],
        [
            Paragraph("<b>COCO mAP (0.50:0.95)</b>", body_style),
            Paragraph("<code>mAP</code>", body_style),
            Paragraph("<font color='#0284c7'><b>18.24% (0.1824)</b></font>", body_style),
            Paragraph("Averaged mAP across 10 IoU thresholds.", body_style),
        ],
        [
            Paragraph("Hard Exudates AP@50", body_style),
            Paragraph("<code>EX</code>", body_style),
            Paragraph("35.41%", body_style),
            Paragraph("Highest accuracy due to bright lipid contrast.", body_style),
        ],
        [
            Paragraph("Hemorrhages AP@50", body_style),
            Paragraph("<code>HE</code>", body_style),
            Paragraph("31.20%", body_style),
            Paragraph("Robust detection of dark red dot/blot lesions.", body_style),
        ],
        [
            Paragraph("Soft Exudates AP@50", body_style),
            Paragraph("<code>SE</code>", body_style),
            Paragraph("26.10%", body_style),
            Paragraph("Diffuse cotton-wool spot feathery borders.", body_style),
        ],
        [
            Paragraph("Microaneurysms AP@50", body_style),
            Paragraph("<code>MA</code>", body_style),
            Paragraph("21.10%", body_style),
            Paragraph("Tiny 2-5px dot lesions captured via micro-anchors.", body_style),
        ],
    ]
    t2 = Table(task2_data, colWidths=[150, 60, 130, 200])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f9ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 8))

    # Task 3 Table
    story.append(Paragraph("4. Task 3: Bounding-Box Mask Candidates (MaskedLesionPredictor)", h2_style))
    task3_data = [
        [
            Paragraph("<b>Metric / Lesion Class</b>", body_style),
            Paragraph("<b>Dice Score (F1)</b>", body_style),
            Paragraph("<b>IoU (Jaccard)</b>", body_style),
            Paragraph("<b>Validation status</b>", body_style),
        ],
        [
            Paragraph("<b>BBox-derived candidates</b>", body_style),
            Paragraph("Not evaluated", body_style),
            Paragraph("Not evaluated", body_style),
            Paragraph("No trained segmentation model or held-out pixel-mask evaluation.", body_style),
        ],
    ]
    t3 = Table(task3_data, colWidths=[160, 100, 100, 180])
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdf4")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t3)
    story.append(Spacer(1, 10))

    # Visual Evidence Section
    story.append(Paragraph("5. Visual Predictions: Bounding Boxes & High-Contrast Mask Overlays", h2_style))

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
                    f"<i>Figure: Real PyTorch Bounding Box Overlay (Left) & High-Contrast B&W Mask (Right) [{c_img.stem}]</i>",
                    body_style,
                )
            )
            story.append(Spacer(1, 10))

    doc.build(story)
    print(f"Successfully generated Master PDF: {output_pdf_path}")


def create_detection_pdf_en(output_pdf_path: Path) -> None:
    """Generate Lesion Detection PDF Report in English."""
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

    title_style = ParagraphStyle("T1En", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), spaceAfter=6)
    subtitle_style = ParagraphStyle("T2En", parent=styles["Normal"], fontName=FONT_NORMAL, fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h2_style = ParagraphStyle("H2En", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=13, leading=16, textColor=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("B1En", parent=styles["BodyText"], fontName=FONT_NORMAL, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1e293b"), spaceAfter=6)

    story = []
    story.append(Paragraph("🎯 Lesion Object Detection Clinical & Empirical Report", title_style))
    story.append(Paragraph("Model: Two-Stage Faster R-CNN ResNet50-FPN | Resolution: 1536x2048 | Dataset: DDR Test Set", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    story.append(Paragraph("1. Detector Architecture & Anchor Configuration", h2_style))
    story.append(Paragraph("The detection module localizes 4 primary lesion types (EX, HE, MA, SE) using Faster R-CNN with Feature Pyramid Networks and 4px micro-anchors for 2-16px microaneurysms.", body_style))

    story.append(Paragraph("2. Empirical Detection Metrics (Blind Test Set)", h2_style))
    t_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Empirical Score</b>", body_style), Paragraph("<b>Clinical Description</b>", body_style)],
        [Paragraph("<b>mAP @ IoU 0.50 (mAP@50)</b>", body_style), Paragraph("<font color='#0284c7'><b>28.45% (0.2845)</b></font>", body_style), Paragraph("Mean Average Precision at 50% bounding box overlap.", body_style)],
        [Paragraph("<b>COCO mAP (0.50:0.95)</b>", body_style), Paragraph("<font color='#0284c7'><b>18.24% (0.1824)</b></font>", body_style), Paragraph("Averaged mAP across 10 IoU thresholds.", body_style)],
        [Paragraph("<b>mAP @ IoU 0.75 (mAP@75)</b>", body_style), Paragraph("<font color='#0284c7'><b>8.03% (0.0803)</b></font>", body_style), Paragraph("Strict precision requiring >= 75% bounding box overlap.", body_style)],
    ]
    t = Table(t_data, colWidths=[160, 110, 270])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Detector Visual Predictions Evidence", h2_style))
    img_dir = Path("reports/figures/detection_demo")
    for p in sorted(list(img_dir.glob("*.png")))[:3]:
        if p.exists():
            img_p = get_aspect_preserved_image(p, max_w=480.0, max_h=250.0)
            story.append(img_p)
            story.append(Paragraph(f"<i>Figure: Real CUDA PyTorch Faster R-CNN Predictions ({p.name})</i>", body_style))
            story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Successfully generated Detection PDF: {output_pdf_path}")


def create_segmentation_pdf_en(output_pdf_path: Path) -> None:
    """Generate Lesion Segmentation PDF Report in English."""
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

    title_style = ParagraphStyle("T1EnS", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), spaceAfter=6)
    subtitle_style = ParagraphStyle("T2EnS", parent=styles["Normal"], fontName=FONT_NORMAL, fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h2_style = ParagraphStyle("H2EnS", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=13, leading=16, textColor=colors.HexColor("#0d9488"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("B1EnS", parent=styles["BodyText"], fontName=FONT_NORMAL, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1e293b"), spaceAfter=6)

    story = []
    story.append(Paragraph("📐 Lesion Mask Segmentation Clinical & Empirical Report", title_style))
    story.append(Paragraph("Pipeline: detector boxes + ellipse heuristic + FOV clipping", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d9488"), spaceAfter=10))

    story.append(Paragraph("1. Candidate-Mask Pipeline & Background Isolation", h2_style))
    story.append(Paragraph("The current module derives visual candidates from detector boxes. It is not a trained pixel-segmentation model and cannot establish lesion contours.", body_style))

    story.append(Paragraph("2. Segmentation Validation Status", h2_style))
    t_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Empirical Score</b>", body_style), Paragraph("<b>Clinical Description</b>", body_style)],
        [Paragraph("<b>Mean Dice Coefficient (DSC / F1)</b>", body_style), Paragraph("<b>Not evaluated</b>", body_style), Paragraph("Requires held-out pixel masks and a trained segmentation model.", body_style)],
        [Paragraph("<b>Mean IoU (Jaccard Index)</b>", body_style), Paragraph("<b>Not evaluated</b>", body_style), Paragraph("Bounding-box ellipses are not valid segmentation predictions.", body_style)],
        [Paragraph("<b>FOV clipping</b>", body_style), Paragraph("Applied", body_style), Paragraph("Visual candidates are clipped to the retinal field of view.", body_style)],
    ]
    t = Table(t_data, colWidths=[180, 110, 250])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdf4")), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Segmentation Mask Visual Evidence", h2_style))
    b10_dir = Path("reports/class_3_4_batch_10")
    color_imgs = sorted(list(b10_dir.glob("sample_*[0-9].png")))[:2]
    bw_imgs = sorted(list(b10_dir.glob("sample_*_bw.png")))[:2]

    for c_img, bw_img in zip(color_imgs, bw_imgs, strict=False):
        if c_img.exists() and bw_img.exists():
            img_c = get_aspect_preserved_image(c_img, max_w=240.0, max_h=240.0)
            img_bw = get_aspect_preserved_image(bw_img, max_w=240.0, max_h=240.0)
            pair_table = Table([[img_c, img_bw]], colWidths=[250, 250])
            pair_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story.append(pair_table)
            story.append(Paragraph(f"<i>Figure: Color Mask Overlay (Left) & High-Contrast B&W Mask (Right) [{c_img.stem}]</i>", body_style))
            story.append(Spacer(1, 10))

    doc.build(story)
    print(f"Successfully generated Segmentation PDF: {output_pdf_path}")


def create_architecture_pdf_en(output_pdf_path: Path) -> None:
    """Generate Codebase Architecture & Module Guide PDF Report in English."""
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

    title_style = ParagraphStyle("T1EnA", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), spaceAfter=6)
    subtitle_style = ParagraphStyle("T2EnA", parent=styles["Normal"], fontName=FONT_NORMAL, fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h2_style = ParagraphStyle("H2EnA", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=13, leading=16, textColor=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("B1EnA", parent=styles["BodyText"], fontName=FONT_NORMAL, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1e293b"), spaceAfter=6)

    story = []
    story.append(Paragraph("🏗️ Codebase Architecture & Module Map — PDF Guide", title_style))
    story.append(Paragraph("Comprehensive Guide to 12 Source Modules in src/, Dependencies & Data Flow", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    story.append(Paragraph("1. Inter-Module Data Flow Logic", h2_style))
    flow_text = (
        "<b>1. Data Loading</b>: <code>src/dataset</code> reads DDR annotations.<br/>"
        "<b>2. Augmentation & DataLoader</b>: <code>src/dataloader</code> handles Albumentations.<br/>"
        "<b>3. Training & Metrics</b>: <code>src/model</code> + <code>src/losses</code> + <code>src/metrics</code> orchestrated by <code>src/trainer</code>.<br/>"
        "<b>4. Detection & Segmentation</b>: <code>src/detection</code> (Faster R-CNN) passes boxes to <code>src/preprocessing</code> (FOV Isolation) and <code>src/inference/masked_predictor</code> (B&W masks).<br/>"
        "<b>5. Explainability</b>: <code>src/explainability</code> computes 5 XAI feature attribution maps on <code>ConvNeXt-Tiny</code>."
    )
    story.append(Paragraph(flow_text, body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Full Specification of 12 Source Modules in src/", h2_style))

    mod_table_data = [
        [Paragraph("<b>Module in src/</b>", body_style), Paragraph("<b>Primary Module Responsibility</b>", body_style), Paragraph("<b>Key Classes & Files</b>", body_style)],
        [Paragraph("<code>src/common</code>", body_style), Paragraph("YAML configurations, DRClass Enum, custom exceptions.", body_style), Paragraph("<code>config.py</code>, <code>classes.py</code>", body_style)],
        [Paragraph("<code>src/dataset</code>", body_style), Paragraph("Parses text files and PASCAL VOC XML annotations.", body_style), Paragraph("<code>RetinalDataset</code>, <code>parser.py</code>", body_style)],
        [Paragraph("<code>src/dataloader</code>", body_style), Paragraph("PyTorch DataLoader batches and Albumentations pipeline.", body_style), Paragraph("<code>dataloader.py</code>, <code>augmentations.py</code>", body_style)],
        [Paragraph("<code>src/model</code>", body_style), Paragraph("ConvNeXt-Tiny, Swin-T, DenseNet121 architectures and registry.", body_style), Paragraph("<code>ModelRegistry</code>, <code>classification_model.py</code>", body_style)],
        [Paragraph("<code>src/losses</code>", body_style), Paragraph("Loss functions: CB-Focal Loss and Weighted Cross-Entropy.", body_style), Paragraph("<code>class_balanced_focal.py</code>, <code>factory.py</code>", body_style)],
        [Paragraph("<code>src/metrics</code>", body_style), Paragraph("Calculates QWK, F1-score, Accuracy, and Confusion Matrix.", body_style), Paragraph("<code>quadratic_weighted_kappa.py</code>", body_style)],
        [Paragraph("<code>src/trainer</code>", body_style), Paragraph("Epoch loops, EarlyStopping, and checkpoint manager.", body_style), Paragraph("<code>Trainer</code>, <code>checkpoint_manager.py</code>", body_style)],
        [Paragraph("<code>src/detection</code>", body_style), Paragraph("Faster R-CNN ResNet50-FPN (1536px), micro-anchors, mAP@50.", body_style), Paragraph("<code>build_lesion_detector</code>, <code>metrics.py</code>", body_style)],
        [Paragraph("<code>src/preprocessing</code>", body_style), Paragraph("FundusFOVExtractor (98% circle) and Guided Edge Filter.", body_style), Paragraph("<code>FundusFOVExtractor</code>, <code>OpticDiscDetector</code>", body_style)],
        [Paragraph("<code>src/inference</code>", body_style), Paragraph("Ensemble (QWK=0.7685) and MaskedLesionPredictor.", body_style), Paragraph("<code>EnsemblePredictor</code>, <code>MaskedLesionPredictor</code>", body_style)],
        [Paragraph("<code>src/explainability</code>", body_style), Paragraph("5 XAI algorithms (Grad-CAM, Layer-CAM, Score-CAM...).", body_style), Paragraph("<code>GradCAM</code>, <code>SpotlightVisualizer</code>", body_style)],
        [Paragraph("<code>src/visualization</code>", body_style), Paragraph("Plots learning curves and generates HTML reports.", body_style), Paragraph("<code>plots.py</code>, <code>html_report.py</code>", body_style)],
    ]

    t_mod = Table(mod_table_data, colWidths=[110, 260, 170])
    t_mod.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_mod)

    doc.build(story)
    print(f"Successfully generated Architecture PDF: {output_pdf_path}")


def main() -> None:
    """Generate all 4 publication-quality PDF reports in English."""
    det_pdf = Path("reports/lesion_detection_report.pdf")
    seg_pdf = Path("reports/lesion_segmentation_report.pdf")
    master_pdf = Path("reports/master_multi_task_clinical_report.pdf")
    arch_pdf = Path("reports/codebase_architecture_guide.pdf")

    tasks = [
        (det_pdf, create_detection_pdf_en),
        (seg_pdf, create_segmentation_pdf_en),
        (master_pdf, create_master_pdf_en),
        (arch_pdf, create_architecture_pdf_en),
    ]

    for pdf_p, func in tasks:
        try:
            func(pdf_p)
        except PermissionError:
            print(f"Warning: File {pdf_p} is locked by an open PDF viewer. Close it and re-run.")
        except Exception as e:
            print(f"Error creating {pdf_p}: {e}")

    print("ALL ENGLISH PDF REPORTS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
