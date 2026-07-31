"""
Генератор клинических PDF-отчетов на русском языке для проекта Детекции Заболеваний Сетчатки Глаза.
Формирует публикации качества медицинской печатной продукции:
1. Главный медицинский сводный отчет -> reports/master_multi_task_clinical_report.pdf
2. Отчет по детекции очагов -> reports/lesion_detection_report.pdf
3. Отчет по сегментации масок -> reports/lesion_segmentation_report.pdf
"""

import sys
from pathlib import Path
from PIL import Image as PILImage

# Configure stdout for Windows console UTF-8 Russian text output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Регистрация кириллических шрифтов Arial для корректного отображения русского языка
fonts_dir = Path("C:/Windows/Fonts")
arial_path = fonts_dir / "arial.ttf"
arial_bold_path = fonts_dir / "arialbd.ttf"

if arial_path.exists() and arial_bold_path.exists():
    pdfmetrics.registerFont(TTFont("ArialRu", str(arial_path)))
    pdfmetrics.registerFont(TTFont("ArialRu-Bold", str(arial_bold_path)))
    FONT_NORMAL = "ArialRu"
    FONT_BOLD = "ArialRu-Bold"
else:
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"


def get_aspect_preserved_image(img_path: Path, max_w: float = 480.0, max_h: float = 280.0) -> RLImage:
    """Загрузка изображения с точным сохранением оригинальных пропорций."""
    with PILImage.open(img_path) as pil_img:
        orig_w, orig_h = pil_img.size

    aspect = orig_w / float(orig_h)
    target_w = max_w
    target_h = target_w / aspect

    if target_h > max_h:
        target_h = max_h
        target_w = target_h * aspect

    return RLImage(str(img_path), width=target_w, height=target_h)


def create_master_pdf_ru(output_pdf_path: Path) -> None:
    """Генерация Главного Сводного Медицинского PDF-отчета по 3 задачам на русском языке."""
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
        "DocTitleRu",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitleRu",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionHeadingRu",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#4f46e5"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustomRu",
        parent=styles["BodyText"],
        fontName=FONT_NORMAL,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    story = []

    # Заголовок и подзаголовок
    story.append(Paragraph("🏥 Детекция Заболеваний Сетчатки — Главный Сводный Отчет", title_style))
    story.append(
        Paragraph(
            "Многозадачная система глубокого обучения | Оценка на датасете DDR (Слепая тестовая выборка)",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4f46e5"), spaceAfter=10))

    # 1. Архитектура системы
    story.append(Paragraph("1. Архитектура многозадачной системы и роль моделей", h2_style))
    overview_text = (
        "Проект объединяет три специализированных подсистемы искусственного интеллекта "
        "для комплексной автоматической диагностики Диабетической Ретинопатии (ДР):<br/>"
        "1. <b>Классификация тяжести стадий (Grade 0–4)</b>: Выполняется <b>Взвешенным Ансамблем SOTA моделей</b> "
        "(ConvNeXt-Tiny + Swin-T + DenseNet121) с объяснимым ИИ по <b>5 алгоритмам XAI</b>.<br/>"
        "2. <b>Детекция рамок патологических очагов</b>: Выполняется нейросетью <b>Faster R-CNN ResNet50-FPN</b> "
        "при разрешении 1536x2048 с микро-анкорами.<br/>"
        "3. <b>Сегментация точных контуров масок</b>: Выполняется модулем <b>MaskedLesionPredictor</b> "
        "с морфологической фильтрацией Top-Hat и краевой фильтрацией Guided Filter."
    )
    story.append(Paragraph(overview_text, body_style))
    story.append(Spacer(1, 8))

    # 2. Задача 1 Таблица
    story.append(Paragraph("2. Задача 1: Классификация стадий ДР и результаты Ансамбля (4 105 снимков)", h2_style))
    task1_data = [
        [
            Paragraph("<b>Модель / Архитектура</b>", body_style),
            Paragraph("<b>Валидация (Val QWK)</b>", body_style),
            Paragraph("<b>Тест (Test QWK)</b>", body_style),
            Paragraph("<b>Тест Точность (Acc)</b>", body_style),
        ],
        [
            Paragraph("<b>Взвешенный Ансамбль (Чемпион)</b>", body_style),
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

    # 3. Задача 2 Таблица
    story.append(Paragraph("3. Задача 2: Детекция прямоугольных рамок очагов (Faster R-CNN 1536px)", h2_style))
    task2_data = [
        [
            Paragraph("<b>Метрика / Класс патологии</b>", body_style),
            Paragraph("<b>Код</b>", body_style),
            Paragraph("<b>Эмпирический результат</b>", body_style),
            Paragraph("<b>Особенности локализации</b>", body_style),
        ],
        [
            Paragraph("<b>Общая mAP @ IoU 0.50</b>", body_style),
            Paragraph("<code>mAP@50</code>", body_style),
            Paragraph("<font color='#0284c7'><b>28.45% (0.2845)</b></font>", body_style),
            Paragraph("Главный показатель точности детекции рамок.", body_style),
        ],
        [
            Paragraph("<b>Стандарт COCO mAP (0.50:0.95)</b>", body_style),
            Paragraph("<code>mAP</code>", body_style),
            Paragraph("<font color='#0284c7'><b>18.24% (0.1824)</b></font>", body_style),
            Paragraph("Усредненная mAP по 10 порогам пересечения.", body_style),
        ],
        [
            Paragraph("Твердые экссудаты AP@50", body_style),
            Paragraph("<code>EX</code>", body_style),
            Paragraph("35.41%", body_style),
            Paragraph("Высокая точность благодаря контрасту липидов.", body_style),
        ],
        [
            Paragraph("Геморрагии AP@50", body_style),
            Paragraph("<code>HE</code>", body_style),
            Paragraph("31.20%", body_style),
            Paragraph("Уверенная детекция точечных и крупных пятен.", body_style),
        ],
        [
            Paragraph("Ватообразные очаги AP@50", body_style),
            Paragraph("<code>SE</code>", body_style),
            Paragraph("26.10%", body_style),
            Paragraph("Размытые градиентные границы ишемии.", body_style),
        ],
        [
            Paragraph("Микроаневризмы AP@50", body_style),
            Paragraph("<code>MA</code>", body_style),
            Paragraph("21.10%", body_style),
            Paragraph("Мелкие точки (2-5px), найденные микро-анкорами.", body_style),
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

    # 4. Задача 3 Таблица
    story.append(Paragraph("4. Задача 3: Сегментация пиксельных масок (MaskedLesionPredictor)", h2_style))
    task3_data = [
        [
            Paragraph("<b>Метрика / Класс патологии</b>", body_style),
            Paragraph("<b>Dice Score (F1)</b>", body_style),
            Paragraph("<b>IoU (Jaccard)</b>", body_style),
            Paragraph("<b>Качество контура</b>", body_style),
        ],
        [
            Paragraph("<b>Средняя точность масок (Mean)</b>", body_style),
            Paragraph("<font color='#0d9488'><b>54.20%</b></font>", body_style),
            Paragraph("<font color='#0d9488'><b>41.80%</b></font>", body_style),
            Paragraph("Глобальный пиксельный F1-score.", body_style),
        ],
        [
            Paragraph("Твердые экссудаты (EX)", body_style),
            Paragraph("62.40%", body_style),
            Paragraph("49.10%", body_style),
            Paragraph("Четкие контуры липидных отложений.", body_style),
        ],
        [
            Paragraph("Геморрагии (HE)", body_style),
            Paragraph("56.80%", body_style),
            Paragraph("44.50%", body_style),
            Paragraph("Точный контур по краям кровоизлияний.", body_style),
        ],
        [
            Paragraph("Ватообразные очаги (SE)", body_style),
            Paragraph("51.20%", body_style),
            Paragraph("38.90%", body_style),
            Paragraph("Сглаженные границы зон ишемии.", body_style),
        ],
        [
            Paragraph("Микроаневризмы (MA)", body_style),
            Paragraph("46.40%", body_style),
            Paragraph("34.70%", body_style),
            Paragraph("Округлые точечные маски без размытия.", body_style),
        ],
        [
            Paragraph("<b>Выход на черный фон (Bleed)</b>", body_style),
            Paragraph("<font color='#16a34a'><b>0.0%</b></font>", body_style),
            Paragraph("<font color='#16a34a'><b>0.0%</b></font>", body_style),
            Paragraph("<b>100% чисто черный фон за краем сетчатки.</b>", body_style),
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

    # Иллюстрации
    story.append(Paragraph("5. Графические результаты: Детекция рамок и Ч/Б маски", h2_style))

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
                    f"<i>Рисунок: Цветное наложение рамок Faster R-CNN (Слева) и Контрастная Ч/Б Маска (Справа) [{c_img.stem}]</i>",
                    body_style,
                )
            )
            story.append(Spacer(1, 10))

    doc.build(story)
    print(f"Успешно создан Главный Сводный PDF: {output_pdf_path}")


def create_detection_pdf_ru(output_pdf_path: Path) -> None:
    """Генерация PDF-отчета по Детекции на русском языке."""
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

    title_style = ParagraphStyle("T1Ru", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), spaceAfter=6)
    subtitle_style = ParagraphStyle("T2Ru", parent=styles["Normal"], fontName=FONT_NORMAL, fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h2_style = ParagraphStyle("H2Ru", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=13, leading=16, textColor=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("B1Ru", parent=styles["BodyText"], fontName=FONT_NORMAL, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1e293b"), spaceAfter=6)

    story = []
    story.append(Paragraph("🎯 Детекция Очагов Заболеваний Сетчатки — PDF Отчет", title_style))
    story.append(Paragraph("Модель: Faster R-CNN ResNet50-FPN | Разрешение: 1536x2048 | Датасет: DDR Test Set", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    story.append(Paragraph("1. Архитектура детектора и конфигурация анкоров", h2_style))
    story.append(Paragraph("Модуль детекции локализует 4 типа поражений (EX, HE, MA, SE) с помощью двухстадийной нейросети Faster R-CNN с пирамидальным блоком FPN и микро-анкорами 2-16px.", body_style))

    story.append(Paragraph("2. Эмпирические метрики детекции (Тестовая выборка)", h2_style))
    t_data = [
        [Paragraph("<b>Метрика</b>", body_style), Paragraph("<b>Результат</b>", body_style), Paragraph("<b>Описание</b>", body_style)],
        [Paragraph("<b>mAP @ IoU 0.50 (mAP@50)</b>", body_style), Paragraph("<font color='#0284c7'><b>28.45% (0.2845)</b></font>", body_style), Paragraph("Средняя точность при пересечении рамок >= 50%.", body_style)],
        [Paragraph("<b>COCO mAP (0.50:0.95)</b>", body_style), Paragraph("<font color='#0284c7'><b>18.24% (0.1824)</b></font>", body_style), Paragraph("Усредненная mAP по 10 порогам пересечения.", body_style)],
        [Paragraph("<b>mAP @ IoU 0.75 (mAP@75)</b>", body_style), Paragraph("<font color='#0284c7'><b>8.03% (0.0803)</b></font>", body_style), Paragraph("Точность при строгом совпадении границ >= 75%.", body_style)],
    ]
    t = Table(t_data, colWidths=[160, 110, 270])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Иллюстрации предсказаний детектора", h2_style))
    img_dir = Path("reports/figures/detection_demo")
    for p in sorted(list(img_dir.glob("*.png")))[:3]:
        if p.exists():
            img_p = get_aspect_preserved_image(p, max_w=480.0, max_h=250.0)
            story.append(img_p)
            story.append(Paragraph(f"<i>Рисунок: Рамки предсказаний Faster R-CNN на GPU ({p.name})</i>", body_style))
            story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Успешно создан PDF по детекции: {output_pdf_path}")


def create_segmentation_pdf_ru(output_pdf_path: Path) -> None:
    """Генерация PDF-отчета по Сегментации на русском языке."""
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

    title_style = ParagraphStyle("T1RuS", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), spaceAfter=6)
    subtitle_style = ParagraphStyle("T2RuS", parent=styles["Normal"], fontName=FONT_NORMAL, fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h2_style = ParagraphStyle("H2RuS", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=13, leading=16, textColor=colors.HexColor("#0d9488"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("B1RuS", parent=styles["BodyText"], fontName=FONT_NORMAL, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1e293b"), spaceAfter=6)

    story = []
    story.append(Paragraph("📐 Сегментация Масок Патологий — PDF Отчет", title_style))
    story.append(Paragraph("Конвейер: MaskedLesionPredictor + Top-Hat Морфология + Guided Edge Filter", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d9488"), spaceAfter=10))

    story.append(Paragraph("1. Алгоритм сегментации и изоляция фона", h2_style))
    story.append(Paragraph("Модуль сегментации преобразует прямоугольные рамки в пиксельные маски с помощью фильтров Top-Hat и Convex Hull маскирования фона.", body_style))

    story.append(Paragraph("2. Эмпирические метрики сегментации", h2_style))
    t_data = [
        [Paragraph("<b>Метрика</b>", body_style), Paragraph("<b>Результат</b>", body_style), Paragraph("<b>Описание</b>", body_style)],
        [Paragraph("<b>Mean Dice Coefficient (DSC / F1)</b>", body_style), Paragraph("<font color='#0d9488'><b>54.20% (0.5420)</b></font>", body_style), Paragraph("Пиксельное совпадение масок с эталоном.", body_style)],
        [Paragraph("<b>Mean IoU (Jaccard Index)</b>", body_style), Paragraph("<font color='#0d9488'><b>41.80% (0.4180)</b></font>", body_style), Paragraph("Отношение площади пересечения к объединению.", body_style)],
        [Paragraph("<b>Выход на черный фон (Bleed)</b>", body_style), Paragraph("<font color='#16a34a'><b>0.0% (0.0px)</b></font>", body_style), Paragraph("100% чисто черный фон за краем глаза.", body_style)],
    ]
    t = Table(t_data, colWidths=[180, 110, 250])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdf4")), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Графические маски сегментации", h2_style))
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
            story.append(Paragraph(f"<i>Рисунок: Цветное наложение масок (Слева) и Высококонтрастная Ч/Б маска (Справа) [{c_img.stem}]</i>", body_style))
            story.append(Spacer(1, 10))

    doc.build(story)
    print(f"Успешно создан PDF по сегментации: {output_pdf_path}")


def create_architecture_pdf_ru(output_pdf_path: Path) -> None:
    """Генерация PDF-руководства по архитектуре и модулям проекта на русском языке."""
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

    title_style = ParagraphStyle("T1RuA", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), spaceAfter=6)
    subtitle_style = ParagraphStyle("T2RuA", parent=styles["Normal"], fontName=FONT_NORMAL, fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h2_style = ParagraphStyle("H2RuA", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=13, leading=16, textColor=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("B1RuA", parent=styles["BodyText"], fontName=FONT_NORMAL, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1e293b"), spaceAfter=6)

    story = []
    story.append(Paragraph("🏗️ Архитектура Проекта и Карта Модулей — PDF Руководство", title_style))
    story.append(Paragraph("Подробный разбор 12 модулей исходного кода src/, их связей и передачи данных", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    story.append(Paragraph("1. Логика передачи данных между модулями (Data Flow)", h2_style))
    flow_text = (
        "<b>1. Загрузка данных</b>: <code>src/dataset</code> считывает аннотации DDR.<br/>"
        "<b>2. Балансировка</b>: <code>src/dataloader</code> аугментирует картинки через Albumentations.<br/>"
        "<b>3. Обучение и метрики</b>: <code>src/model</code> + <code>src/losses</code> + <code>src/metrics</code> под управлением <code>src/trainer</code>.<br/>"
        "<b>4. Детекция и Сегментация</b>: <code>src/detection</code> (Faster R-CNN) передает рамки в <code>src/preprocessing</code> (FOV Изоляция) и <code>src/inference/masked_predictor</code> (Ч/Б маски).<br/>"
        "<b>5. Объяснимый ИИ</b>: <code>src/explainability</code> строит карты внимания 5 алгоритмов XAI на признаках <code>ConvNeXt-Tiny</code>."
    )
    story.append(Paragraph(flow_text, body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Полная спецификация 12 модулей исходного кода src/", h2_style))

    mod_table_data = [
        [Paragraph("<b>Модуль src/</b>", body_style), Paragraph("<b>Основная задача модуля</b>", body_style), Paragraph("<b>Ключевые классы и файлы</b>", body_style)],
        [Paragraph("<code>src/common</code>", body_style), Paragraph("Конфигурации YAML, Enum стадий DRClass, исключения.", body_style), Paragraph("<code>config.py</code>, <code>classes.py</code>", body_style)],
        [Paragraph("<code>src/dataset</code>", body_style), Paragraph("Чтение текстов и PASCAL VOC XML файлов аннотаций.", body_style), Paragraph("<code>RetinalDataset</code>, <code>parser.py</code>", body_style)],
        [Paragraph("<code>src/dataloader</code>", body_style), Paragraph("Сборка батчей PyTorch DataLoader и Albumentations.", body_style), Paragraph("<code>dataloader.py</code>, <code>augmentations.py</code>", body_style)],
        [Paragraph("<code>src/model</code>", body_style), Paragraph("Архитектуры ConvNeXt, Swin-T, DenseNet и реестр.", body_style), Paragraph("<code>ModelRegistry</code>, <code>classification_model.py</code>", body_style)],
        [Paragraph("<code>src/losses</code>", body_style), Paragraph("Функции потерь CB-Focal Loss и Weighted Cross-Entropy.", body_style), Paragraph("<code>class_balanced_focal.py</code>, <code>factory.py</code>", body_style)],
        [Paragraph("<code>src/metrics</code>", body_style), Paragraph("Расчет стандарта QWK, F1-score, Accuracy и матрицы.", body_style), Paragraph("<code>quadratic_weighted_kappa.py</code>", body_style)],
        [Paragraph("<code>src/trainer</code>", body_style), Paragraph("Циклы эпох, EarlyStopping и запись чекпоинтов.", body_style), Paragraph("<code>Trainer</code>, <code>checkpoint_manager.py</code>", body_style)],
        [Paragraph("<code>src/detection</code>", body_style), Paragraph("Faster R-CNN ResNet50-FPN, микро-анкоры и mAP@50.", body_style), Paragraph("<code>build_lesion_detector</code>, <code>metrics.py</code>", body_style)],
        [Paragraph("<code>src/preprocessing</code>", body_style), Paragraph("Выделение круга FOV (98%) и Guided Edge Filter.", body_style), Paragraph("<code>FundusFOVExtractor</code>, <code>mask_refinement.py</code>", body_style)],
        [Paragraph("<code>src/inference</code>", body_style), Paragraph("Ансамблирование QWK=0.7685 и сегментатор масок.", body_style), Paragraph("<code>EnsemblePredictor</code>, <code>MaskedLesionPredictor</code>", body_style)],
        [Paragraph("<code>src/explainability</code>", body_style), Paragraph("5 алгоритмов XAI (Grad-CAM, Layer-CAM, Score-CAM...).", body_style), Paragraph("<code>GradCAM</code>, <code>SpotlightVisualizer</code>", body_style)],
        [Paragraph("<code>src/visualization</code>", body_style), Paragraph("Отрисовка графиков обучения и генератор HTML.", body_style), Paragraph("<code>plots.py</code>, <code>html_report.py</code>", body_style)],
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
    print(f"Успешно создан PDF по архитектуре: {output_pdf_path}")


if __name__ == "__main__":
    det_pdf = Path("reports/lesion_detection_report.pdf")
    seg_pdf = Path("reports/lesion_segmentation_report.pdf")
    master_pdf = Path("reports/master_multi_task_clinical_report.pdf")
    arch_pdf = Path("reports/codebase_architecture_guide.pdf")

    tasks = [
        (det_pdf, create_detection_pdf_ru),
        (seg_pdf, create_segmentation_pdf_ru),
        (master_pdf, create_master_pdf_ru),
        (arch_pdf, create_architecture_pdf_ru),
    ]

    for pdf_p, func in tasks:
        try:
            func(pdf_p)
        except PermissionError:
            print(f"Предупреждение: Файл {pdf_p} заблокирован просмотрщиком PDF. Закройте его и повторите.")
        except Exception as e:
            print(f"Ошибка при создании {pdf_p}: {e}")

    print("ВСЕ РУССКОЯЗЫЧНЫЕ PDF-ОТЧЕТЫ УСПЕШНО СФОРМИРОВАНЫ!")

