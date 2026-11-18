"""Build the consolidated PathSnap technical report PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "PathSnap_Technical_Report.pdf"
ROUTE_FIGURE = ROOT / "report" / "figures" / "trip_00000000_routes.png"

NAVY = colors.HexColor("#18324f")
BLUE = colors.HexColor("#3978f6")
TEAL = colors.HexColor("#148c67")
ORANGE = colors.HexColor("#ed8b2c")
INK = colors.HexColor("#2d3748")
MUTED = colors.HexColor("#637286")
PALE = colors.HexColor("#f2f6fb")
GRID = colors.HexColor("#d8e1eb")


class PipelineDiagram(Flowable):
    """Compact local pipeline drawing with no external image dependency."""

    def __init__(self, height=78):
        super().__init__()
        self.height = height

    def wrap(self, avail_width, avail_height):
        self.width = avail_width
        return self.width, self.height

    def draw(self):
        labels = [
            "React +\nLeaflet",
            "FastAPI\nservice",
            "Route-free\nfeatures",
            "Saved Random\nForest",
            "Score\nnormalization",
            "Directed HMM\n/ Viterbi",
        ]
        gap = 11
        box_width = (self.width - gap * (len(labels) - 1)) / len(labels)
        box_height = 48
        bottom = 15
        for index, label in enumerate(labels):
            x = index * (box_width + gap)
            fill = PALE if index not in (0, 5) else colors.HexColor("#e8f0ff" if index == 0 else "#e7f5ef")
            self.canv.setFillColor(fill)
            self.canv.setStrokeColor(GRID)
            self.canv.roundRect(x, bottom, box_width, box_height, 6, stroke=1, fill=1)
            self.canv.setFillColor(NAVY)
            self.canv.setFont("Helvetica-Bold", 7.8)
            lines = label.split("\n")
            line_gap = 10
            start_y = bottom + box_height / 2 + (len(lines) - 1) * line_gap / 2 - 2
            for line_index, line in enumerate(lines):
                self.canv.drawCentredString(x + box_width / 2, start_y - line_index * line_gap, line)
            if index < len(labels) - 1:
                x1 = x + box_width + 1
                x2 = x + box_width + gap - 2
                y = bottom + box_height / 2
                self.canv.setStrokeColor(MUTED)
                self.canv.setFillColor(MUTED)
                self.canv.setLineWidth(0.9)
                self.canv.line(x1, y, x2, y)
                self.canv.line(x2 - 3, y + 2.5, x2, y)
                self.canv.line(x2 - 3, y - 2.5, x2, y)
        self.canv.setFillColor(MUTED)
        self.canv.setFont("Helvetica", 7.5)
        self.canv.drawCentredString(self.width / 2, 2, "Inference reads .track, .nodes, and .arcs; ground-truth .route is not read.")


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25,
            leading=30, alignment=TA_LEFT, textColor=NAVY, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11,
            leading=15, textColor=MUTED, spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Section", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16,
            leading=20, textColor=NAVY, spaceBefore=4, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Subsection", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
            leading=14, textColor=NAVY, spaceBefore=5, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8,
            leading=12.2, textColor=INK, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5,
            leading=10, textColor=MUTED, spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "TableText", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5,
            leading=9, textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.2,
            leading=8.5, textColor=colors.white, alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9.3,
            leading=13, textColor=NAVY, alignment=TA_LEFT,
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontName="Courier", fontSize=8,
            leading=10.5, textColor=colors.HexColor("#263445"),
        ),
    }


def paragraph(text, style):
    return Paragraph(text, style)


def make_table(rows, widths, header=True, font_size=7.5, row_background=True):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    if row_background:
        start = 1 if header else 0
        for row in range(start, len(rows)):
            if (row - start) % 2 == 1:
                commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    table.setStyle(TableStyle(commands))
    return table


def on_page(canvas, doc):
    canvas.saveState()
    page_width, _page_height = letter
    canvas.setTitle("PathSnap - Technical Report")
    canvas.setAuthor("PathSnap Project Team")
    canvas.setSubject("Distributed map matching and route reconstruction implementation")
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(page_width - doc.rightMargin, 23, f"18 November 2026  |  Page {doc.page}")
    canvas.line(doc.leftMargin, 34, page_width - doc.rightMargin, 34)
    canvas.restoreState()


def build():
    if not ROUTE_FIGURE.is_file():
        raise FileNotFoundError(f"Route figure is missing: {ROUTE_FIGURE}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, leftMargin=48, rightMargin=48,
        topMargin=48, bottomMargin=46, title="PathSnap - Technical Report",
    )
    story = []

    # Page 1: overview and architecture.
    story.extend([
        Spacer(1, 0.16 * inch),
        paragraph("PathSnap", styles["title"]),
        paragraph("Distributed Map Matching and Route Reconstruction", styles["subtitle"]),
        paragraph("Technical report | 18 November 2026 | Selected model: Tuned Random Forest Classifier", styles["small"]),
        Spacer(1, 0.1 * inch),
        paragraph("Executive summary", styles["h1"]),
        paragraph(
            "PathSnap converts noisy GPS observations into directed road-arc sequences. The integrated application generates route-free Phase 1 candidates, scores them with the saved Phase 2 model, normalizes candidate scores within each observation, and decodes a connected sequence with directed HMM/Viterbi. A React/Leaflet dashboard consumes the FastAPI service, which also creates Folium maps.",
            styles["body"],
        ),
        paragraph(
            "Inference reads only <font name='Courier'>.track</font>, <font name='Courier'>.nodes</font>, and <font name='Courier'>.arcs</font>. It never loads the ground-truth <font name='Courier'>.route</font> file. Training labels and final evaluation remain separate workflows.",
            styles["body"],
        ),
    ])
    summary_rows = [
        [paragraph("Python tests", styles["table_header"]), paragraph("Frontend build", styles["table_header"]), paragraph("Live API", styles["table_header"]), paragraph("Route smoke", styles["table_header"])],
        [paragraph("41 passed", styles["callout"]), paragraph("Vite build passed", styles["callout"]), paragraph("Model + HMM ready", styles["callout"]), paragraph("2,503 / 2,503 observations", styles["callout"])],
    ]
    story.append(make_table(summary_rows, [129, 129, 129, 129], header=True, row_background=False))
    story.extend([
        Spacer(1, 0.12 * inch),
        paragraph("System architecture", styles["h1"]),
        PipelineDiagram(),
        Spacer(1, 0.08 * inch),
        paragraph("Major components", styles["h2"]),
    ])
    component_rows = [
        [paragraph("Component", styles["table_header"]), paragraph("Role", styles["table_header"])],
        [paragraph("Phase 1", styles["table"]), paragraph("Candidate generation and approved route-free feature engineering.", styles["table"])],
        [paragraph("Phase 2", styles["table"]), paragraph("Trip-grouped validation, model selection, optimization, and saved estimator artifacts.", styles["table"])],
        [paragraph("HMM / Viterbi", styles["table"]), paragraph("Directed arc transitions and route sequences with explicit reset warnings.", styles["table"])],
        [paragraph("FastAPI + React", styles["table"]), paragraph("Trip preview/upload, inference endpoints, diagnostics, Leaflet layers, and Folium map access.", styles["table"])],
    ]
    story.append(make_table(component_rows, [118, 398]))

    # Page 2: data and metric rankings.
    story.extend([PageBreak(), paragraph("Data and model evaluation", styles["h1"])])
    story.append(paragraph(
        "Training contains 1,870,717 candidate rows from 80 trips and 208,417 observations. Classifier fitting retains 178,534 observations with a ground-truth candidate (1,742,906 candidate rows). Five-fold GroupKFold separates trips before filtering; all learned preprocessing is fitted within each training fold.",
        styles["body"],
    ))
    story.append(paragraph(
        "Fourteen numeric features are approved. Trip ID, observation index, arc ID, target label, and ground-truth candidate flag are excluded from model inputs. Candidate miss rate is reported separately because the classifier cannot recover a road that candidate generation did not produce.",
        styles["body"],
    ))
    story.append(paragraph("Pooled out-of-fold rankings", styles["h2"]))
    header_labels = ["Model", "Macro<br/>F1", "Positive<br/>precision", "Candidate<br/>accuracy", "Top-1<br/>accuracy", "F1<br/>rank", "Precision<br/>rank", "Candidate<br/>accuracy rank", "Top-1<br/>rank"]
    metric_rows = [[paragraph(value, styles["table_header"]) for value in header_labels]]
    metric_data = [
        ("LightGBM", "0.8645", "0.7822", "0.9516", "0.7863", "1", "2", "1", "2"),
        ("Random Forest", "0.8544", "0.8161", "0.9507", "0.7897", "2", "1", "2", "1"),
        ("Shallow MLP", "0.8462", "0.7802", "0.9470", "0.7704", "3", "3", "3", "3"),
        ("Gaussian NB", "0.7851", "0.5332", "0.9071", "0.7489", "4", "5", "5", "5"),
        ("Logistic Regression", "0.7828", "0.7127", "0.9296", "0.7586", "5", "4", "4", "4"),
    ]
    metric_rows.extend([[paragraph(value, styles["table"]) for value in row] for row in metric_data])
    story.append(make_table(metric_rows, [82, 48, 58, 62, 53, 42, 57, 65, 49]))
    story.extend([
        Spacer(1, 0.08 * inch),
        paragraph(
            "Candidate binary accuracy evaluates candidate rows at a 0.5 threshold. Top-1 accuracy selects the highest-scored candidate for each eligible observation. LightGBM leads Macro F1 and candidate binary accuracy; Random Forest leads positive precision and top-1 accuracy.",
            styles["body"],
        ),
        paragraph("Selection and final evaluation", styles["h2"]),
        paragraph(
            "The selected baseline is Random Forest under a weighted ranking of top-1 accuracy, Macro F1, route continuity, log loss, fold stability, and run times. Five Optuna trials tuned log loss: mean CV Macro F1 moved from 0.8585 to 0.8609, and mean CV Log Loss from 0.1217 to 0.1195.",
            styles["body"],
        ),
    ])
    final_rows = [[paragraph("Student-test metric", styles["table_header"]), paragraph("Value", styles["table_header"])]]
    final_rows.extend([
        [paragraph(name, styles["table"]), paragraph(value, styles["table"])]
        for name, value in [
            ("Top-1 Accuracy", "0.8206"),
            ("Candidate Macro F1", "0.8792"),
            ("Candidate Log Loss", "0.0983"),
            ("Candidate Miss Rate", "0.0014"),
            ("Route Continuity", "0.2651"),
        ]
    ])
    story.append(make_table(final_rows, [300, 216]))
    story.append(Paragraph(
        "Student-test labels were used once for this final evaluation, not for selection or optimization. The student-test set contains 10 trips; results are descriptive and do not replace grouped cross-validation.",
        styles["small"],
    ))

    # Page 3: inference, decoder, and API.
    story.extend([PageBreak(), paragraph("Inference, decoder, and API", styles["h1"])])
    story.append(paragraph("Route-free inference", styles["h2"]))
    story.append(paragraph(
        "At startup, the API loads the saved estimator, preprocessing artifact, feature schema, metadata, and tuned hyperparameters once. It validates feature order and expected feature counts. The inference builder reads GPS and road graph files only, then produces the approved feature matrix without route alignment. Scores are normalized independently within each trip/observation candidate group for HMM emissions; they are not calibrated probabilities.",
        styles["body"],
    ))
    story.append(paragraph("Directed HMM / Viterbi", styles["h2"]))
    story.append(paragraph(
        "Viterbi operates in log space. A transition is allowed when the selected arc remains the same or the next arc directly follows it in the directed graph. Transition scores compare road distance with GPS displacement and elapsed time. Empty candidate groups, long gaps, or disconnected candidate sets split the sequence and add warnings; the decoder does not invent connecting arcs.",
        styles["body"],
    ))
    endpoint_rows = [[paragraph("Method and route", styles["table_header"]), paragraph("Purpose", styles["table_header"])]]
    endpoints = [
        ("GET /health", "Readiness of API, model, HMM, and data."),
        ("GET /api/model", "Selected model and actual saved schema metadata."),
        ("GET /api/trips", "Available route-free demo and uploaded trip IDs."),
        ("GET /api/trips/{id}", "GPS preview and graph counts; no ground-truth labels."),
        ("POST /api/trips/upload", "Validate one .track, .nodes, .arcs bundle; 25 MiB per file."),
        ("POST /api/map-match", "Build candidates, score, decode, and generate Folium HTML."),
        ("GET /api/trips/{id}/results", "Return the latest in-memory process result."),
        ("GET /api/trips/{id}/map", "Serve the generated Folium HTML map."),
    ]
    endpoint_rows.extend([[paragraph(route, styles["table"]), paragraph(purpose, styles["table"])] for route, purpose in endpoints])
    story.append(make_table(endpoint_rows, [208, 308]))
    story.extend([
        Spacer(1, 0.08 * inch),
        paragraph("The React dashboard loads health/model/trip data from the API, validates a trip bundle before upload, requests map matching, and displays GPS, candidate, classifier, and HMM layers. Accuracy/F1 against ground truth is not part of an inference response.", styles["body"]),
        paragraph("Failure behavior", styles["h2"]),
        paragraph("Missing artifacts leave health queryable but degraded; inference returns 503. Invalid trip data returns 422, unknown trips return 404, and files over the size limit return 413.", styles["body"]),
    ])

    # Page 4: live route smoke and static geometry figure.
    story.extend([PageBreak(), paragraph("Live API route smoke run", styles["h1"])])
    story.append(paragraph(
        "The service was started at 127.0.0.1:8000. Health reported the model and HMM ready, and the trip list contained 10 student-test IDs. Trip 00000000 was reconstructed using the route-free API path.",
        styles["body"],
    ))
    smoke_rows = [[paragraph("Measurement", styles["table_header"]), paragraph("Observed value", styles["table_header"])]]
    smoke_rows.extend([[paragraph(k, styles["table"]), paragraph(v, styles["table"])] for k, v in [
        ("GPS observations", "2,503"),
        ("Road nodes / directed arcs", "8,542 / 18,196"),
        ("Candidate rows scored", "13,705"),
        ("Observations with candidates", "2,503 / 2,503"),
        ("Classifier route continuity", "0.9492"),
        ("HMM route continuity", "1.0000"),
        ("HMM route distance", "52,473.7 m"),
        ("Decoder / processing time", "completed / 8.42 s"),
        ("Folium map response", "HTTP 200; HTML; 1,251,213 bytes"),
    ]])
    story.append(make_table(smoke_rows, [255, 261]))
    story.append(Spacer(1, 0.06 * inch))
    image = Image(str(ROUTE_FIGURE), width=7.05 * inch, height=4.60 * inch, kind="proportional")
    image.hAlign = "CENTER"
    story.append(image)
    story.append(Paragraph(
        "Figure 1. GPS observation trace and model-produced routes from the live API response. Local coordinates only; no basemap tiles were requested for this figure.",
        styles["small"],
    ))
    story.append(Paragraph(
        "This is one route-level smoke run, not a ground-truth accuracy measurement. The browser automation bridge was unavailable, so the interactive browser rendering was not visually inspected.",
        styles["small"],
    ))

    # Page 5: verification, deployment, and limitations.
    story.extend([PageBreak(), paragraph("Verification, deployment, and limits", styles["h1"])])
    checks = [
        "Python test suite: 41 passed in 12.76 seconds on the final recorded run.",
        "Frontend dependency installation: npm ci passed after synchronizing package-lock.json.",
        "Frontend lint: passed; production build: passed with Vite 8.3.8.",
        "Bundle: 391.50 kB JavaScript (118.48 kB gzip); 26.44 kB CSS (9.59 kB gzip).",
        "Live health, trip listing, map matching, result payload, and Folium response checked.",
        "Manual browser interaction and LAN connectivity remain deployment-machine checks.",
        "npm audit reports five high-severity Vite dependency advisories; its suggested fix is a major downgrade, so it was not applied without compatibility review.",
    ]
    for item in checks:
        story.append(Paragraph(item, styles["body"], bulletText="-"))
    story.append(paragraph("Two-laptop run", styles["h2"]))
    code = (
        "# Backend laptop, repository root\n"
        "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000\n\n"
        "# Frontend laptop, frontend directory\n"
        "npm ci\n"
        "npm run dev -- --host 0.0.0.0\n\n"
        "# Frontend .env.local\n"
        "VITE_API_BASE_URL=http://<backend-lan-ip>:8000"
    )
    code_table = Table([[PreformattedSafe(code, styles["code"])]], colWidths=[516])
    code_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(code_table)
    story.append(paragraph("Operational limits", styles["h2"]))
    limits = [
        "Candidate misses originate upstream and cannot be fixed by the classifier.",
        "The decoder only allows direct directed-arc transitions; gaps and disconnected sets remain visible as resets and warnings.",
        "Within-observation score normalization does not guarantee calibration under distribution shift.",
        "Student-test evaluation covers 10 trips and is descriptive.",
        "Interactive basemaps use OpenStreetMap tiles when opened; the static report figure was rendered locally without a basemap request.",
        "No commit or push was created for this implementation pass.",
    ]
    for item in limits:
        story.append(Paragraph(item, styles["body"], bulletText="-"))
    story.append(paragraph("Project references", styles["h2"]))
    story.append(paragraph(
        "README.md - setup and two-laptop deployment. docs/api-specification.md - endpoint contract. docs/PHASE2_ML_IMPLEMENTATION.md - model metrics and selection. docs/PROJECT_ARCHITECTURE_AUDIT.md - initial integration state. docs/PHASE2_DATASET_SCHEMA.md - feature/data schema.",
        styles["small"],
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(OUTPUT)


def PreformattedSafe(text, style):
    from reportlab.platypus import Preformatted

    return Preformatted(text, style, maxLineLength=92)


if __name__ == "__main__":
    build()
