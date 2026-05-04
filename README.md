# GradeIQ: Smart Automated Grading

GradeIQ is a computer vision and natural language processing application designed to automate the grading of handwritten student assessments. It scans physical answer sheets, reads the handwriting, and assigns grades by comparing student responses against a reference key.

### The Core Problem It Solves

Manual grading is time-consuming and tedious. GradeIQ accelerates this process by handling the initial evaluation. Unlike basic keyword-matching tools, GradeIQ evaluates the meaning of an answer. If a student explains a concept correctly using different phrasing than the answer key, the system can still recognize the correct logic and award points.

---

### How It Works

The process is designed to be straightforward:

1. **Upload**: Provide a ZIP file containing images or scans of student answer sheets.
2. **Pre-process**: The system cleans the images, removing shadows and adjusting contrast to isolate the handwriting.
3. **Extract Text**: The vision engine identifies text blocks and converts the handwriting into digital text.
4. **Evaluate Logic**: The grading engine compares the extracted text against your reference answer. It checks for semantic similarity (meaning) and logical consistency (ensuring there are no direct contradictions).
5. **Review Results**: The system outputs a detailed report with suggested scores, which can be exported as a CSV.

---

### Technical Architecture

For developers and technical users, GradeIQ relies on a modular pipeline.

#### The Data Flow

```mermaid
graph LR
    A[Raw Scans] --> B[Image Filter]
    B --> C[OCR Engine]
    C --> D[Semantic Grader]
    D --> E[Scoring Output]
```

#### System Components

```mermaid
graph TD
    User((User)) --> Dashboard[Web Interface]
    Dashboard --> Server[Flask API]

    subgraph "Machine Learning Engines"
    Server --> Vision[Vision Pipeline]
    Vision --> Logic[NLP Pipeline]
    end

    Vision -- Extracted Text --> Logic
    Logic -- Evaluated Scores --> Server
    Server --> Dashboard
```

---

### Technology Stack

The application is built using established machine learning models and web frameworks:

- **Backend Application**: Python, Flask
- **Document Processing**: OpenCV, Pillow, Pillow-Heif
- **Optical Character Recognition (OCR)**: Microsoft TrOCR and DocTR
- **Natural Language Processing (NLP)**: DeBERTa-v3 and LaBSE via Sentence-Transformers
- **Frontend Interface**: HTML, CSS, and Vanilla JavaScript

---

## Setup and Configuration

### Local Installation

1.  **Clone the repository**: Download the code to your local machine.
2.  **Environment Setup**: Create a `.env` file based on `.env.example` and add your `NGROK_TOKEN`.
3.  **Install Dependencies**: Run `pip install -r requirements.txt`.
4.  **Run the App**: Start the server using `python app.py`.

### Customization Options

You can adjust the system behavior via the `.env` file:

- **Grading Strictness**: Modify similarity thresholds for scoring.
- **Image Processing**: Adjust contrast and shadow-removal intensity.
- **Model Selection**: Change the HuggingFace model IDs for OCR or NLP.

---

## Running in Google Colab

Google Colab is recommended for its free GPU access, which significantly speeds up the OCR and grading process.

1. **Open a new Notebook**: Go to [Google Colab](https://colab.research.google.com/).
2. **Switch to GPU**: Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. **Run the following cell**:
   ```python
   # 1. Clone the project
   !git clone https://github.com/ayash911/gradeiq.git
   %cd gradeiq

   # 2. Install dependencies
   !pip install -r requirements.txt -q

   # 3. Start the server (Replace NGROK_TOKEN with your actual token)
   !python3 app.py \
     --ngrok "NGROK_TOKEN" \
     --domain "NGROK_DOMAIN" \
     --port 5000 \
     --debug True
   ```
4. **Access the UI**: Click the **Public URL** (ngrok link) in the output to open the GradeIQ dashboard.

> [!NOTE]
> **Cold Start**: The first grading request will download several GBs of models. This takes 2-5 minutes but only happens once per session.

---

## CLI Reference

You can now configure the system directly via command-line arguments. These override any settings in your `.env` file.

| Parameter | Description |
| :--- | :--- |
| `--ngrok` | Your Ngrok Auth Token |
| `--domain` | Static Ngrok domain (e.g., `myapp.ngrok-free.dev`) |
| `--port` | Server port (default: 5000) |
| `--debug` | Enable Flask debug mode (True/False) |
| `--det` | OCR detection model (default: `db_resnet50`) |
| `--reco` | OCR recognition model (default: `crnn_vgg16_bn`) |
| `--trocr` | TrOCR model name |
| `--labse` | Sentence similarity model (default: `LaBSE`) |
| `--nli` | NLI model for logic verification |

For a full list of all available parameters, run:
```bash
python3 app.py --help
```
