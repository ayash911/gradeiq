# grading engine start
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
import re
import os
import warnings
import logging

# quiet mode
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def load_env(): # simple env loader
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    k = k.strip()
                    if k not in os.environ:
                        os.environ[k] = v.strip().strip("'").strip('"')

load_env() # load the secrets

labse_name = os.getenv('LABSE_MODEL', 'LaBSE') # labse brain
print(f"[grader] Loading LaBSE model ({labse_name})...")
labse = SentenceTransformer(labse_name)
nli_name = os.getenv('NLI_MODEL', 'cross-encoder/nli-deberta-v3-small') # nli brain
print(f"[grader] Loading NLI model ({nli_name})...")
nli = pipeline("text-classification", model=nli_name)
print("[grader] Models loaded.")


def clean_for_grading(text: str) -> str: # wash the text
    text = re.sub(r'[^a-zA-Z0-9.,;:!?\' -]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def text_quality_score(text: str) -> float: # check if ocr did okay
    words = text.split()
    if len(words) == 0:
        return 0.0
    good = sum(1 for w in words if len(w) >= 2 and sum(c.isalpha() for c in w) / len(w) > 0.7)
    return good / len(words)


def extract_key_sentences(text: str, n: int = 3) -> str: # grab the best bits
    text = clean_for_grading(text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 2]
    if not sentences:
        return text
    return ' '.join(sentences[:n])


def split_sentences(text: str) -> list: # chop into sentences
    text = clean_for_grading(text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = [s.strip() for s in sentences if len(s.split()) >= 3]
    if not result and text.strip():
        result = [text.strip()]
    return result


def check_nli_long(model_answer: str, student_answer: str) -> dict: # logic check!
    short_model = ' '.join(model_answer.split()[:60])
    sentences = split_sentences(student_answer)
    sentences = sentences[:6]                   

    contradiction_count = 0
    entailment_count = 0
    total = 0
    sentence_results = []

    c_thresh = float(os.getenv('NLI_CONTRADICTION_THRESHOLD', 0.70))
    e_thresh = float(os.getenv('NLI_ENTAILMENT_THRESHOLD', 0.65))

    for sentence in sentences:
        result = nli({"text": short_model, "text_pair": sentence}) # compare pairs
        if isinstance(result, list):
            result = result[0]

        label = result['label'].upper()
        score = result['score']
        sentence_results.append({
            "sentence": sentence[:60],
            "label": label,
            "score": round(score, 3)
        })

        if label == 'CONTRADICTION' and score > c_thresh:
            contradiction_count += 1
        elif label == 'ENTAILMENT' and score > e_thresh:
            entailment_count += 1
        total += 1

    if total == 0:
        return {
            "verdict": "NEUTRAL",
            "contradiction_count": 0,
            "entailment_count": 0,
            "total_sentences": 0,
            "sentence_results": []
        }

    contradiction_ratio = contradiction_count / total
    entailment_ratio = entailment_count / total

    c_ratio = float(os.getenv('NLI_CONTRADICTION_RATIO', 0.3))
    e_ratio = float(os.getenv('NLI_ENTAILMENT_RATIO', 0.5))
    p_ratio = float(os.getenv('NLI_PARTIAL_RATIO', 0.15))

    if contradiction_ratio > c_ratio:
        verdict = "CONTRADICTION"
    elif entailment_ratio > e_ratio:
        verdict = "CORRECT"
    elif entailment_ratio > p_ratio:
        verdict = "PARTIAL"
    else:
        verdict = "NEUTRAL"

    return {
        "verdict": verdict,
        "contradiction_count": contradiction_count,
        "entailment_count": entailment_count,
        "total_sentences": total,
        "sentence_results": sentence_results
    }


def combine_signals(similarity: float, nli_label: str, max_marks: int,
                    text_quality: float = 1.0) -> float: # mix the ingredients
    if nli_label == "CONTRADICTION":
        return 0.0

    if nli_label == "CORRECT":
        high = float(os.getenv('SIMILARITY_CORRECT_HIGH', 0.75))
        med = float(os.getenv('SIMILARITY_CORRECT_MED', 0.50))
        low = float(os.getenv('SIMILARITY_CORRECT_LOW', 0.30))

        if similarity >= high:
            return float(max_marks)
        elif similarity >= med:
            return round(max_marks * 0.90, 1)
        elif similarity >= low:
            return round(max_marks * 0.80, 1)
        else:
            return round(max_marks * 0.65, 1)

    if nli_label == "PARTIAL":
        if similarity > 0.65:
            return round(max_marks * 0.80, 1)
        elif similarity > 0.40:
            return round(max_marks * 0.65, 1)
        elif similarity > 0.20:
            return round(max_marks * 0.50, 1)
        else:
            return round(max_marks * 0.35, 1)

    if nli_label == "NEUTRAL":
        sim_min = float(os.getenv('SIMILARITY_MIN_THRESHOLD', 0.27))
        sim_max = float(os.getenv('SIMILARITY_MAX_THRESHOLD', 0.70))

        if text_quality < 0.5:
            if similarity > 0.50:
                return round(max_marks * 0.6, 1)
            elif similarity > 0.30:
                return round(max_marks * 0.3, 1)
            else:
                return 0.0
        else:
            if similarity <= sim_min:
                return 0.0
            elif similarity > sim_max:
                return float(max_marks)
            else:
                scaled = (similarity - sim_min) / (sim_max - sim_min)
                return round(scaled * max_marks, 1)

    return 0.0


def get_confidence(similarity: float, nli_label: str) -> str: # how sure are we?
    sim_min = float(os.getenv('SIMILARITY_MIN_THRESHOLD', 0.27))
    sim_max = float(os.getenv('SIMILARITY_MAX_THRESHOLD', 0.70))

    if nli_label == "CONTRADICTION":
        return "HIGH — contradiction detected"
    if nli_label == "CORRECT":
        return "HIGH — entailment confirmed"
    if nli_label == "PARTIAL":
        return "MEDIUM — partially correct, review recommended"
    if similarity > sim_max or similarity < sim_min:
        return "MEDIUM — NLI uncertain, similarity used"
    return "LOW — manual checking needed"


def grade_answer(model_answer: str, student_answer: str, max_marks: int) -> dict: # score one answer
    clean_model = clean_for_grading(model_answer)
    clean_student = clean_for_grading(student_answer)
    quality = text_quality_score(clean_student)
    model_key = extract_key_sentences(clean_model, 3)
    student_key = extract_key_sentences(clean_student, 3)

    model_vec = labse.encode([model_key])
    student_vec = labse.encode([student_key])
    similarity = float(cosine_similarity(model_vec, student_vec)[0][0])

    nli_result = check_nli_long(clean_model, clean_student)
    nli_label = nli_result["verdict"]

    final_score = combine_signals(similarity, nli_label, max_marks, quality)

    return {
        "similarity": round(similarity, 3),
        "nli_label": nli_label,
        "nli_detail": nli_result,
        "suggested_marks": final_score,
        "max_marks": max_marks,
        "confidence": get_confidence(similarity, nli_label)
    }


def grade_entire_class(model_answer: str, student_answers: dict, max_marks: int) -> list: # grade them all!
    results = []
    for student_name, answer in student_answers.items():
        print(f"  Grading {student_name}...")
        result = grade_answer(model_answer, answer, max_marks)
        result["student"] = student_name
        result["answer_given"] = answer.strip()
        results.append(result)

    results.sort(key=lambda x: x["suggested_marks"], reverse=True)
    return results


def print_results(results: list): # show the scores
    print("\n" + "=" * 60)
    print("GRADING RESULTS")
    print("=" * 60)

    for r in results:
        print(f"\n{r['student']}: {r['suggested_marks']:.1f}/{r['max_marks']}")
        print(f"  Similarity  : {r['similarity']}")
        print(f"  NLI verdict : {r['nli_label']}")
        print(f"  Contradictions found : {r['nli_detail']['contradiction_count']}/{r['nli_detail']['total_sentences']} sentences")
        print(f"  Entailments found    : {r['nli_detail']['entailment_count']}/{r['nli_detail']['total_sentences']} sentences")
        print(f"  Confidence  : {r['confidence']}")

