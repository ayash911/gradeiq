# ocr engine start
import os
import re
import cv2
import base64
import numpy as np
from PIL import Image, ImageEnhance
import torch
import warnings
import logging

# quiet mode
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from pillow_heif import register_heif_opener
register_heif_opener()

def load_env():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    k = k.strip()
                    if k not in os.environ:
                        os.environ[k] = v.strip().strip("'").strip('"')

load_env()

_ocr_predictor = None
_trocr_processor = None
_trocr_model = None
_device = None


def load_ocr_models(): # wake up the ocr brains
    global _ocr_predictor, _trocr_processor, _trocr_model, _device

    if _ocr_predictor is None:
        from doctr.models import ocr_predictor
        det = os.getenv('DET_ARCH', 'db_resnet50') # detection brain
        reco = os.getenv('RECO_ARCH', 'crnn_vgg16_bn') # recognition brain
        print(f"[ocr] Loading DocTR models ({det}/{reco})...")
        _ocr_predictor = ocr_predictor(
            det_arch=det,
            reco_arch=reco,
            pretrained=True
        )
        print("[ocr] DocTR loaded.")

    if _trocr_processor is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        model_name = os.getenv('TROCR_MODEL', 'microsoft/trocr-large-handwritten') # trocr brain
        print(f"[ocr] Loading TrOCR ({model_name})...")
        _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _trocr_model = _trocr_model.to(_device)
        _trocr_model.eval()
        print(f"[ocr] TrOCR loaded on {_device}.")

    return _ocr_predictor, _trocr_processor, _trocr_model


def scan_filter(img_np: np.ndarray) -> np.ndarray: # make the scan look pretty
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    sigma = int(os.getenv('SCAN_SIGMA', 51))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma) #gaussian blur ( remove shadows)
    shadow_free = cv2.divide(gray, bg, scale=255) # remove shadows

    block_size = int(os.getenv('SCAN_BLOCK_SIZE', 21))
    c_val = int(os.getenv('SCAN_C', 10))

    binary = cv2.adaptiveThreshold(
        shadow_free, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,                     
        C=c_val                                          
    ) # make it black and white (Adaptive Thresholding)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # scrub the noise (Morphological Operations)

    result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    return result


def numpy_to_base64(img_np: np.ndarray) -> str: # turn pixels into text for web
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(img_np)
    import io
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def detect_strikethrough(line_crop: np.ndarray) -> bool: # check for crossed out text
    if line_crop is None or line_crop.size == 0:
        return False

    h, w = line_crop.shape[:2]
    if h < 10 or w < 30:
        return False

    if len(line_crop.shape) == 3:
        gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_crop.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_width = max(int(w * 0.4), 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    margin = int(h * 0.3)
    middle_region = horizontal_lines[margin:h - margin, :]

    if middle_region.size == 0:
        return False

    line_pixels = cv2.countNonZero(middle_region)
    region_area = middle_region.shape[0] * middle_region.shape[1]

    if region_area == 0:
        return False

    strike_thresh = float(os.getenv('STRIKE_THRESHOLD', 0.2))
    return (line_pixels / region_area) > strike_thresh


def recognize_line(line_crop_pil: Image.Image) -> str: # read one line of text
    global _trocr_processor, _trocr_model, _device
    if line_crop_pil.mode != 'RGB':
        line_crop_pil = line_crop_pil.convert('RGB')

    pixel_values = _trocr_processor(line_crop_pil, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(_device)

    with torch.no_grad():
        generated_ids = _trocr_model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=5
        )

    text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def clean_ocr_text(text: str) -> str: # tidy up the ocr output
    text = re.sub(r'\b\d\b', '', text)
    text = re.sub(r'[.,;:!?()\[\]]{2,}', '.', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\.(\s*\.)+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^[^a-zA-Z]+', '', text)
    text = re.sub(r'[^a-zA-Z.]+$', '', text)
    return text.strip()


def extract_text_from_image(image_path: str, models=None, return_debug=False) -> dict: # run ocr on one image
    if models is None:
        ocr_pred, trocr_proc, trocr_mod = load_ocr_models()
    else:
        ocr_pred, trocr_proc, trocr_mod = models

    try:
        pil_img = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Could not read image {image_path}: {e}")

    img_h, img_w = img_np.shape[:2]

    scanned = scan_filter(img_np) # clean the scan
    scanned_bgr = cv2.cvtColor(scanned, cv2.COLOR_RGB2BGR)

    result = ocr_pred([scanned]) # find text blocks
    json_output = result.export()

    all_lines = []
    for page in json_output['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                all_words = line['words']
                if not all_words:
                    continue

                x_mins = [w['geometry'][0][0] for w in all_words]
                y_mins = [w['geometry'][0][1] for w in all_words]
                x_maxs = [w['geometry'][1][0] for w in all_words]
                y_maxs = [w['geometry'][1][1] for w in all_words]

                all_lines.append({
                    'xmin': min(x_mins), 'ymin': min(y_mins),
                    'xmax': max(x_maxs), 'ymax': max(y_maxs)
                })

    all_lines.sort(key=lambda l: l['ymin']) # sort lines top to bottom

    recognized_lines = []
    debug_data = []

    pad_x_val = float(os.getenv('BOX_PAD_X', 0.05))
    pad_y_val = float(os.getenv('BOX_PAD_Y', 0.15))

    for line_info in all_lines:
        xmin, ymin = line_info['xmin'], line_info['ymin']
        xmax, ymax = line_info['xmax'], line_info['ymax']

        pad_x = int((xmax - xmin) * img_w * pad_x_val)
        pad_y = int((ymax - ymin) * img_h * pad_y_val)

        x1 = max(0, int(xmin * img_w) - pad_x)
        y1 = max(0, int(ymin * img_h) - pad_y)
        x2 = min(img_w, int(xmax * img_w) + pad_x)
        y2 = min(img_h, int(ymax * img_h) + pad_y)

        line_crop_bgr = scanned_bgr[y1:y2, x1:x2]
        line_crop_rgb = scanned[y1:y2, x1:x2]

        if line_crop_bgr.size == 0:
            continue

        is_struck = detect_strikethrough(line_crop_bgr) # check if it's crossed out

        if not is_struck:
            line_pil = Image.fromarray(line_crop_rgb)
            text = recognize_line(line_pil) # read it!
            if text:
                recognized_lines.append(text)
                if return_debug:
                    debug_data.append({
                        "text": text,
                        "box": [xmin, ymin, xmax, ymax],
                        "is_struck": False
                    })
        elif return_debug:
            debug_data.append({
                "text": "[STRUCK]",
                "box": [xmin, ymin, xmax, ymax],
                "is_struck": True
            })

    full_text = ' '.join(recognized_lines).strip()
    full_text = clean_ocr_text(full_text)

    result_dict = {
        "text": full_text,
        "debug": debug_data if return_debug else []
    }

    if return_debug:
        result_dict["scanned_b64"] = numpy_to_base64(scanned)

    return result_dict


def process_answer_sheets(image_paths: list, models=None, return_debug=False) -> dict: # process all images
    if models is None:
        models = load_ocr_models()

    results = {}
    for i, path in enumerate(image_paths, 1):
        filename = os.path.splitext(os.path.basename(path))[0]
        print(f"[ocr] Processing {i}/{len(image_paths)}: {filename}")
        try:
            res = extract_text_from_image(path, models, return_debug=return_debug)
            results[filename] = res
            print(f"  → {len(res['text'].split())} words: {res['text'][:80]}...")
        except Exception as e:
            print(f"  → ERROR: {e}")
            results[filename] = {"text": f"[OCR ERROR: {str(e)}]", "debug": []}
    return results


if __name__ == "__main__": # test it out
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path> [image_path2 ...]")
        sys.exit(1)
    paths = sys.argv[1:]
    models = load_ocr_models()
    for path in paths:
        print(f"\n{'='*50}")
        print(f"File: {path}")
        print('='*50)
        res = extract_text_from_image(path, models)
        print(f"\nExtracted text:\n{res['text']}")
        print(f"\nWord count: {len(res['text'].split())}")
