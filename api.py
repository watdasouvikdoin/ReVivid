# api.py
import io
import json
import traceback
from typing import Tuple

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import cv2

# Import your functions from app.py (assumes app.py contains restore_image and enhance_image)
# If your file name is ReVivid_app.py change the import accordingly:
try:
    from app import restore_image, enhance_image, read_image_from_bytes
except Exception:
    # fallback: try the alternative name used earlier
    from ReVivid_app import restore_image, enhance_image, read_image_from_bytes

app = Flask("reVivid_api")
CORS(app, resources={r"/api/*": {"origins": "*"}})


def _read_image_from_request_file(file_storage) -> np.ndarray:
    """
    Read Werkzeug FileStorage -> OpenCV BGR image (uint8)
    """
    data = file_storage.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _image_to_response_bytes(img_bgr: np.ndarray, fmt="png") -> Tuple[bytes, str]:
    """Encode BGR image to bytes and return (bytes, mime-type)"""
    ok, buf = cv2.imencode(f".{fmt}", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes(), f"image/{fmt}"


@app.route("/api/restore", methods=["POST"])
def api_restore():
    """
    POST /api/restore
    multipart/form-data:
      - image: file
      - params: JSON string describing { mode:'auto'|'manual', steps: [ {name, params}, ... ] }

    Returns binary image (PNG) on success.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        f = request.files["image"]
        params_raw = request.form.get("params", "{}")
        params = json.loads(params_raw) if isinstance(params_raw, str) and params_raw else {}

        img = _read_image_from_request_file(f)
        if img is None:
            return jsonify({"error": "Could not decode input image"}), 400

        # Determine steps
        steps = params.get("steps", [])
        # If no steps and mode == auto, create default auto pipeline
        if not steps and params.get("mode", "") == "auto":
            strength = float(params.get("strength", 1.0)) if params.get("strength") else 1.0
            denoise_h = 8.0 * strength
            rl_iters = int(15 * strength)
            steps = [
                ("denoise", {"method": "Non-Local Means", "h": denoise_h}),
                ("deblur_rl", {"iterations": rl_iters, "psf_size": 9, "psf_sigma": 2.0}),
                ("inpaint_specks", {"min_area": 3, "max_area": 500})
            ]
        else:
            # ensure tuple structure expected by restore_image
            steps = [(s.get("name"), s.get("params", {})) for s in steps]

        out = restore_image(img, steps)

        data, mime = _image_to_response_bytes(out, fmt="png")
        return send_file(io.BytesIO(data), mimetype=mime, as_attachment=False, attachment_filename="restored.png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    """
    POST /api/enhance
    multipart/form-data:
      - image: file
      - params: JSON string describing { steps: [ {name, params}, ... ] }
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        f = request.files["image"]
        params_raw = request.form.get("params", "{}")
        params = json.loads(params_raw) if isinstance(params_raw, str) and params_raw else {}

        img = _read_image_from_request_file(f)
        if img is None:
            return jsonify({"error": "Could not decode input image"}), 400

        steps = params.get("steps", [])
        steps = [(s.get("name"), s.get("params", {})) for s in steps]

        out = enhance_image(img, steps)

        data, mime = _image_to_response_bytes(out, fmt="png")
        return send_file(io.BytesIO(data), mimetype=mime, as_attachment=False, attachment_filename="enhanced.png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ReVivid API"})


if __name__ == "__main__":
    # for development only: run flask dev server
    app.run(host="0.0.0.0", port=8000, debug=False)
