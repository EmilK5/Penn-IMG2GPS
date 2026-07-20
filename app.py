import csv
import html
import importlib.util
import math
import os
import sys
import time
import traceback
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError


# Configure this to point at the checkpoint you want to test.
MODEL_CHECKPOINT = "final_224_model.pt"

# Use the current model architecture/export wrapper.
MODEL_FILE_CANDIDATES = ("model.py",)

# These names match the current model.py. If your architecture file changes,
# this is the main place to adjust the factory/class names.
MODEL_FACTORY_NAME = "get_model"
MODEL_CLASS_NAME = "Model"
UTM_ZONE = 18

# preprocess.py currently exposes prepare_data(csv_path). If you later add a
# single-image function, add its name here and it will be used automatically.
PREPROCESS_SINGLE_IMAGE_FUNCTIONS = (
    "preprocess_image",
    "preprocess_pil_image",
    "prepare_image",
    "transform_image",
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".heic", ".heif"}
_MODEL_CACHE = None


try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_STATUS = "HEIF/HEIC support enabled through pillow-heif."
except Exception:
    HEIF_STATUS = (
        "HEIF/HEIC support is not active. Install pillow-heif if HEIC files "
        "do not open correctly."
    )


def _import_external_gradio():
    """Import installed Gradio even if this folder contains gradio.py."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    original_path = list(sys.path)
    original_module = sys.modules.get("gradio")

    try:
        local_module = getattr(original_module, "__file__", "") if original_module else ""
        if local_module and os.path.abspath(local_module).startswith(script_dir):
            del sys.modules["gradio"]

        sys.path = [
            path
            for path in sys.path
            if os.path.abspath(path or os.getcwd()) != script_dir
        ]
        import gradio as gr

        return gr
    finally:
        sys.path = original_path


gr = _import_external_gradio()


def project_path(*parts: str) -> str:
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), *parts)


def load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_existing_file(candidates: Tuple[str, ...]) -> str:
    for candidate in candidates:
        path = project_path(candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any of: {', '.join(candidates)}")


def rational_to_float(value: Any) -> float:
    if isinstance(value, Fraction):
        return float(value)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def dms_to_decimal(dms: Any, ref: Any) -> float:
    degrees = rational_to_float(dms[0])
    minutes = rational_to_float(dms[1])
    seconds = rational_to_float(dms[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    ref_text = str(ref).upper()
    if ref_text in {"S", "W"}:
        decimal *= -1
    return decimal


def extract_gps_from_exif(image_path: str) -> Tuple[Optional[Tuple[float, float]], str]:
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            if not exif:
                return None, "No EXIF metadata found in this image."

            gps_tag = next(
                tag for tag, name in ExifTags.TAGS.items() if name == "GPSInfo"
            )
            try:
                gps_info = exif.get_ifd(gps_tag)
            except Exception:
                gps_info = exif.get(gps_tag)

            if not gps_info:
                return None, "EXIF metadata was found, but it does not include GPS data."

            gps: Dict[str, Any] = {
                ExifTags.GPSTAGS.get(key, key): value
                for key, value in dict(gps_info).items()
            }

            required = (
                "GPSLatitude",
                "GPSLatitudeRef",
                "GPSLongitude",
                "GPSLongitudeRef",
            )
            missing = [name for name in required if name not in gps]
            if missing:
                return None, f"GPS metadata is incomplete. Missing: {', '.join(missing)}."

            latitude = dms_to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
            longitude = dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])
            return (latitude, longitude), f"{latitude:.8f}, {longitude:.8f}"
    except UnidentifiedImageError:
        return None, "Unsupported or unreadable image format."
    except Exception as exc:
        return None, f"Could not read EXIF GPS metadata: {exc}"


def haversine_meters(
    point_a: Tuple[float, float], point_b: Tuple[float, float]
) -> float:
    lat1, lon1 = point_a
    lat2, lon2 = point_b
    radius_m = 6_371_000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_m * c


def clean_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def instantiate_model(model_module):
    if hasattr(model_module, MODEL_FACTORY_NAME):
        return getattr(model_module, MODEL_FACTORY_NAME)()
    if hasattr(model_module, MODEL_CLASS_NAME):
        return getattr(model_module, MODEL_CLASS_NAME)()
    if hasattr(model_module, "EfficientNetGPS"):
        return getattr(model_module, "EfficientNetGPS")()
    raise AttributeError(
        "Could not find a model constructor. Adjust MODEL_FACTORY_NAME or "
        "MODEL_CLASS_NAME near the top of app.py."
    )


def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch could not be imported: {exc}") from exc

    checkpoint_path = project_path(MODEL_CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Update MODEL_CHECKPOINT near the top of app.py."
        )

    model_file = find_existing_file(MODEL_FILE_CANDIDATES)
    model_module = load_module_from_file("img2gps_model_module", model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate_model(model_module)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    elif hasattr(checkpoint, "state_dict"):
        model = checkpoint
        state_dict = None
    else:
        raise TypeError("Unsupported checkpoint format.")

    if state_dict is not None:
        state_dict = clean_state_dict_keys(state_dict)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint keys do not exactly match model.py. This usually means "
                "the wrong model file or checkpoint is being used."
            ) from exc

    model.to(device)
    model.eval()
    _MODEL_CACHE = (model, device)
    return _MODEL_CACHE


def preprocess_image(image_path: str, ground_truth: Optional[Tuple[float, float]]):
    preprocess_path = project_path("preprocess.py")
    if not os.path.exists(preprocess_path):
        raise FileNotFoundError("preprocess.py was not found in the project directory.")

    preprocess_module = load_module_from_file("img2gps_preprocess_module", preprocess_path)

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch could not be imported: {exc}") from exc

    with Image.open(image_path) as image:
        pil_rgb = ImageOps.exif_transpose(image).convert("RGB")

    for function_name in PREPROCESS_SINGLE_IMAGE_FUNCTIONS:
        function = getattr(preprocess_module, function_name, None)
        if function is None:
            continue
        output = function(pil_rgb)
        if isinstance(output, torch.Tensor):
            return output.unsqueeze(0) if output.ndim == 3 else output

    if not hasattr(preprocess_module, "prepare_data"):
        raise AttributeError(
            "preprocess.py does not expose prepare_data or a recognized "
            "single-image preprocessing function. Add a function name to "
            "PREPROCESS_SINGLE_IMAGE_FUNCTIONS if needed."
        )

    lat, lon = ground_truth if ground_truth is not None else (0.0, 0.0)
    temp_csv_path = project_path(".img2gps_single_image_input.csv")
    try:
        with open(temp_csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=("file_name", "Latitude", "Longitude")
            )
            writer.writeheader()
            writer.writerow(
                {
                    "file_name": os.path.abspath(image_path),
                    "Latitude": lat,
                    "Longitude": lon,
                }
            )
        inputs, _ = preprocess_module.prepare_data(temp_csv_path)
    finally:
        try:
            os.remove(temp_csv_path)
        except OSError:
            pass

    return inputs


def model_output_to_latlon_exact(model, inputs):
    """Match notebook evaluation: local UTM meters -> global UTM -> lat/lon."""
    if not hasattr(model, "base_model") or not hasattr(model, "origin"):
        return None

    try:
        from pyproj import Proj
    except Exception as exc:
        raise RuntimeError(
            "pyproj is required for exact UTM-to-lat/lon conversion. "
            "Install it with `pip install pyproj`."
        ) from exc

    local_meters = model.base_model(inputs)
    if local_meters.ndim != 2 or local_meters.shape[1] < 2:
        raise RuntimeError("Base model output did not contain easting/northing meters.")

    lat_orig, lon_orig = model.origin.detach().cpu().double().tolist()[:2]
    utm_proj = Proj(proj="utm", zone=UTM_ZONE, ellps="WGS84", preserve_units=False)
    e_min, n_min = utm_proj(lon_orig, lat_orig)

    dx_easting = float(local_meters[0, 0].detach().cpu())
    dy_northing = float(local_meters[0, 1].detach().cpu())
    pred_lon, pred_lat = utm_proj(
        e_min + dx_easting,
        n_min + dy_northing,
        inverse=True,
    )
    return float(pred_lat), float(pred_lon)


def model_output_to_latlon_fallback(model, inputs):
    output = model(inputs)
    if isinstance(output, (list, tuple)):
        import torch

        output = torch.as_tensor(output)
    output = output.detach().cpu().float()
    values = output.reshape(-1).tolist()
    if len(values) < 2:
        raise RuntimeError("Model output did not contain latitude and longitude.")

    latitude, longitude = float(values[0]), float(values[1])
    return latitude, longitude


def predict_coordinates(
    image_path: str, ground_truth: Optional[Tuple[float, float]]
) -> Tuple[Optional[Tuple[float, float]], str]:
    try:
        import torch
    except Exception as exc:
        return None, f"PyTorch could not be imported: {exc}"

    try:
        model, device = load_model()
        inputs = preprocess_image(image_path, ground_truth).to(device)

        with torch.no_grad():
            exact_prediction = model_output_to_latlon_exact(model, inputs)
            if exact_prediction is not None:
                latitude, longitude = exact_prediction
                conversion_note = "UTM conversion"
            else:
                latitude, longitude = model_output_to_latlon_fallback(model, inputs)
                conversion_note = "model forward conversion"

        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            return None, "Model prediction was not finite."
        return (
            (latitude, longitude),
            f"{latitude:.8f}, {longitude:.8f} ({conversion_note})",
        )
    except Exception as exc:
        print(traceback.format_exc())
        return None, f"Inference failed: {exc}"


def marker_popup(label: str, point: Tuple[float, float]) -> str:
    return f"{label}<br>{point[0]:.8f}, {point[1]:.8f}"


def build_map_html(
    ground_truth: Optional[Tuple[float, float]],
    prediction: Optional[Tuple[float, float]],
) -> str:
    points = [point for point in (ground_truth, prediction) if point is not None]
    if not points:
        return "<div class='empty-map'>No coordinates available to map yet.</div>"

    try:
        import folium
    except Exception as exc:
        return (
            "<div class='empty-map'>Install folium to display maps. "
            f"Import error: {html.escape(str(exc))}</div>"
        )

    center_lat = sum(point[0] for point in points) / len(points)
    center_lon = sum(point[1] for point in points) / len(points)
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=16)

    if ground_truth is not None:
        folium.Marker(
            location=list(ground_truth),
            popup=marker_popup("Ground truth EXIF GPS", ground_truth),
            tooltip="Ground truth EXIF GPS",
            icon=folium.Icon(color="green", icon="screenshot"),
        ).add_to(fmap)

    if prediction is not None:
        folium.Marker(
            location=list(prediction),
            popup=marker_popup("Model prediction", prediction),
            tooltip="Model prediction",
            icon=folium.Icon(color="red", icon="flag"),
        ).add_to(fmap)

    if ground_truth is not None and prediction is not None:
        folium.PolyLine(
            [list(ground_truth), list(prediction)],
            color="#2563eb",
            weight=4,
            opacity=0.85,
            tooltip="Prediction error",
        ).add_to(fmap)
        bounds = [list(ground_truth), list(prediction)]
        fmap.fit_bounds(bounds, padding=(35, 35))

    return fmap._repr_html_()


def load_display_image(image_path: str):
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def make_confetti_signal(should_confetti: bool, confetti_shape: Optional[str] = None) -> str:
    run_id = str(time.time_ns())
    confetti_value = "1" if should_confetti else "0"
    shape = html.escape((confetti_shape or "").strip(), quote=True)
    return (
        "<div id='img2gps-confetti-signal' "
        f"data-confetti='{confetti_value}' "
        f"data-run-id='{run_id}' "
        f"data-shape='{shape}'></div>"
    )


def run_app(image_path: Optional[str], confetti_shape: Optional[str]):
    if not image_path:
        return (
            None,
            "Upload an image first.",
            "No prediction yet.",
            "No distance calculated.",
            build_map_html(None, None),
            "Waiting for an uploaded image.",
            make_confetti_signal(False, confetti_shape),
        )

    extension = os.path.splitext(str(image_path).lower())[1]
    if extension not in SUPPORTED_EXTENSIONS:
        return (
            None,
            f"Unsupported file extension '{extension}'. Please upload JPG, JPEG, HEIC, or HEIF.",
            "No prediction.",
            "No distance calculated.",
            build_map_html(None, None),
            "Unsupported image format.",
            make_confetti_signal(False, confetti_shape),
        )

    try:
        display_image = load_display_image(image_path)
    except Exception as exc:
        return (
            None,
            f"Could not open image: {exc}",
            "No prediction.",
            "No distance calculated.",
            build_map_html(None, None),
            "Image loading failed.",
            make_confetti_signal(False, confetti_shape),
        )

    ground_truth, exif_message = extract_gps_from_exif(image_path)
    prediction, prediction_message = predict_coordinates(image_path, ground_truth)

    if ground_truth is not None and prediction is not None:
        error_meters = haversine_meters(ground_truth, prediction)
        error_message = f"{error_meters:,.2f} meters"
        should_confetti = error_meters < 25.0
    else:
        error_meters = None
        should_confetti = False
        if ground_truth is None:
            error_message = "Distance not calculated because EXIF GPS coordinates are missing."
        else:
            error_message = "Distance not calculated because model prediction is unavailable."

    status_parts = [HEIF_STATUS]
    if error_meters is not None and should_confetti:
        status_parts.append("Great prediction: error is under 25 meters.")
    elif error_meters is not None:
        status_parts.append("Inference completed successfully.")
    else:
        status_parts.append("Completed with missing coordinate information.")

    return (
        display_image,
        exif_message,
        prediction_message,
        error_message,
        build_map_html(ground_truth, prediction),
        " ".join(status_parts),
        make_confetti_signal(should_confetti, confetti_shape),
    )


CONFETTI_HEAD = """
<style>
  .img2gps-title { margin-bottom: 0.25rem; }
  .img2gps-help { color: #475569; margin-top: 0; }
  .empty-map {
    align-items: center;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #475569;
    display: flex;
    height: 320px;
    justify-content: center;
  }
  #img2gps-confetti-layer {
    inset: 0;
    pointer-events: none;
    position: fixed;
    z-index: 999999;
  }
  .img2gps-confetti-piece {
    left: 0;
    line-height: 1;
    position: absolute;
    top: 0;
    transform: translate3d(0, 0, 0);
    user-select: none;
    will-change: opacity, transform;
  }
  .img2gps-confetti-rect {
    border-radius: 2px;
    display: block;
  }
  #img2gps-confetti-signal {
    height: 0;
    overflow: hidden;
    width: 0;
  }
</style>
<script>
(function () {
  let lastRunId = null;
  let activeCleanup = null;

  function removeExistingConfetti() {
    const oldLayer = document.getElementById("img2gps-confetti-layer");
    if (oldLayer) oldLayer.remove();
  }

  function launchConfetti(shape) {
    if (activeCleanup) activeCleanup();
    removeExistingConfetti();

    const layer = document.createElement("div");
    layer.id = "img2gps-confetti-layer";
    document.body.appendChild(layer);

    const colors = ["#16a34a", "#2563eb", "#f59e0b", "#ef4444", "#9333ea"];
    const duration = 10000;
    const start = performance.now();
    const emojiShape = (shape || "").trim();
    const timers = [];

    function addTimer(timerId, clearFn) {
      timers.push({ id: timerId, clear: clearFn });
    }

    function addParticle() {
      if (!document.body.contains(layer)) return;

      const piece = document.createElement("span");
      piece.className = "img2gps-confetti-piece";

      const startX = Math.random() * window.innerWidth;
      const startY = -40 - Math.random() * 120;
      const drift = -180 + Math.random() * 360;
      const endX = startX + drift;
      const endY = window.innerHeight + 90 + Math.random() * 180;
      const rotationStart = Math.random() * 360;
      const rotationEnd = rotationStart + 360 + Math.random() * 720;
      const fallDuration = 4200 + Math.random() * 3200;

      piece.style.opacity = "1";
      piece.style.transform =
        `translate3d(${startX}px, ${startY}px, 0) rotate(${rotationStart}deg)`;
      piece.style.webkitTransform = piece.style.transform;

      if (emojiShape) {
        piece.textContent = emojiShape;
        piece.style.fontFamily =
          '"Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
        piece.style.fontSize = `${22 + Math.random() * 18}px`;
      } else {
        const rect = document.createElement("span");
        rect.className = "img2gps-confetti-rect";
        rect.style.background = colors[Math.floor(Math.random() * colors.length)];
        rect.style.height = `${6 + Math.random() * 8}px`;
        rect.style.width = `${5 + Math.random() * 12}px`;
        piece.appendChild(rect);
      }

      layer.appendChild(piece);

      requestAnimationFrame(function () {
        piece.style.transition =
          `transform ${fallDuration}ms cubic-bezier(0.16, 0.7, 0.35, 1), ` +
          `opacity ${fallDuration}ms linear`;
        piece.style.transform =
          `translate3d(${endX}px, ${endY}px, 0) rotate(${rotationEnd}deg)`;
        piece.style.webkitTransform = piece.style.transform;
        piece.style.opacity = "0";
      });

      addTimer(
        window.setTimeout(function () {
          piece.remove();
        }, fallDuration + 250),
        window.clearTimeout
      );
    }

    function burst(count) {
      for (let i = 0; i < count; i++) {
        addParticle();
      }
    }

    burst(80);
    const emitter = window.setInterval(function () {
      if (performance.now() - start >= duration) {
        window.clearInterval(emitter);
        return;
      }
      burst(8);
    }, 90);
    addTimer(emitter, window.clearInterval);

    const removalTimer = window.setTimeout(function () {
      layer.remove();
      if (activeCleanup === cleanup) activeCleanup = null;
    }, duration + 8000);
    addTimer(removalTimer, window.clearTimeout);

    function cleanup() {
      timers.forEach(function (timer) {
        timer.clear(timer.id);
      });
      layer.remove();
      if (activeCleanup === cleanup) activeCleanup = null;
    }

    activeCleanup = cleanup;
  }

  function scanForSignal() {
    const signals = document.querySelectorAll("#img2gps-confetti-signal");
    const signal = signals.length ? signals[signals.length - 1] : null;
    if (!signal) return;
    const runId = signal.getAttribute("data-run-id");
    const shouldRun = signal.getAttribute("data-confetti") === "1";
    const shape = signal.getAttribute("data-shape") || "";
    if (shouldRun && runId && runId !== lastRunId) {
      lastRunId = runId;
      launchConfetti(shape);
    }
  }

  function initConfettiWatcher() {
    const observer = new MutationObserver(scanForSignal);
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ["data-confetti", "data-run-id", "data-shape"],
      childList: true,
      subtree: true
    });
    window.setInterval(scanForSignal, 500);
    scanForSignal();
  }

  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", initConfettiWatcher);
  } else {
    initConfettiWatcher();
  }
})();
</script>
"""


with gr.Blocks(title="img2gps Tester", head=CONFETTI_HEAD) as demo:
    gr.Markdown(
        """
        <h1 class="img2gps-title">img2gps Model Tester</h1>
        <p class="img2gps-help">
        Upload a JPEG/JPG or HEIF/HEIC image. The app reads EXIF GPS as ground truth,
        runs the model, calculates haversine error, and maps both points.
        </p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(
                label="Upload image",
                file_types=["image", ".jpg", ".jpeg", ".heic", ".heif"],
                type="filepath",
            )
            confetti_shape = gr.Textbox(
                label="Emoji",
                value="🥟",
                placeholder="Try 🎉, ⭐, 📍, or ✨",
                max_lines=1,
            )
            run_button = gr.Button("Run img2gps", variant="primary")
            status_output = gr.Textbox(label="Status", lines=3)
        with gr.Column(scale=1):
            image_output = gr.Image(label="Uploaded image", type="pil")

    with gr.Row():
        exif_output = gr.Textbox(label="Ground-truth EXIF GPS", lines=2)
        prediction_output = gr.Textbox(label="Model-predicted coordinates", lines=2)
        error_output = gr.Textbox(label="Distance error", lines=2)

    map_output = gr.HTML(label="Map")
    confetti_output = gr.HTML(visible=True)

    run_button.click(
        fn=run_app,
        inputs=[upload, confetti_shape],
        outputs=[
            image_output,
            exif_output,
            prediction_output,
            error_output,
            map_output,
            status_output,
            confetti_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True)
