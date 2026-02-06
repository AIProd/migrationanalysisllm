# app.py — Deterministic scratch assay + LLM Independent comparison + optional overlay QC
import base64
import io
import json
import os
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import matplotlib.pyplot as plt

from skimage.filters import gaussian, sobel
from skimage import morphology, measure

# Optional LLM deps
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
    from pydantic import BaseModel, Field
except Exception:
    OPENAI_AVAILABLE = False


# =====================================================
# Compatibility helpers (skimage deprecations)
# =====================================================

def remove_small_objects_compat(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    scikit-image 0.26+ deprecates remove_small_objects(min_size=...) in favor of max_size=...
    Warning note: new threshold removes objects <= value, old removed < value.
    To approximate the old behavior, pass max_size=min_size-1 when supported.
    Falls back to min_size for older versions.
    """
    ms = int(min_size)
    if ms <= 0:
        return mask
    try:
        return morphology.remove_small_objects(mask, max_size=max(ms - 1, 0))
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=ms)


def closing(mask: np.ndarray, radius: int) -> np.ndarray:
    return morphology.closing(mask, footprint=morphology.disk(int(radius)))


def opening(mask: np.ndarray, radius: int) -> np.ndarray:
    return morphology.opening(mask, footprint=morphology.disk(int(radius)))


def dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    return morphology.dilation(mask, footprint=morphology.disk(int(radius)))


# =====================================================
# Core image / mask utilities
# =====================================================

def to_gray(image_pil: Image.Image, gaussian_sigma: float) -> np.ndarray:
    """Convert PIL image to grayscale float and apply Gaussian blur."""
    gray = np.array(ImageOps.grayscale(image_pil)).astype(np.float32)
    blurred = gaussian(gray, sigma=float(gaussian_sigma))
    return blurred


def build_wound_mask_from_t0(
    gray_blur: np.ndarray,
    wound_low_grad_percentile: float,
    morph_kernel_radius: int,
    min_wound_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify the wound region at time 0.
    Steps:
        1) Sobel gradient
        2) Pixels below percentile gradient = candidate wound
        3) Morph cleanup
        4) Keep largest connected component
    """
    grad0 = sobel(gray_blur)

    thr = np.percentile(grad0, float(wound_low_grad_percentile))
    wound_candidate = grad0 < thr

    wound_candidate = remove_small_objects_compat(wound_candidate, int(min_wound_size))
    wound_candidate = closing(wound_candidate, int(morph_kernel_radius))
    wound_candidate = opening(wound_candidate, int(morph_kernel_radius))

    labeled, _ = measure.label(wound_candidate, return_num=True)
    sizes = np.bincount(labeled.ravel())
    if sizes.size == 0:
        raise ValueError("No wound-like region detected in first frame.")
    sizes[0] = 0
    biggest_label = sizes.argmax()
    wound_mask = labeled == biggest_label

    return wound_mask, grad0


def make_band_mask(wound_mask: np.ndarray, band_thickness_px: int) -> np.ndarray:
    """Ring just outside wound used as monolayer reference for normalization."""
    dilated = dilation(wound_mask, int(band_thickness_px))
    band_mask = np.logical_and(dilated, ~wound_mask)
    return band_mask


def parse_hours_from_name(name: str) -> float:
    """
    Extract time (hours) from filename.
    Supports:
      - 01d00h00m
      - 24H / 72 H
    """
    m = re.search(r"(\d+)\s*[dD]\s*(\d+)\s*[hH]", name)
    if m:
        days = float(m.group(1))
        hours = float(m.group(2))
        return days * 24.0 + hours

    m = re.search(r"(\d+)\s*[hH]", name)
    if m:
        return float(m.group(1))

    return 0.0


def overlay_debug_rgb(
    img_pil: Image.Image,
    wound_mask: np.ndarray,
    wound_cells_mask: np.ndarray,
    alpha_wound: float = 0.4,
    alpha_cells: float = 0.4,
) -> Image.Image:
    """
    QC overlay:
      - wound ROI (t0) tinted blue
      - detected cells within ROI tinted green
    """
    base = np.array(img_pil.convert("RGB")).astype(np.float32)
    out = base.copy()

    blue = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)

    out[wound_mask] = (1 - float(alpha_wound)) * out[wound_mask] + float(alpha_wound) * blue
    out[wound_cells_mask] = (1 - float(alpha_cells)) * out[wound_cells_mask] + float(alpha_cells) * green

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =====================================================
# Metric computation (deterministic)
# =====================================================

def _cell_threshold(grad: np.ndarray, band_mask: np.ndarray, cell_percentile: float) -> float:
    """
    Adaptive threshold using Sobel gradient.
    LOWER percentile => LOWER threshold => MORE green (more sensitive, more FPs)
    HIGHER percentile => HIGHER threshold => LESS green (stricter)
    """
    p = float(cell_percentile)
    if band_mask.sum() == 0:
        return float(np.percentile(grad, p))
    return float(np.percentile(grad[band_mask], p))


def analyze_timepoint(
    gray_blur: np.ndarray,
    wound_mask: np.ndarray,
    band_mask: np.ndarray,
    w0_frac: float,
    cell_percentile: float,
    min_cell_size: int,
) -> Dict[str, object]:
    grad = sobel(gray_blur)
    thr_cell = _cell_threshold(grad, band_mask, float(cell_percentile))

    wound_cells_mask = np.logical_and(wound_mask, grad > thr_cell)
    band_cells_mask = np.logical_and(band_mask, grad > thr_cell)

    if int(min_cell_size) > 0:
        wound_cells_mask = remove_small_objects_compat(wound_cells_mask, int(min_cell_size))
        band_cells_mask = remove_small_objects_compat(band_cells_mask, int(min_cell_size))

    wound_area = wound_mask.sum()
    band_area = band_mask.sum() if band_mask.sum() > 0 else 1

    w_frac = wound_cells_mask.sum() / max(wound_area, 1)
    c_frac = band_cells_mask.sum() / band_area

    wound_confluence_pct = 100.0 * w_frac

    denom = (c_frac - w0_frac)
    if abs(denom) < 1e-9:
        rwd_pct = 0.0
    else:
        rwd_pct = 100.0 * (w_frac - w0_frac) / denom

    rwd_pct = float(np.clip(rwd_pct, 0, 100))

    return {
        "wound_confluence_pct": float(wound_confluence_pct),
        "relative_wound_density_pct": float(rwd_pct),
        "w_frac": float(w_frac),
        "c_frac": float(c_frac),
        "wound_cells_mask": wound_cells_mask,
    }


def run_full_analysis(
    images: List[Image.Image],
    names: List[str],
    gaussian_sigma: float,
    wound_low_grad_percentile: float,
    morph_kernel_radius: int,
    min_wound_size: int,
    band_thickness_px: int,
    cell_percentile: float,
    min_cell_size: int,
) -> Tuple[pd.DataFrame, List[Image.Image], List[Image.Image], List[str], List[float]]:
    """
    Returns:
      df_metrics, overlays, images_sorted, names_sorted, hours_sorted
    """
    hours_list = [parse_hours_from_name(n) for n in names]
    order = np.argsort(hours_list)

    images_sorted = [images[i] for i in order]
    names_sorted = [names[i] for i in order]
    hours_sorted = [float(hours_list[i]) for i in order]

    gray_series = [to_gray(im, float(gaussian_sigma)) for im in images_sorted]

    wound_mask, _ = build_wound_mask_from_t0(
        gray_series[0],
        float(wound_low_grad_percentile),
        int(morph_kernel_radius),
        int(min_wound_size),
    )

    band_mask = make_band_mask(wound_mask, int(band_thickness_px))

    # baseline w0
    grad_first = sobel(gray_series[0])
    thr_cell_first = _cell_threshold(grad_first, band_mask, float(cell_percentile))
    wound_cells_first = np.logical_and(wound_mask, grad_first > thr_cell_first)

    if int(min_cell_size) > 0:
        wound_cells_first = remove_small_objects_compat(wound_cells_first, int(min_cell_size))

    w0_frac = wound_cells_first.sum() / max(wound_mask.sum(), 1)

    rows: List[Dict[str, object]] = []
    overlays: List[Image.Image] = []

    for img_pil, gray_img, hr, nm in zip(images_sorted, gray_series, hours_sorted, names_sorted):
        metrics = analyze_timepoint(
            gray_img,
            wound_mask,
            band_mask,
            w0_frac=w0_frac,
            cell_percentile=float(cell_percentile),
            min_cell_size=int(min_cell_size),
        )

        rows.append({
            "Image": nm,
            "Hours": float(hr),
            "Wound Confluence (%)": float(metrics["wound_confluence_pct"]),
            "Relative Wound Density (%)": float(metrics["relative_wound_density_pct"]),
        })

        ov = overlay_debug_rgb(img_pil, wound_mask, metrics["wound_cells_mask"])
        overlays.append(ov)

    df_metrics = pd.DataFrame(rows)

    # Guarantee Hours exists and is numeric
    if "Hours" in df_metrics.columns:
        df_metrics["Hours"] = pd.to_numeric(df_metrics["Hours"], errors="coerce")
        df_metrics = df_metrics.sort_values("Hours").reset_index(drop=True)

    return df_metrics, overlays, images_sorted, names_sorted, hours_sorted


# =====================================================
# Plotting + export helpers
# =====================================================

def plot_metric(hours: np.ndarray, values: np.ndarray, ylabel: str, title: str, scale: float):
    fig_w = 4 * float(scale)
    fig_h = 3 * float(scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    ax.plot(hours, values, marker="o", linewidth=2)
    ax.set_xlabel("Hours")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _interp_time_to_pct(hours: np.ndarray, y: np.ndarray, pct: float) -> Optional[float]:
    if len(hours) < 2:
        return None
    for i in range(1, len(hours)):
        if (y[i - 1] < pct) and (y[i] >= pct):
            x0, x1 = hours[i - 1], hours[i]
            y0, y1 = y[i - 1], y[i]
            if abs(y1 - y0) < 1e-12:
                return float(x1)
            t = (pct - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return None


def compute_kinetics(df_metrics: pd.DataFrame) -> Dict[str, Optional[float]]:
    if "Hours" not in df_metrics.columns or "Wound Confluence (%)" not in df_metrics.columns:
        return {
            "slope_pct_per_hour": None,
            "time_to_50pct_hours": None,
            "time_to_80pct_hours": None,
            "monotonic_violations_count": None,
            "final_confluence_pct": None,
        }

    hours = df_metrics["Hours"].to_numpy(dtype=float)
    conf = df_metrics["Wound Confluence (%)"].to_numpy(dtype=float)

    slope = None
    if len(hours) >= 2 and np.std(hours) > 1e-9:
        slope = float(np.polyfit(hours, conf, 1)[0])

    t50 = _interp_time_to_pct(hours, conf, 50.0)
    t80 = _interp_time_to_pct(hours, conf, 80.0)
    monotonic_violations = int(np.sum(np.diff(conf) < -1e-6)) if len(conf) >= 2 else 0

    return {
        "slope_pct_per_hour": slope,
        "time_to_50pct_hours": t50,
        "time_to_80pct_hours": t80,
        "monotonic_violations_count": float(monotonic_violations),
        "final_confluence_pct": float(conf[-1]) if len(conf) else None,
    }


# =====================================================
# LLM utilities
# =====================================================

def _pil_to_data_url(img: Image.Image, max_side: int = 1024, quality: int = 85) -> str:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _has_openai_key() -> bool:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        return True
    return bool(os.getenv("OPENAI_API_KEY"))


def _get_openai_client() -> "OpenAI":
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai/pydantic not installed.")
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return OpenAI()


if OPENAI_AVAILABLE:
    class LLMIndependentFrame(BaseModel):
        image_name: str
        hours: float
        wound_confluence_pct_est: float = Field(ge=0, le=100)
        rwd_like_pct_est: Optional[float] = Field(default=None, ge=0, le=100)
        confidence_0to1: float = Field(ge=0, le=1)
        drift_or_fov_change_suspected: bool
        issues: List[str] = Field(default_factory=list)
        notes: str

    class LLMIndependentResult(BaseModel):
        method_summary: str
        frames: List[LLMIndependentFrame]

    class LLMOverlayFrameReview(BaseModel):
        image_name: str
        hours: float
        wound_mask_ok: bool
        cell_detection_ok: bool
        issues: List[str] = Field(default_factory=list)
        notes: str

    class LLMOverlayQCResult(BaseModel):
        overall_assessment: str
        frame_reviews: List[LLMOverlayFrameReview]
        recommended_actions: List[str] = Field(default_factory=list)


def run_llm_independent_analysis(
    images_sorted: List[Image.Image],
    names_sorted: List[str],
    hours_sorted: List[float],
    model: str,
    detail: str,
    max_frames: int,
) -> Dict:
    """
    LLM sees ONLY originals (baseline t0 + each frame) and independently estimates closure.
    """
    if not _has_openai_key():
        raise RuntimeError("OPENAI_API_KEY not set (env var or .streamlit/secrets.toml).")

    client = _get_openai_client()
    n = len(images_sorted)
    if n == 0:
        raise ValueError("No images provided.")

    if n > int(max_frames):
        idxs = np.linspace(0, n - 1, int(max_frames)).round().astype(int).tolist()
        idxs = sorted(set(idxs))
    else:
        idxs = list(range(n))

    t0_img = images_sorted[0]
    t0_name = names_sorted[0]
    t0_hr = float(hours_sorted[0])

    content = [
        {"type": "input_text", "text": (
            "You are an expert analyzing scratch-wound (migration) assay microscopy images.\n"
            "You will see a BASELINE frame (t0) and multiple CURRENT frames.\n\n"
            "For each CURRENT frame, estimate independently:\n"
            "- wound_confluence_pct_est: % of the ORIGINAL baseline gap area now covered by cells (0–100).\n"
            "- rwd_like_pct_est (optional): gap interior density relative to surrounding monolayer (0–100).\n"
            "- confidence_0to1: lower if uncertain.\n"
            "- drift_or_fov_change_suspected: True if FOV moved/cropped/rotated.\n"
            "- issues: short phrases.\n"
            "- notes: 1–3 sentences.\n\n"
            "Important: Do NOT assume overlays/masks. Keep t0 near 0% unless you clearly see cells/debris in the gap."
        )},
        {"type": "input_text", "text": f"BASELINE (t0): {t0_name} | {t0_hr:.2f} h"},
        {"type": "input_image", "image_url": _pil_to_data_url(t0_img), "detail": detail},
    ]

    for i in idxs:
        nm = names_sorted[i]
        hr = float(hours_sorted[i])
        content.extend([
            {"type": "input_text", "text": f"CURRENT: {nm} | {hr:.2f} h"},
            {"type": "input_image", "image_url": _pil_to_data_url(images_sorted[i]), "detail": detail},
        ])

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Return only the structured result."},
            {"role": "user", "content": content},
        ],
        text_format=LLMIndependentResult,
    )
    return resp.output_parsed.model_dump()


def run_llm_overlay_qc(
    originals_sorted: List[Image.Image],
    overlays: List[Image.Image],
    names_sorted: List[str],
    hours_sorted: List[float],
    params: Dict[str, float],
    model: str,
    detail: str,
    max_frames: int,
) -> Dict:
    """
    LLM sees ORIGINAL + OVERLAY to critique ROI/cell detection. Advisory only.
    """
    if not _has_openai_key():
        raise RuntimeError("OPENAI_API_KEY not set (env var or .streamlit/secrets.toml).")

    client = _get_openai_client()
    n = len(overlays)
    if n == 0:
        raise ValueError("No overlays to QC.")

    idxs = sorted(set([0, n // 2, n - 1]))
    if int(max_frames) > len(idxs):
        extra = np.linspace(0, n - 1, int(max_frames)).round().astype(int).tolist()
        idxs = sorted(set(idxs + extra))[:int(max_frames)]
    else:
        idxs = idxs[:int(max_frames)]

    content = [
        {"type": "input_text", "text": (
            "You are QC-reviewing scratch-wound assay overlays.\n"
            "Legend: Blue = baseline wound ROI (t0). Green = detected cells within ROI at that time.\n\n"
            "Assess: (1) wound ROI accuracy/stability, (2) whether green looks like true cells vs debris/edge halos,\n"
            "and list issues. Provide recommended_actions as high-level goals (no numeric slider values).\n\n"
            f"Context parameters: {json.dumps(params)}"
        )},
    ]

    for i in idxs:
        nm = names_sorted[i]
        hr = float(hours_sorted[i])
        content.extend([
            {"type": "input_text", "text": f"FRAME: {nm} | {hr:.2f} h | ORIGINAL"},
            {"type": "input_image", "image_url": _pil_to_data_url(originals_sorted[i]), "detail": detail},
            {"type": "input_text", "text": f"FRAME: {nm} | {hr:.2f} h | OVERLAY"},
            {"type": "input_image", "image_url": _pil_to_data_url(overlays[i]), "detail": detail},
        ])

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Return only the structured result."},
            {"role": "user", "content": content},
        ],
        text_format=LLMOverlayQCResult,
    )
    return resp.output_parsed.model_dump()


def build_comparison_df(df_rule: pd.DataFrame, llm_ind: Dict) -> pd.DataFrame:
    """
    Robust merge that guarantees Hours exists (prevents KeyError 'Hours').
    """
    frames = llm_ind.get("frames", []) if isinstance(llm_ind, dict) else []
    df_llm = pd.DataFrame([{
        "Image": f.get("image_name"),
        "Hours": float(f.get("hours", np.nan)),
        "LLM Wound Confluence (%)": float(f.get("wound_confluence_pct_est", np.nan)),
        "LLM RWD-like (%)": None if f.get("rwd_like_pct_est") is None else float(f.get("rwd_like_pct_est")),
        "LLM Confidence": float(f.get("confidence_0to1", np.nan)),
        "LLM Drift Suspected": bool(f.get("drift_or_fov_change_suspected", False)),
        "LLM Issues": "; ".join(f.get("issues", []) or []),
        "LLM Notes": f.get("notes", ""),
    } for f in frames])

    df = df_rule.copy()

    if not df_llm.empty:
        if (
            "Image" in df.columns and "Hours" in df.columns
            and "Image" in df_llm.columns and "Hours" in df_llm.columns
        ):
            df = df.merge(df_llm, on=["Image", "Hours"], how="left")
        elif "Image" in df.columns and "Image" in df_llm.columns:
            df = df.merge(df_llm.drop(columns=["Hours"], errors="ignore"), on=["Image"], how="left")

    if "Hours" not in df.columns:
        for c in ["Hours_x", "Hours_y"]:
            if c in df.columns:
                df["Hours"] = df[c]
                break

    if "Wound Confluence (%)" in df.columns and "LLM Wound Confluence (%)" in df.columns:
        df["Δ Confluence (LLM - Rule)"] = df["LLM Wound Confluence (%)"] - df["Wound Confluence (%)"]
        df["Abs Δ Confluence"] = df["Δ Confluence (LLM - Rule)"].abs()
    else:
        df["Δ Confluence (LLM - Rule)"] = np.nan
        df["Abs Δ Confluence"] = np.nan

    if "Hours" in df.columns:
        df = df.sort_values("Hours").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def heuristic_qc_for_disagreement(row: pd.Series) -> str:
    issues = str(row.get("LLM Issues", "") or "").lower()
    drift = bool(row.get("LLM Drift Suspected", False))
    d = row.get("Δ Confluence (LLM - Rule)")

    if pd.isna(d):
        return "No LLM estimate for this frame."

    hints: List[str] = []
    if drift:
        hints.append("LLM suspects FOV drift: t0 ROI may be misaligned -> consider registration.")
    if ("debris" in issues) or ("speckle" in issues) or ("artifact" in issues) or ("halo" in issues):
        if d < 0:
            hints.append(
                "LLM flags debris/FPs: rule-based may inflate confluence -> make detection stricter "
                "(increase cell_percentile, increase min_cell_size)."
            )
        else:
            hints.append("LLM is conservative due to debris/texture; it may under-call closure.")
    if ("illumination" in issues) or ("contrast" in issues) or ("focus" in issues):
        hints.append("Contrast/focus issues: both methods can be unstable; review raw + overlays.")
    if ("irregular" in issues) or ("scratch" in issues) or ("boundary" in issues):
        hints.append("Irregular boundaries can cause ROI leaks or edge-halo detection; review wound ROI.")

    if not hints:
        hints.append("No clear artifact flags: review overlays and consider parameter sweep.")

    return " ".join(hints)


# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="Wound Healing Analysis + LLM Independent Comparison", layout="wide")
st.title("Wound Healing Analysis (Deterministic) + LLM Independent Comparison (Optional)")

st.write(
    "Deterministic quantification + optional LLM vision that independently estimates closure from raw images "
    "and compares against the deterministic metrics."
)

with st.form("analysis_form"):
    uploaded_files = st.file_uploader(
        "Upload all timepoints from one well (e.g. 0h, 24h, 48h, 72h). All images should use the same magnification.",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    st.markdown("### Analysis settings (optional)")
    st.caption("Defaults approximate Incucyte-style behavior. Adjust only if wound ROI or cell detection looks off.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gaussian_sigma = st.slider("Gaussian blur σ", 0.0, 5.0, 1.0, 0.1)
        wound_low_grad_percentile = st.slider("Wound smoothness percentile", 5, 60, 30, 1)

    with col2:
        morph_kernel_radius = st.slider("Wound edge smoothing (px)", 1, 30, 10, 1)
        band_thickness_px = st.slider("Reference band thickness (px)", 10, 200, 50, 5)

    with col3:
        min_wound_size = st.number_input("Min wound size (px area)", 100, 200000, 500, 100)
        cell_percentile = st.slider("Cell texture percentile", 1, 50, 5, 1)

    with col4:
        min_cell_size = st.slider("Min cell object size (px)", 0, 500, 80, 10)

    st.markdown("### Display settings")
    d1, d2 = st.columns(2)
    with d1:
        plot_scale = st.slider("Plot scale", 0.5, 1.5, 0.8, 0.1)
    with d2:
        overlay_cols = st.slider("Overlay columns", 2, 4, 3, 1)

    st.markdown("### Optional: LLM Independent Analysis + Comparison")
    run_llm_independent_flag = st.checkbox(
        "Run LLM independent analysis (t0 + each frame originals only) and compare to deterministic",
        value=False,
    )

    llm_ind_model = "gpt-5.2-pro"
    llm_ind_detail = "high"
    llm_ind_max_frames = 12
    disagree_thr = 10.0

    if run_llm_independent_flag:
        if not OPENAI_AVAILABLE:
            st.warning("LLM features require openai + pydantic in requirements.txt.")
        llm_ind_model = st.selectbox("LLM independent model", ["gpt-5.2", "gpt-5.2-pro"], index=1)
        llm_ind_detail = st.selectbox("LLM image detail", ["low", "high", "auto"], index=1)
        llm_ind_max_frames = st.slider("Max frames to analyze (cost control)", 3, 24, 12, 1)
        disagree_thr = st.slider("Flag disagreement if |Δ confluence| ≥", 2.0, 30.0, 10.0, 1.0)

    st.markdown("### Optional: LLM Overlay QC (debug overlays)")
    run_llm_overlay_qc_flag = st.checkbox(
        "Run LLM overlay QC (original + overlays)",
        value=False,
    )

    llm_qc_model = "gpt-5.2"
    llm_qc_detail = "high"
    llm_qc_max_frames = 4

    if run_llm_overlay_qc_flag:
        if not OPENAI_AVAILABLE:
            st.warning("LLM features require openai + pydantic in requirements.txt.")
        llm_qc_model = st.selectbox("LLM QC model", ["gpt-5.2", "gpt-5.2-pro"], index=0)
        llm_qc_detail = st.selectbox("QC image detail", ["low", "high", "auto"], index=1)
        llm_qc_max_frames = st.slider("QC frames (cost control)", 2, 10, 4, 1)

    submitted = st.form_submit_button("Analyze")

if submitted:
    if not uploaded_files:
        st.warning("Please upload at least one image series.")
        st.stop()

    imgs = [Image.open(f).convert("RGB") for f in uploaded_files]
    names = [f.name for f in uploaded_files]

    try:
        df_metrics, overlays, imgs_sorted, names_sorted, hours_sorted = run_full_analysis(
            images=imgs,
            names=names,
            gaussian_sigma=gaussian_sigma,
            wound_low_grad_percentile=wound_low_grad_percentile,
            morph_kernel_radius=morph_kernel_radius,
            min_wound_size=min_wound_size,
            band_thickness_px=band_thickness_px,
            cell_percentile=cell_percentile,
            min_cell_size=min_cell_size,
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

    st.header("Metrics (Deterministic)")
    styled = df_metrics.style.format({
        "Hours": "{:.2f}",
        "Wound Confluence (%)": "{:.2f}",
        "Relative Wound Density (%)": "{:.2f}",
    })
    st.dataframe(styled, width="stretch")

    st.download_button(
        label="Download deterministic CSV",
        data=df_to_csv_bytes(df_metrics),
        file_name="wound_metrics_deterministic.csv",
        mime="text/csv",
    )

    st.header("Deterministic Kinetics (from metrics)")
    st.json(compute_kinetics(df_metrics))

    st.header("Time-Series Plots")
    hours_arr = df_metrics["Hours"].to_numpy(dtype=float)
    conf_arr = df_metrics["Wound Confluence (%)"].to_numpy(dtype=float)
    rwd_arr = df_metrics["Relative Wound Density (%)"].to_numpy(dtype=float)

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.pyplot(
            plot_metric(hours_arr, conf_arr, "Wound Confluence (%)", "Wound Confluence vs Time", plot_scale),
            clear_figure=True
        )
    with pcol2:
        st.pyplot(
            plot_metric(hours_arr, rwd_arr, "Relative Wound Density (%)", "Relative Wound Density vs Time", plot_scale),
            clear_figure=True
        )

    st.header("Overlay QC (Deterministic)")
    st.caption("Blue: wound ROI from first timepoint. Green: detected cells inside wound ROI at each timepoint.")
    cols = st.columns(int(overlay_cols))
    for i, (row, overlay_img) in enumerate(zip(df_metrics.itertuples(index=False), overlays)):
        col = cols[i % int(overlay_cols)]
        with col:
            st.caption(f"{row.Image}  ({row.Hours:.2f} h)")
            st.image(overlay_img, width="stretch")

    # --- LLM Independent ---
    if run_llm_independent_flag:
        st.header("LLM Independent Analysis vs Deterministic (Comparison)")

        if not OPENAI_AVAILABLE:
            st.error("LLM features require openai + pydantic installed.")
            st.stop()
        if not _has_openai_key():
            st.error("OPENAI_API_KEY not set. Add to Streamlit secrets or env var.")
            st.stop()

        with st.spinner("Running LLM independent analysis..."):
            llm_ind = run_llm_independent_analysis(
                images_sorted=imgs_sorted,
                names_sorted=names_sorted,
                hours_sorted=hours_sorted,
                model=llm_ind_model,
                detail=llm_ind_detail,
                max_frames=int(llm_ind_max_frames),
            )

        df_cmp = build_comparison_df(df_metrics, llm_ind)
        df_cmp["Disagree?"] = df_cmp["Abs Δ Confluence"] >= float(disagree_thr)
        df_cmp["QC Hint (heuristic)"] = df_cmp.apply(heuristic_qc_for_disagreement, axis=1)

        st.subheader("Comparison table")
        st.dataframe(df_cmp, width="stretch")

        st.download_button(
            label="Download comparison CSV",
            data=df_to_csv_bytes(df_cmp),
            file_name="wound_metrics_comparison_llm_vs_rule.csv",
            mime="text/csv",
        )

        n_bad = int(df_cmp["Disagree?"].sum())
        st.subheader("Disagreement summary")
        st.write(f"Frames with |Δ confluence| ≥ {float(disagree_thr):.1f}%: **{n_bad} / {len(df_cmp)}**")

        st.subheader("Rule vs LLM confluence")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.plot(df_cmp["Hours"], df_cmp["Wound Confluence (%)"], marker="o", linewidth=2, label="Deterministic")
        ax.plot(df_cmp["Hours"], df_cmp["LLM Wound Confluence (%)"], marker="o", linewidth=2, label="LLM independent")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Wound Confluence (%)")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        st.subheader("Δ Confluence (LLM - Rule) vs time")
        fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=120)
        ax2.axhline(0, linewidth=1)
        ax2.plot(df_cmp["Hours"], df_cmp["Δ Confluence (LLM - Rule)"], marker="o", linewidth=2)
        ax2.set_xlabel("Hours")
        ax2.set_ylabel("Δ Confluence (%)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2, clear_figure=True)

        st.subheader("LLM method summary")
        st.write(llm_ind.get("method_summary", ""))

    # --- LLM Overlay QC ---
    if run_llm_overlay_qc_flag:
        st.header("LLM Overlay QC (advisory)")

        if not OPENAI_AVAILABLE:
            st.error("LLM features require openai + pydantic installed.")
            st.stop()
        if not _has_openai_key():
            st.error("OPENAI_API_KEY not set. Add to Streamlit secrets or env var.")
            st.stop()

        params = {
            "gaussian_sigma": float(gaussian_sigma),
            "wound_low_grad_percentile": float(wound_low_grad_percentile),
            "morph_kernel_radius": float(morph_kernel_radius),
            "min_wound_size": float(min_wound_size),
            "band_thickness_px": float(band_thickness_px),
            "cell_percentile": float(cell_percentile),
            "min_cell_size": float(min_cell_size),
        }

        with st.spinner("Running LLM overlay QC..."):
            qc = run_llm_overlay_qc(
                originals_sorted=imgs_sorted,
                overlays=overlays,
                names_sorted=names_sorted,
                hours_sorted=hours_sorted,
                params=params,
                model=llm_qc_model,
                detail=llm_qc_detail,
                max_frames=int(llm_qc_max_frames),
            )
        st.json(qc)
