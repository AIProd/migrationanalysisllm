import base64
import io
import json
import re
from dataclasses import asdict
from typing import List, Tuple, Dict, Optional, Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import matplotlib.pyplot as plt

from skimage.filters import gaussian, sobel
from skimage import morphology, measure

# --- Optional (LLM) ---
try:
    from openai import OpenAI
    from pydantic import BaseModel, Field
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# =====================================================
# Core image / mask utilities (same as yours)
# =====================================================

def to_gray(image_pil: Image.Image, gaussian_sigma: float) -> np.ndarray:
    gray = np.array(ImageOps.grayscale(image_pil)).astype(np.float32)
    blurred = gaussian(gray, sigma=gaussian_sigma)
    return blurred


def build_wound_mask_from_t0(
    gray_blur: np.ndarray,
    wound_low_grad_percentile: float,
    morph_kernel_radius: int,
    min_wound_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    grad0 = sobel(gray_blur)

    thr = np.percentile(grad0, wound_low_grad_percentile)
    wound_candidate = grad0 < thr

    wound_candidate = morphology.remove_small_objects(
        wound_candidate, min_size=min_wound_size
    )
    wound_candidate = morphology.binary_closing(
        wound_candidate, morphology.disk(morph_kernel_radius)
    )
    wound_candidate = morphology.binary_opening(
        wound_candidate, morphology.disk(morph_kernel_radius)
    )

    labeled, _ = measure.label(wound_candidate, return_num=True)
    sizes = np.bincount(labeled.ravel())
    if sizes.size == 0:
        raise ValueError("No wound-like region detected in first frame.")
    sizes[0] = 0
    biggest_label = sizes.argmax()
    wound_mask = labeled == biggest_label

    return wound_mask, grad0


def make_band_mask(
    wound_mask: np.ndarray,
    band_thickness_px: int,
) -> np.ndarray:
    dilated = morphology.binary_dilation(
        wound_mask, morphology.disk(band_thickness_px)
    )
    band_mask = np.logical_and(dilated, ~wound_mask)
    return band_mask


def parse_hours_from_name(name: str) -> float:
    m = re.search(r'(\d+)\s*[dD]\s*(\d+)\s*[hH]', name)
    if m:
        days = float(m.group(1))
        hours = float(m.group(2))
        return days * 24.0 + hours

    m = re.search(r'(\d+)\s*[hH]', name)
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
    base = np.array(img_pil.convert("RGB")).astype(np.float32)
    out = base.copy()

    blue = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)

    out[wound_mask] = (1 - alpha_wound) * out[wound_mask] + alpha_wound * blue
    out[wound_cells_mask] = (
        (1 - alpha_cells) * out[wound_cells_mask] + alpha_cells * green
    )

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =====================================================
# Metric computation (same as yours)
# =====================================================

def _cell_threshold(grad: np.ndarray, band_mask: np.ndarray, cell_percentile: float):
    if band_mask.sum() == 0:
        return np.percentile(grad, cell_percentile)
    return np.percentile(grad[band_mask], cell_percentile)


def analyze_timepoint(
    gray_blur: np.ndarray,
    wound_mask: np.ndarray,
    band_mask: np.ndarray,
    w0_frac: float,
    cell_percentile: float,
    min_cell_size: int,
) -> Dict[str, float]:
    grad = sobel(gray_blur)
    thr_cell = _cell_threshold(grad, band_mask, cell_percentile)

    wound_cells_mask = np.logical_and(wound_mask, grad > thr_cell)
    band_cells_mask = np.logical_and(band_mask, grad > thr_cell)

    if min_cell_size > 0:
        wound_cells_mask = morphology.remove_small_objects(
            wound_cells_mask, min_size=min_cell_size
        )
        band_cells_mask = morphology.remove_small_objects(
            band_cells_mask, min_size=min_cell_size
        )

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
) -> Tuple[pd.DataFrame, List[Image.Image]]:
    hours_list = [parse_hours_from_name(n) for n in names]
    order = np.argsort(hours_list)

    images_sorted = [images[i] for i in order]
    names_sorted = [names[i] for i in order]
    hours_sorted = [hours_list[i] for i in order]

    gray_series = [to_gray(im, gaussian_sigma) for im in images_sorted]

    wound_mask, _grad0 = build_wound_mask_from_t0(
        gray_series[0],
        wound_low_grad_percentile,
        morph_kernel_radius,
        min_wound_size,
    )

    band_mask = make_band_mask(wound_mask, band_thickness_px)

    grad_first = sobel(gray_series[0])
    thr_cell_first = _cell_threshold(grad_first, band_mask, cell_percentile)
    wound_cells_first = np.logical_and(wound_mask, grad_first > thr_cell_first)

    if min_cell_size > 0:
        wound_cells_first = morphology.remove_small_objects(
            wound_cells_first, min_size=min_cell_size
        )

    w0_frac = wound_cells_first.sum() / max(wound_mask.sum(), 1)

    rows = []
    overlays = []

    for img_pil, gray_img, hr, nm in zip(
        images_sorted, gray_series, hours_sorted, names_sorted
    ):
        metrics = analyze_timepoint(
            gray_img,
            wound_mask,
            band_mask,
            w0_frac=w0_frac,
            cell_percentile=cell_percentile,
            min_cell_size=min_cell_size,
        )

        rows.append({
            "Image": nm,
            "Hours": hr,
            "Wound Confluence (%)": metrics["wound_confluence_pct"],
            "Relative Wound Density (%)": metrics["relative_wound_density_pct"],
        })

        wound_cells_now = metrics["wound_cells_mask"]
        ov = overlay_debug_rgb(img_pil, wound_mask, wound_cells_now)
        overlays.append(ov)

    df_metrics = pd.DataFrame(rows).sort_values("Hours").reset_index(drop=True)
    return df_metrics, overlays


# =====================================================
# Plotting + export helpers (same as yours)
# =====================================================

def plot_metric(
    hours: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    scale: float,
):
    fig_w = 4 * scale
    fig_h = 3 * scale

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


# =====================================================
# NEW: deterministic “insights” from metrics
# =====================================================

def _interp_time_to_pct(hours: np.ndarray, y: np.ndarray, pct: float) -> Optional[float]:
    """
    Linear interpolation for first time y crosses pct.
    Returns None if never crosses.
    """
    if len(hours) < 2:
        return None
    for i in range(1, len(hours)):
        if (y[i - 1] < pct) and (y[i] >= pct):
            # interpolate between i-1 and i
            x0, x1 = hours[i - 1], hours[i]
            y0, y1 = y[i - 1], y[i]
            if abs(y1 - y0) < 1e-9:
                return float(x1)
            t = (pct - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return None


def compute_kinetics(df_metrics: pd.DataFrame) -> Dict[str, Optional[float]]:
    hours = df_metrics["Hours"].to_numpy(dtype=float)
    conf = df_metrics["Wound Confluence (%)"].to_numpy(dtype=float)

    # simple slope (least squares)
    if len(hours) >= 2 and np.std(hours) > 1e-9:
        slope = float(np.polyfit(hours, conf, 1)[0])  # % per hour
    else:
        slope = None

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
# NEW: LLM vision QC + parameter suggestions + narrative
# =====================================================

def _pil_to_data_url(img: Image.Image, max_side: int = 1024, fmt: str = "jpeg", quality: int = 85) -> str:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    if fmt.lower() == "png":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _get_openai_client() -> "OpenAI":
    # OpenAI SDK reads OPENAI_API_KEY from env automatically.
    # This just also supports Streamlit secrets if you want.
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return OpenAI()


if OPENAI_AVAILABLE:
    class FrameReview(BaseModel):
        image_name: str
        hours: float
        wound_mask_ok: bool
        cell_detection_ok: bool
        issues: List[str] = Field(default_factory=list)
        notes: str

    class ParamSuggestion(BaseModel):
        parameter: Literal[
            "gaussian_sigma",
            "wound_low_grad_percentile",
            "morph_kernel_radius",
            "min_wound_size",
            "band_thickness_px",
            "cell_percentile",
            "min_cell_size",
        ]
        action: Literal["increase", "decrease", "keep"]
        amount: float = 0.0
        reason: str

    class LLMQCResult(BaseModel):
        overall_assessment: str
        frame_reviews: List[FrameReview]
        parameter_suggestions: List[ParamSuggestion]


def run_llm_qc(
    df_metrics: pd.DataFrame,
    originals: List[Image.Image],
    overlays: List[Image.Image],
    params: Dict[str, float],
    model: str,
    detail: str,
    max_frames: int,
) -> Dict:
    """
    Uses GPT vision to QC the segmentation overlays + suggest parameter changes.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai/pydantic not installed. pip install openai pydantic")

    client = _get_openai_client()

    # sample frames to control cost (first, middle, last)
    n = len(df_metrics)
    if n == 0:
        raise ValueError("No frames to QC.")

    idxs = sorted(set([
        0,
        n // 2,
        n - 1,
    ]))
    # if user wants more, fill evenly
    if max_frames > len(idxs):
        extra = np.linspace(0, n - 1, num=max_frames).round().astype(int).tolist()
        idxs = sorted(set(idxs + extra))[:max_frames]
    else:
        idxs = idxs[:max_frames]

    content = []
    content.append({
        "type": "input_text",
        "text": (
            "You are reviewing scratch-wound assay image analysis QC.\n"
            "You will see ORIGINAL microscopy frames and an OVERLAY.\n"
            "Overlay legend:\n"
            " - Blue = wound region defined at t0 (open region)\n"
            " - Green = detected cells inside that wound region (closure)\n\n"
            "Your task:\n"
            "1) For each frame: assess if the blue wound ROI looks correct and stable; "
            "and whether green closure looks plausible (not noise/debris).\n"
            "2) List concrete issues (e.g., wound ROI too wide/narrow, green picks debris, low contrast, focus drift).\n"
            "3) Suggest parameter changes to improve robustness.\n\n"
            "IMPORTANT: Be conservative; if QC is uncertain, say so.\n\n"
            f"Current parameters: {json.dumps(params)}\n"
        )
    })

    for i in idxs:
        row = df_metrics.iloc[i]
        nm = str(row["Image"])
        hr = float(row["Hours"])

        content.append({"type": "input_text", "text": f"\nFRAME: {nm} | {hr:.2f} h | ORIGINAL"})
        content.append({
            "type": "input_image",
            "image_url": _pil_to_data_url(originals[i], max_side=1024, fmt="jpeg", quality=85),
            "detail": detail,
        })
        content.append({"type": "input_text", "text": f"FRAME: {nm} | {hr:.2f} h | OVERLAY"})
        content.append({
            "type": "input_image",
            "image_url": _pil_to_data_url(overlays[i], max_side=1024, fmt="jpeg", quality=85),
            "detail": detail,
        })

    # Prefer strict structured output parsing
    try:
        resp = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": "Return only the requested structured QC result."},
                {"role": "user", "content": content},
            ],
            text_format=LLMQCResult,
        )
        parsed = resp.output_parsed
        return parsed.model_dump()
    except Exception:
        # Fallback: JSON-only + best-effort parse
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Return valid JSON only. No markdown. No extra text."},
                {"role": "user", "content": content},
            ],
        )
        raw = resp.output_text
        try:
            return json.loads(raw)
        except Exception:
            # last resort: attempt to extract JSON object substring
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                raise RuntimeError(f"LLM returned non-JSON: {raw[:500]}")
            return json.loads(m.group(0))


def llm_assay_summary(
    df_metrics: pd.DataFrame,
    kinetics: Dict[str, Optional[float]],
    model: str,
) -> str:
    """
    Text-only LLM: produces a succinct insights paragraph for the assay.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai not installed. pip install openai")

    client = _get_openai_client()

    payload = {
        "metrics_table": df_metrics.to_dict(orient="records"),
        "kinetics": kinetics,
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": (
                "You are a bioimage analysis expert. "
                "Write a concise, technical interpretation of scratch-wound assay metrics. "
                "Focus on kinetics, anomalies, and QC caveats. "
                "No fluff."
            )},
            {"role": "user", "content": [
                {"type": "input_text", "text": f"Summarize these results:\n{json.dumps(payload)}"}
            ]},
        ],
    )
    return resp.output_text


# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="Wound Healing Analysis + LLM QC", layout="wide")

st.title("Wound Healing Analysis (Deterministic) + LLM Vision QC (Optional)")

st.write(
    "Deterministic quantification (your current pipeline) + optional GPT vision layer for QC, "
    "parameter tuning suggestions, and richer assay insights."
)

with st.form("analysis_form"):
    uploaded_files = st.file_uploader(
        "Upload all timepoints from one well (e.g. 0h, 24h, 48h, 72h). "
        "All images should use the same magnification.",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    st.markdown("### Analysis settings (optional)")
    st.caption(
        "Defaults approximate Incucyte-style behavior. Adjust only if wound ROI or cell detection looks off."
    )

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
    disp_col1, disp_col2 = st.columns(2)
    with disp_col1:
        plot_scale = st.slider("Plot scale", 0.5, 1.5, 0.8, 0.1)
    with disp_col2:
        overlay_cols = st.slider("Overlay columns", 2, 4, 3, 1)

    st.markdown("### Optional: LLM Vision QC + Insights")
    use_llm = st.checkbox("Enable LLM Vision QC (requires OPENAI_API_KEY + openai package)", value=False)

    llm_model = "gpt-5.2"
    llm_detail = "low"
    llm_max_frames = 3
    if use_llm:
        if not OPENAI_AVAILABLE:
            st.warning("LLM enabled but openai/pydantic not installed. Run: pip install openai pydantic")
        llm_model = st.selectbox("Vision model", ["gpt-5.2", "gpt-5.2-pro"], index=0)
        llm_detail = st.selectbox("Image detail", ["low", "high", "auto"], index=0)
        llm_max_frames = st.slider("Frames to QC (cost control)", 2, 8, 3, 1)

    submitted = st.form_submit_button("Analyze")

if submitted:
    if not uploaded_files:
        st.warning("Please upload at least one image series.")
    else:
        imgs = [Image.open(f).convert("RGB") for f in uploaded_files]
        names = [f.name for f in uploaded_files]

        try:
            df_metrics, overlays = run_full_analysis(
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

        # Metrics table
        st.header("Metrics")
        styled = df_metrics.style.format({
            "Hours": "{:.2f}",
            "Wound Confluence (%)": "{:.2f}",
            "Relative Wound Density (%)": "{:.2f}",
        })
        st.dataframe(styled, use_container_width=True)

        csv_data = df_to_csv_bytes(df_metrics)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="wound_metrics.csv",
            mime="text/csv",
        )

        # Deterministic kinetics
        st.header("Deterministic Kinetics (from metrics)")
        kinetics = compute_kinetics(df_metrics)
        st.json(kinetics)

        # Plots
        st.header("Time-Series Plots")
        hours_arr = df_metrics["Hours"].to_numpy(dtype=float)
        conf_arr = df_metrics["Wound Confluence (%)"].to_numpy(dtype=float)
        rwd_arr = df_metrics["Relative Wound Density (%)"].to_numpy(dtype=float)

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            fig_conf = plot_metric(hours_arr, conf_arr, "Wound Confluence (%)", "Wound Confluence vs Time", plot_scale)
            st.pyplot(fig_conf, clear_figure=True)

        with pcol2:
            fig_rwd = plot_metric(hours_arr, rwd_arr, "Relative Wound Density (%)", "Relative Wound Density vs Time", plot_scale)
            st.pyplot(fig_rwd, clear_figure=True)

        # Overlays
        st.header("Overlay QC (Deterministic)")
        st.caption("Blue: wound region defined at first timepoint. Green: detected cells inside wound region (closure).")
        cols = st.columns(overlay_cols)
        for i, (row, overlay_img) in enumerate(zip(df_metrics.itertuples(index=False), overlays)):
            col = cols[i % overlay_cols]
            with col:
                st.caption(f"{row.Image}  ({row.Hours:.2f} h)")
                st.image(overlay_img, use_container_width=True)

        # LLM QC / insights
        if use_llm:
            st.header("LLM Vision QC + Parameter Suggestions")
            st.caption(
                "This uses GPT vision to critique the overlays and suggest parameter tweaks. "
                "Quantification remains deterministic; LLM is advisory."
            )

            params = {
                "gaussian_sigma": gaussian_sigma,
                "wound_low_grad_percentile": wound_low_grad_percentile,
                "morph_kernel_radius": morph_kernel_radius,
                "min_wound_size": min_wound_size,
                "band_thickness_px": band_thickness_px,
                "cell_percentile": cell_percentile,
                "min_cell_size": min_cell_size,
            }

            # IMPORTANT: originals need to be in the same sorted order as df_metrics/overlays.
            # We can reconstruct the sorted originals by sorting filenames the same way.
            # Easiest: re-run sorting here.
            hours_list = [parse_hours_from_name(n) for n in names]
            order = np.argsort(hours_list)
            imgs_sorted = [imgs[i] for i in order]

            try:
                qc = run_llm_qc(
                    df_metrics=df_metrics,
                    originals=imgs_sorted,
                    overlays=overlays,
                    params=params,
                    model=llm_model,
                    detail=llm_detail,
                    max_frames=llm_max_frames,
                )
            except Exception as e:
                st.error(f"LLM QC failed: {e}")
                st.stop()

            st.subheader("LLM QC Result (structured)")
            st.json(qc)

            st.subheader("LLM Assay Summary (text)")
            try:
                summary = llm_assay_summary(df_metrics, kinetics, model=llm_model)
                st.write(summary)
            except Exception as e:
                st.error(f"LLM summary failed: {e}")
