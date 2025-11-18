# app.py
import base64
import cv2
import numpy as np
import streamlit as st
from skimage import restoration, img_as_float

# ----------------------------
# Page config (must be after imports)
# ----------------------------
st.set_page_config(page_title="ReVivid — Restore & Enhance", layout="wide")

# ----------------------------
# Utility helpers
# ----------------------------
def _clip8(x):
    return np.clip(x, 0, 255).astype(np.uint8)

def read_image_from_bytes(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def bgr_to_display(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def make_download_link(img_bgr, filename="output.png"):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f'data:file/png;base64,{b64}', b64

# ----------------------------
# Enhancement pipeline
# ----------------------------
def enhance_image(img_bgr, steps):
    out = img_bgr.copy().astype(np.uint8)

    for name, params in steps:
        p = {} if params is None else dict(params)

        if name == "brightness_contrast":
            alpha = float(p.get("alpha", 1.0))
            beta  = int(p.get("beta", 0))
            out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        elif name == "gamma":
            gamma = float(p.get("gamma", 1.0))
            gamma = max(gamma, 1e-6)
            inv = 1.0 / gamma
            table = _clip8((np.arange(256) / 255.0) ** inv * 255.0)
            out = cv2.LUT(out, table)

        elif name == "clahe":
            clip = float(p.get("clip_limit", 2.0))
            tile = int(p.get("tile_grid", 8))
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            L, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            L2 = clahe.apply(L)
            out = cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)

        elif name == "sharpen":
            amount = float(p.get("amount", 1.0))
            radius = int(p.get("radius", 1))
            if radius < 1: radius = 1
            blur = cv2.GaussianBlur(out, (0, 0), radius)
            out = cv2.addWeighted(out, 1 + amount, blur, -amount, 0)

        elif name == "saturation":
            scale = float(p.get("scale", 1.2))
            val_scale = float(p.get("value_scale", 1.0))
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[...,1] *= scale
            hsv[...,2] *= val_scale
            hsv[...,1:] = np.clip(hsv[...,1:], 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif name == "auto_contrast":
            low = float(p.get("low_pct", 1.0))
            high = float(p.get("high_pct", 99.0))
            chs = cv2.split(out)
            chs2 = []
            for c in chs:
                lo = np.percentile(c, low)
                hi = np.percentile(c, high)
                if hi <= lo:
                    chs2.append(c)
                else:
                    c2 = (c.astype(np.float32) - lo) * (255.0 / (hi - lo))
                    chs2.append(_clip8(c2))
            out = cv2.merge(chs2)

        elif name == "white_balance":
            mode = str(p.get("mode", "grayworld")).lower()
            if mode == "grayworld":
                b, g, r = cv2.split(out.astype(np.float32))
                mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
                mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
                b *= (mean_gray / (mean_b + 1e-9))
                g *= (mean_gray / (mean_g + 1e-9))
                r *= (mean_gray / (mean_r + 1e-9))
                out = _clip8(cv2.merge([b, g, r]))
            else:
                out = out

        else:
            # unknown -> skip
            continue

    return out

# ----------------------------
# Restoration functions
# ----------------------------
def denoise_image_restoration(img_bgr, method="Non-Local Means", **kwargs):
    if method == "Median":
        k = int(kwargs.get("ksize", 3))
        k = max(3, k | 1)
        return cv2.medianBlur(img_bgr, k)
    elif method == "Bilateral":
        d = int(kwargs.get("diameter", 9))
        sc = float(kwargs.get("sigma_color", 75.0))
        ss = float(kwargs.get("sigma_space", 75.0))
        return cv2.bilateralFilter(img_bgr, d, sc, ss)
    elif method == "Non-Local Means":
        h = float(kwargs.get("h", 10.0))
        template = int(kwargs.get("templateWindowSize", 7))
        search = int(kwargs.get("searchWindowSize", 21))
        if img_bgr.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, template, search)
        else:
            return cv2.fastNlMeansDenoising(img_bgr, None, h, template, search)
    else:
        return img_bgr

def _gaussian_psf(size=15, sigma=3.0):
    k = size
    ax = np.arange(-k//2 + 1., k//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    return psf

def deblur_richardson_lucy(img_bgr, iterations=20, psf_size=9, psf_sigma=2.0):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    imgf = img_as_float(img_rgb)
    psf = _gaussian_psf(psf_size, psf_sigma)
    out_channels = []
    for c in range(3):
        channel = imgf[..., c]
        restored = restoration.richardson_lucy(channel, psf, iterations=iterations, clip=False)
        out_channels.append(restored)
    out = np.stack(out_channels, axis=-1)
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def deblur_wiener(img_bgr, balance=0.01, psf_size=9, psf_sigma=2.0):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    imgf = img_as_float(img_rgb)
    psf = _gaussian_psf(psf_size, psf_sigma)
    out_channels = []
    for c in range(3):
        channel = imgf[..., c]
        restored = restoration.wiener(channel, psf, balance=balance, clip=False)
        out_channels.append(restored)
    out = np.stack(out_channels, axis=-1)
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def remove_dust_specks(img_bgr, min_area=5, max_area=200):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, tb = cv2.threshold(bright, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, td = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(tb, td)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            keep[labels == i] = 255
    if keep.sum() == 0:
        return img_bgr
    result = cv2.inpaint(img_bgr, keep, 3, cv2.INPAINT_TELEA)
    return result

def restore_image(img_bgr, steps):
    out = img_bgr.copy()
    for name, p in steps:
        p = {} if p is None else dict(p)
        if name == "denoise":
            method = str(p.get("method", "Non-Local Means"))
            out = denoise_image_restoration(out, method=method,
                                            ksize=int(p.get("ksize",3)),
                                            diameter=int(p.get("diameter",9)),
                                            sigma_color=float(p.get("sigma_color",75.0)),
                                            sigma_space=float(p.get("sigma_space",75.0)),
                                            h=float(p.get("h",10.0)))
        elif name == "deblur_rl":
            iters = int(p.get("iterations", 20))
            psf_size = int(p.get("psf_size", 9))
            psf_sigma = float(p.get("psf_sigma", 2.0))
            out = deblur_richardson_lucy(out, iterations=iters, psf_size=psf_size, psf_sigma=psf_sigma)
        elif name == "deblur_wiener":
            balance = float(p.get("balance", 0.01))
            psf_size = int(p.get("psf_size", 9))
            psf_sigma = float(p.get("psf_sigma", 2.0))
            out = deblur_wiener(out, balance=balance, psf_size=psf_size, psf_sigma=psf_sigma)
        elif name == "inpaint_specks":
            min_area = int(p.get("min_area", 5))
            max_area = int(p.get("max_area", 200))
            out = remove_dust_specks(out, min_area=min_area, max_area=max_area)
        else:
            continue
    return out

# ----------------------------
# UI: Upload + Controls
# ----------------------------
st.title("ReVivid — Image Restoration, Enhancement & Colorization (Demo)")
st.caption("Restore old/blurred photos, then enhance (CLAHE, sharpen, saturation, etc.). Run restoration first, then enhancement.")

colA, colB = st.columns([1, 1])

with colA:
    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    # store bytes so we can reuse without re-reading stream
    if uploaded is not None:
        st.session_state['orig_bytes'] = uploaded.getvalue()

with colB:
    st.markdown("### Quick tips")
    st.write("- For large images, reduce resolution before heavy RL deblur (speed).")
    st.write("- Use Auto Restore for a quick one-click pipeline.")
    st.write("- Restoration ⟶ Enhancement (order matters).")

# ----------------------------
# Restoration controls
# ----------------------------
st.sidebar.header("Restoration")
use_auto_restore = st.sidebar.checkbox("Auto Restore (denoise → deblur → inpaint)", value=True)
if use_auto_restore:
    auto_strength = st.sidebar.slider("Auto strength", 0.5, 2.0, 1.0, 0.1)

st.sidebar.subheader("Manual restoration")
do_denoise = st.sidebar.checkbox("Denoise (manual)", value=False)
denoise_method = st.sidebar.selectbox("Method", ["Non-Local Means", "Median", "Bilateral"])
denoise_h = st.sidebar.slider("NLM: strength (h)", 1.0, 30.0, 10.0, 0.5)
denoise_kernel = st.sidebar.slider("Median kernel size", 3, 11, 3, step=2)
do_deblur_rl = st.sidebar.checkbox("Deblur (Richardson-Lucy)", value=False)
rl_iters = st.sidebar.slider("RL iterations", 5, 60, 20, 1)
rl_psf = st.sidebar.slider("PSF size (odd)", 3, 31, 9, step=2)
rl_sigma = st.sidebar.slider("PSF sigma", 0.5, 5.0, 2.0, 0.1)
do_deblur_wiener = st.sidebar.checkbox("Deblur (Wiener)", value=False)
wiener_balance = st.sidebar.slider("Wiener balance", 0.001, 0.1, 0.01, 0.001)
do_inpaint = st.sidebar.checkbox("Auto inpaint specks", value=False)
min_area = st.sidebar.slider("Inpaint: min area", 1, 50, 5)
max_area = st.sidebar.slider("Inpaint: max area", 50, 400, 200)

# ----------------------------
# Enhancement controls
# ----------------------------
st.sidebar.header("Enhancement steps (in order)")
use_wb   = st.sidebar.checkbox("White balance (Gray-World)", value=True)

use_bc   = st.sidebar.checkbox("Brightness / Contrast", value=False)
alpha    = st.sidebar.slider("Contrast (alpha)", 0.5, 3.0, 1.2, 0.1, disabled=not use_bc)
beta     = st.sidebar.slider("Brightness (beta)", -100, 100, 10, 1, disabled=not use_bc)

use_gamma = st.sidebar.checkbox("Gamma correction", value=False)
gamma     = st.sidebar.slider("Gamma", 0.3, 3.0, 1.0, 0.05, disabled=not use_gamma)

use_clahe = st.sidebar.checkbox("CLAHE (local contrast)", value=True)
clip      = st.sidebar.slider("CLAHE: Clip limit", 1.0, 7.0, 2.0, 0.1, disabled=not use_clahe)
tile      = st.sidebar.slider("CLAHE: Tile grid", 4, 16, 8, disabled=not use_clahe)

use_auto  = st.sidebar.checkbox("Auto-contrast (percentile stretch)", value=False)
low_pct   = st.sidebar.slider("Low percentile", 0.0, 10.0, 1.0, 0.5, disabled=not use_auto)
high_pct  = st.sidebar.slider("High percentile", 90.0, 100.0, 99.0, 0.5, disabled=not use_auto)

use_sharp = st.sidebar.checkbox("Sharpen (Unsharp mask)", value=True)
amount    = st.sidebar.slider("Sharpen amount", 0.0, 3.0, 1.0, 0.1, disabled=not use_sharp)
radius    = st.sidebar.slider("Sharpen radius", 1, 5, 1, disabled=not use_sharp)

use_sat   = st.sidebar.checkbox("Saturation boost", value=False)
sat_scale = st.sidebar.slider("Saturation scale", 0.0, 2.5, 1.15, 0.05, disabled=not use_sat)
val_scale = st.sidebar.slider("Value (brightness) scale in HSV", 0.5, 1.5, 1.0, 0.05, disabled=not use_sat)

# ----------------------------
# Buttons: Run Restore / Enhance / Both
# ----------------------------
run_restore_auto = st.sidebar.button("Run Auto Restore")
run_restore_manual = st.sidebar.button("Run Manual Restore")
run_enhance = st.sidebar.button("Run Enhancement")
run_restore_then_enhance = st.sidebar.button("Restore → Enhance (auto)")

# ----------------------------
# Processing logic
# ----------------------------
orig_img = None
if 'orig_bytes' in st.session_state:
    orig_img = read_image_from_bytes(st.session_state['orig_bytes'])

restored_img = None
enhanced_img = None

if run_restore_auto and orig_img is not None:
    denoise_h = 8.0 * auto_strength
    rl_iters = int(15 * auto_strength)
    steps_restore = [
        ("denoise", {"method": "Non-Local Means", "h": denoise_h}),
        ("deblur_rl", {"iterations": rl_iters, "psf_size": 9, "psf_sigma": 2.0}),
        ("inpaint_specks", {"min_area": 3, "max_area": 500})
    ]
    with st.spinner("Running auto restore... (may be slow for large images)"):
        restored_img = restore_image(orig_img, steps_restore)
        st.session_state['restored'] = cv2.imencode(".png", restored_img)[1].tobytes()

if run_restore_manual and orig_img is not None:
    steps_restore = []
    if do_denoise:
        if denoise_method == "Non-Local Means":
            steps_restore.append(("denoise", {"method":"Non-Local Means", "h": denoise_h}))
        elif denoise_method == "Median":
            steps_restore.append(("denoise", {"method":"Median", "ksize": denoise_kernel}))
        else:
            steps_restore.append(("denoise", {"method":"Bilateral", "diameter":9, "sigma_color":75, "sigma_space":75}))
    if do_deblur_rl:
        steps_restore.append(("deblur_rl", {"iterations": rl_iters, "psf_size": rl_psf, "psf_sigma": rl_sigma}))
    if do_deblur_wiener:
        steps_restore.append(("deblur_wiener", {"balance": wiener_balance, "psf_size": 9, "psf_sigma": 2.0}))
    if do_inpaint:
        steps_restore.append(("inpaint_specks", {"min_area": min_area, "max_area": max_area}))

    if not steps_restore:
        st.warning("No manual restore steps selected.")
    else:
        with st.spinner("Running manual restore..."):
            restored_img = restore_image(orig_img, steps_restore)
            st.session_state['restored'] = cv2.imencode(".png", restored_img)[1].tobytes()

# Enhancement run (on either orig or restored if available)
if run_enhance and orig_img is not None:
    base_for_enhance = orig_img.copy()
    steps = []
    if use_wb:   steps.append(("white_balance", {}))
    if use_bc:   steps.append(("brightness_contrast", {"alpha": alpha, "beta": int(beta)}))
    if use_gamma:steps.append(("gamma", {"gamma": float(gamma)}))
    if use_clahe:steps.append(("clahe", {"clip_limit": float(clip), "tile_grid": int(tile)}))
    if use_auto: steps.append(("auto_contrast", {"low_pct": float(low_pct), "high_pct": float(high_pct)}))
    if use_sharp:steps.append(("sharpen", {"amount": float(amount), "radius": int(radius)}))
    if use_sat:  steps.append(("saturation", {"scale": float(sat_scale), "value_scale": float(val_scale)}))
    with st.spinner("Running enhancement..."):
        enhanced_img = enhance_image(base_for_enhance, steps)
        st.session_state['enhanced'] = cv2.imencode(".png", enhanced_img)[1].tobytes()

# Restore then enhance (auto restore -> default enhancement)
if run_restore_then_enhance and orig_img is not None:
    denoise_h = 8.0 * auto_strength
    rl_iters = int(15 * auto_strength)
    steps_restore = [
        ("denoise", {"method": "Non-Local Means", "h": denoise_h}),
        ("deblur_rl", {"iterations": rl_iters, "psf_size": 9, "psf_sigma": 2.0}),
        ("inpaint_specks", {"min_area": 3, "max_area": 500})
    ]
    with st.spinner("Restoring..."):
        restored_img = restore_image(orig_img, steps_restore)
        st.session_state['restored'] = cv2.imencode(".png", restored_img)[1].tobytes()
    # then enhancement
    steps = []
    if use_wb:   steps.append(("white_balance", {}))
    if use_bc:   steps.append(("brightness_contrast", {"alpha": alpha, "beta": int(beta)}))
    if use_gamma:steps.append(("gamma", {"gamma": float(gamma)}))
    if use_clahe:steps.append(("clahe", {"clip_limit": float(clip), "tile_grid": int(tile)}))
    if use_auto: steps.append(("auto_contrast", {"low_pct": float(low_pct), "high_pct": float(high_pct)}))
    if use_sharp:steps.append(("sharpen", {"amount": float(amount), "radius": int(radius)}))
    if use_sat:  steps.append(("saturation", {"scale": float(sat_scale), "value_scale": float(val_scale)}))
    with st.spinner("Applying enhancement..."):
        enhanced_img = enhance_image(restored_img, steps)
        st.session_state['enhanced'] = cv2.imencode(".png", enhanced_img)[1].tobytes()

# ----------------------------
# Display results
# ----------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Original")
    if orig_img is not None:
        st.image(bgr_to_display(orig_img), use_column_width=True)
        if 'orig_bytes' in st.session_state:
            href = make_download_link(orig_img, "original.png")[0]
            st.markdown(f'<a download="original.png" href="{href}">⬇️ Download original</a>', unsafe_allow_html=True)
    else:
        st.info("Upload an image to begin.")

with right_col:
    if 'restored' in st.session_state:
        st.subheader("Restored (last)")
        restored_bytes = st.session_state['restored']
        restored_img_display = read_image_from_bytes(restored_bytes)
        st.image(bgr_to_display(restored_img_display), use_column_width=True)
        href = make_download_link(restored_img_display, "restored.png")[0]
        st.markdown(f'<a download="restored.png" href="{href}">⬇️ Download restored</a>', unsafe_allow_html=True)
    elif 'enhanced' in st.session_state:
        st.subheader("Enhanced (last)")
        enhanced_bytes = st.session_state['enhanced']
        enhanced_img_display = read_image_from_bytes(enhanced_bytes)
        st.image(bgr_to_display(enhanced_img_display), use_column_width=True)
        href = make_download_link(enhanced_img_display, "enhanced.png")[0]
        st.markdown(f'<a download="enhanced.png" href="{href}">⬇️ Download enhanced</a>', unsafe_allow_html=True)
    else:
        st.info("No result yet. Use the sidebar to run Restore or Enhance.")

# footer note
st.markdown("---")
st.caption("Restore first for best results (denoise → deblur → inpaint), then apply enhancement (CLAHE → sharpen → saturation). RL deblur can be slow on big images.")
