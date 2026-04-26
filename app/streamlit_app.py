import os
import glob
import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib
import matplotlib.cm as cm
from scipy.ndimage import zoom

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization

# ---------------------------------------------------------------------------
# CRITICAL: All file paths must be resolved relative to THIS script's
# location — not the working directory the process was launched from.
# HF Streamlit Docker sets WORKDIR=/app but this file lives in /app/src/.
# Using bare relative paths like "model_weights.msgpack" would look in /app/
# and fail silently (well, loudly with a red error box).
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Tumour Grade Predictor", layout="centered")


# ---------------------------------------------------------------------------
# MODEL ARCHITECTURE
# Exact replica of the training notebook. Do not touch field names or order —
# Flax serialisation is name-sensitive; a single rename breaks weight loading.
# ---------------------------------------------------------------------------
class HMIL_Flax(nn.Module):
    hidden_dim:   int
    n_classes:    int   = 3
    region_size:  int   = 100
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, mask, deterministic: bool):
        B, N, D = x.shape

        h = nn.Dense(self.hidden_dim, name="proj")(x)
        h = nn.relu(h)

        # ---- pad to multiple of region_size ----
        pad_len = (self.region_size - (N % self.region_size)) % self.region_size
        if pad_len > 0:
            h    = jnp.pad(h,    ((0, 0), (0, pad_len), (0, 0)))
            mask = jnp.pad(mask, ((0, 0), (0, pad_len)))

        num_regions = h.shape[1] // self.region_size

        # ---- local transformer ----
        h_reg = h.reshape((B * num_regions, self.region_size, self.hidden_dim))
        m_reg = mask.reshape((B * num_regions, self.region_size))
        lcls  = self.param(
            "local_cls", nn.initializers.normal(0.02),
            (1, 1, self.hidden_dim)
        )
        h_reg = jnp.concatenate(
            [jnp.broadcast_to(lcls, (B * num_regions, 1, self.hidden_dim)), h_reg], axis=1
        )
        lmask = jnp.concatenate(
            [jnp.ones((B * num_regions, 1), dtype=jnp.bool_), m_reg], axis=1
        )
        h_reg = h_reg + nn.SelfAttention(
            num_heads=4,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            name="local_trans",
        )(
            nn.LayerNorm(name="local_ln")(h_reg),
            mask=jnp.expand_dims(lmask, (1, 2)),
            deterministic=deterministic,
        )

        # ---- pool local CLS tokens ----
        h_pool = h_reg[:, 0, :].reshape((B, num_regions, self.hidden_dim))

        # ---- global transformer ----
        gvalid = m_reg.any(axis=1).reshape((B, num_regions))
        gcls   = self.param(
            "global_cls", nn.initializers.normal(0.02),
            (1, 1, self.hidden_dim)
        )
        h_glob = jnp.concatenate(
            [jnp.broadcast_to(gcls, (B, 1, self.hidden_dim)), h_pool], axis=1
        )
        gmask = jnp.concatenate(
            [jnp.ones((B, 1), dtype=jnp.bool_), gvalid], axis=1
        )
        h_glob = h_glob + nn.SelfAttention(
            num_heads=4,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            name="global_trans",
        )(
            nn.LayerNorm(name="global_ln")(h_glob),
            mask=jnp.expand_dims(gmask, (1, 2)),
            deterministic=deterministic,
        )

        # ---- classifier head ----
        z = nn.Dense(self.hidden_dim // 2, name="classifier_1")(h_glob[:, 0, :])
        z = nn.relu(z)
        z = nn.Dropout(self.dropout_rate)(z, deterministic=deterministic)
        return nn.Dense(self.n_classes, name="classifier_2")(z)


# ---------------------------------------------------------------------------
# MODEL LOADING — cached so it only runs once per container lifetime.
# Exact hyperparameters from v3 Trial 0 (best QWK=0.5861, F1=0.6296).
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(SCRIPT_DIR, "model_weights.msgpack")

    if not os.path.exists(model_path):
        st.error(
            f"Cannot find model_weights.msgpack in {SCRIPT_DIR}. "
            "Make sure it is committed inside the src/ folder."
        )
        st.stop()

    model = HMIL_Flax(hidden_dim=256, dropout_rate=0.18276391449291166)

    dummy_x    = jnp.ones((1, 100, 1024))
    dummy_mask = jnp.ones((1, 100), dtype=jnp.bool_)
    rng        = jax.random.PRNGKey(0)
    variables  = model.init(rng, dummy_x, dummy_mask, deterministic=True)

    with open(model_path, "rb") as f:
        loaded_params = serialization.from_bytes(variables["params"], f.read())

    return model, loaded_params


model, params = load_model()


# ---------------------------------------------------------------------------
# SAMPLE MAP
# ---------------------------------------------------------------------------
SAMPLES = {
    "Sample 1 — TCGA-BA-4078": "TCGA-BA-4078",
    "Sample 2 — TCGA-BA-4076": "TCGA-BA-4076",
    "Sample 3 — TCGA-BA-4074": "TCGA-BA-4074",
    "Sample 4 — TCGA-BA-5152": "TCGA-BA-5152",
    "Sample 5 — TCGA-BA-5149": "TCGA-BA-5149",
}

GRADE_LABELS = {0: "Grade 1 (Low)", 1: "Grade 2 (Intermediate)", 2: "Grade 3 (High)"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Tumour Grade Prediction — H-MIL")
st.write(
    "Select a pre-processed whole-slide image sample. "
    "The app runs live JAX inference using a Hierarchical MIL transformer "
    "and overlays gradient-based saliency on the WSI thumbnail."
)

selected_label = st.selectbox("Select a sample WSI:", list(SAMPLES.keys()))
case_id        = SAMPLES[selected_label]

pt_path    = os.path.join(SCRIPT_DIR, f"{case_id}.pt")
image_hits = (
    glob.glob(os.path.join(SCRIPT_DIR, f"{case_id}*.jpg")) +
    glob.glob(os.path.join(SCRIPT_DIR, f"{case_id}*.png"))
)
image_path = image_hits[0] if image_hits else None

if not os.path.exists(pt_path):
    st.error(f"Missing embedding file: {case_id}.pt in src/")
    st.stop()
if image_path is None:
    st.error(f"Missing thumbnail image for {case_id} in src/")
    st.stop()

st.subheader(f"WSI Thumbnail — {case_id}")
thumbnail = Image.open(image_path).convert("RGBA")
st.image(thumbnail, caption=f"Original WSI Thumbnail ({case_id})", use_container_width=True)

if st.button("Run Tumour Grade Inference", type="primary"):
    with st.spinner("Running H-MIL inference and computing saliency map…"):
        try:
            # ------------------------------------------------------------------
            # 1. Load UNI embeddings from .pt file
            # ------------------------------------------------------------------
            data = torch.load(pt_path, map_location="cpu", weights_only=True)

            if not (isinstance(data, dict) and "coords" in data and "features" in data):
                st.error("The .pt file must contain 'coords' and 'features' keys.")
                st.stop()

            coords     = data["coords"]    # (N, 2) patch coordinates in original WSI space
            embeddings = data["features"]  # (N, 1024) UNI patch embeddings

            # ------------------------------------------------------------------
            # 2. JAX inference + gradient saliency
            # ------------------------------------------------------------------
            feats = jnp.array(embeddings.numpy())[None, ...]   # (1, N, 1024)
            masks = jnp.ones((1, feats.shape[1]), dtype=jnp.bool_)

            def score_fn(f):
                logits     = model.apply({"params": params}, f, masks, deterministic=True)
                probs      = jax.nn.softmax(logits, axis=-1)[0]
                pred_class = jnp.argmax(probs)
                return probs[pred_class], (probs, pred_class)

            grad_fn = jax.value_and_grad(score_fn, has_aux=True)
            (_, (probs, predicted_class)), grads = grad_fn(feats)

            # grads[0] is (N, 1024) — batch dim already stripped by indexing
            # L2 norm over embedding dim → (N,) scalar importance per patch
            patch_importance = jnp.linalg.norm(grads[0], axis=-1)
            patch_importance = (
                (patch_importance - jnp.min(patch_importance))
                / (jnp.max(patch_importance) - jnp.min(patch_importance) + 1e-8)
            )
            patch_importance = np.array(patch_importance)

            predicted_class_idx = int(predicted_class)
            final_grade         = GRADE_LABELS[predicted_class_idx]
            confidence          = float(probs[predicted_class_idx]) * 100

            # ------------------------------------------------------------------
            # 3. Saliency heatmap overlay — proper 2D grid, not confetti dots
            # ------------------------------------------------------------------
            PATCH_SIZE = 256
            thumb_w, thumb_h = thumbnail.size

            x_coords = coords[:, 0].float()
            y_coords = coords[:, 1].float()

            orig_w = int(x_coords.max().item()) + PATCH_SIZE
            orig_h = int(y_coords.max().item()) + PATCH_SIZE

            # Percentile-based normalisation — min-max collapses everything
            # when one patch dominates the gradient magnitude
            p_low  = float(np.percentile(patch_importance, 50))  # ignore bottom half
            p_high = float(np.percentile(patch_importance, 99))  # clip top 1% outliers

            importance_clipped = np.clip(patch_importance, p_low, p_high)
            importance_norm    = (importance_clipped - p_low) / (p_high - p_low + 1e-8)

            # Build sparse importance grid in original WSI patch-grid space
            grid_cols = int(np.ceil(orig_w / PATCH_SIZE))
            grid_rows = int(np.ceil(orig_h / PATCH_SIZE))
            importance_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            count_grid      = np.zeros((grid_rows, grid_cols), dtype=np.float32)

            for i in range(len(coords)):
                x_orig  = int(coords[i][0].item())
                y_orig  = int(coords[i][1].item())
                col_idx = x_orig // PATCH_SIZE
                row_idx = y_orig // PATCH_SIZE
                if 0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols:
                    importance_grid[row_idx, col_idx] += importance_norm[i]
                    count_grid[row_idx, col_idx]      += 1

            # Average in case multiple patches map to same cell
            count_grid      = np.maximum(count_grid, 1)
            importance_grid /= count_grid

            # Bilinear upscale to thumbnail dimensions
            zoom_y       = thumb_h / grid_rows
            zoom_x       = thumb_w / grid_cols
            heatmap_full = zoom(importance_grid, (zoom_y, zoom_x), order=1)
            heatmap_full = np.clip(heatmap_full, 0, 1)

            # Power transform — squashes distribution toward 1.0 so mid-range
            # patches become visually loud instead of near-transparent whispers.
            # Without this, most values cluster near 0 after percentile norm
            # and the overlay is invisible at normal viewing distance.
            importance_display = np.power(heatmap_full, 0.4)

            # Solid cyan (R=0, G=220, B=255) — maximum contrast on H&E pink/purple.
            # Alpha only encodes importance — colour is constant and saturated.
            # Patches below threshold get zero alpha so whitespace stays clean.
            overlay_rgba = np.zeros((*heatmap_full.shape, 4), dtype=np.uint8)
            overlay_rgba[:, :, 0] = 0    # R
            overlay_rgba[:, :, 1] = 220  # G  } bright cyan
            overlay_rgba[:, :, 2] = 255  # B  }
            overlay_rgba[:, :, 3] = np.where(
                importance_display < 0.2,
                0,
                np.clip(importance_display * 230, 0, 255)
            ).astype(np.uint8)

            overlay = Image.fromarray(overlay_rgba, mode="RGBA")
            blended = Image.alpha_composite(thumbnail, overlay)

            # ------------------------------------------------------------------
            # 4. Results display
            # ------------------------------------------------------------------
            st.success("Inference complete.")

            st.markdown(f"### Predicted Tumour Grade: **{final_grade}**")
            st.markdown(f"**Model confidence:** {confidence:.1f}%")

            col1, col2, col3 = st.columns(3)
            col1.metric("Grade 1 prob", f"{float(probs[0]) * 100:.1f}%")
            col2.metric("Grade 2 prob", f"{float(probs[1]) * 100:.1f}%")
            col3.metric("Grade 3 prob", f"{float(probs[2]) * 100:.1f}%")

            st.image(
                blended,
                caption="Gradient saliency heatmap (cyan = high model attention)",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Inference failed: {e}")
            raise