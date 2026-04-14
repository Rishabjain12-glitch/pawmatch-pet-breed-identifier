""" app.py — Pet Breed Identifier | Streamlit UI
-------------------------------------------------
Run with: streamlit run app.py
"""

import io
import textwrap
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawMatch — Pet Breed Identifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Lazy import to avoid loading model at import time ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from breed_identifier import BreedIdentifier
    return BreedIdentifier()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.title("🐾 PawMatch — Pet Breed Identifier")
st.markdown("> Upload two pet photos to check if they're the **same breed**.")

st.divider()

# ── Upload section ─────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("📷 Pet Image 1")
    f1 = st.file_uploader("Image 1", type=["jpg", "jpeg", "png", "webp"], key="img1", label_visibility="collapsed")
    if f1:
        st.image(Image.open(f1), use_container_width=True)

with col_r:
    st.subheader("📷 Pet Image 2")
    f2 = st.file_uploader("Image 2", type=["jpg", "jpeg", "png", "webp"], key="img2", label_visibility="collapsed")
    if f2:
        st.image(Image.open(f2), use_container_width=True)

st.divider()

# ── Compare button ─────────────────────────────────────────────────────────────
if st.button("🔍 Identify & Compare", type="primary", disabled=not (f1 and f2)):
    with st.spinner("Loading CLIP model and analysing images…"):
        identifier = load_model()
        img1 = Image.open(f1)
        img2 = Image.open(f2)
        result = identifier.compare(img1, img2)

    # ── Verdict banner ────────────────────────────────────────────────────────
    if result.same_breed:
        st.success(f"✅ SAME BREED — {result.verdict}")
    else:
        st.error(f"❌ DIFFERENT BREEDS — {result.verdict}")

    st.metric("Confidence", f"{result.confidence * 100:.1f}%")
    st.metric("Cosine Similarity", f"{result.similarity_score:.4f}")

    st.divider()

    # ── Per-image predictions ─────────────────────────────────────────────────
    res_l, res_r = st.columns(2)

    with res_l:
        p = result.image1_prediction
        icon = "🐕" if p.pet_type == "dog" else "🐈"
        st.subheader(f"{icon} Image 1: {p.breed}")
        st.write(f"Confidence: **{p.confidence * 100:.1f}%**")
        for rank, (breed, score) in enumerate(p.top_3, 1):
            st.write(f"{rank}. {breed} — {score * 100:.1f}%")

    with res_r:
        p = result.image2_prediction
        icon = "🐕" if p.pet_type == "dog" else "🐈"
        st.subheader(f"{icon} Image 2: {p.breed}")
        st.write(f"Confidence: **{p.confidence * 100:.1f}%**")
        for rank, (breed, score) in enumerate(p.top_3, 1):
            st.write(f"{rank}. {breed} — {score * 100:.1f}%")

elif not (f1 and f2):
    st.info("Upload both images above, then click **Identify & Compare**.")
