"""Streamlit interface for the M.Tech image-fusion project."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import torch

from defusion_mtech.baselines import FUSION_METHODS, fuse
from defusion_mtech.image_io import align_pair, encode_png, load_rgb
from defusion_mtech.inference import fuse_with_model, load_model
from defusion_mtech.metrics import fusion_metrics

METHOD_LABELS = {
    "average": "Average",
    "pca": "PCA",
    "laplacian": "Laplacian pyramid",
    "wavelet": "Wavelet",
    "local_focus": "Local-focus weighted",
    "cud": "Self-supervised CUD model",
}


@st.cache_resource(show_spinner=False)
def cached_model(checkpoint: str, device: str):
    return load_model(Path(checkpoint), device=device)


def main() -> None:
    st.set_page_config(page_title="DeFusion M.Tech Project", page_icon="🧩", layout="wide")
    st.title("Label-Free Image Fusion")
    st.caption(
        "An educational common/unique feature-decomposition system for multi-focus, "
        "multi-exposure, and infrared-visible image pairs."
    )

    with st.sidebar:
        st.header("Fusion settings")
        task = st.selectbox(
            "Application",
            ["Multi-focus", "Multi-exposure", "Infrared-visible", "General aligned pair"],
        )
        method = st.selectbox(
            "Method",
            [*FUSION_METHODS, "cud"],
            format_func=lambda name: METHOD_LABELS[name],
            index=2,
        )
        alignment = st.selectbox(
            "Size handling",
            ["strict", "resize_b_to_a", "center_crop"],
            format_func={
                "strict": "Require identical sizes",
                "resize_b_to_a": "Resize source B to source A",
                "center_crop": "Center-crop both to shared size",
            }.get,
        )
        checkpoint = ""
        if method == "cud":
            checkpoint = st.text_input("Trained checkpoint path", placeholder="runs/cud/final.pt")
            st.info(
                "A trained checkpoint is required. The app never presents random-weight output "
                "as a result."
            )
        st.warning(
            "Inputs must depict the same registered scene. Size matching does not correct camera "
            "motion or cross-sensor misalignment."
        )

    upload_a, upload_b = st.columns(2)
    with upload_a:
        source_a = st.file_uploader(
            "Source image A", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
        )
    with upload_b:
        source_b = st.file_uploader(
            "Source image B", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
        )

    if not source_a or not source_b:
        st.info("Upload two aligned source images to begin.")
        methodology()
        return

    try:
        image_a, image_b = align_pair(load_rgb(source_a), load_rgb(source_b), alignment)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return

    preview_a, preview_b = st.columns(2)
    preview_a.image(image_a, caption=f"Source A - {image_a.shape[1]} × {image_a.shape[0]}")
    preview_b.image(image_b, caption=f"Source B - {image_b.shape[1]} × {image_b.shape[0]}")

    if not st.button("Fuse images", type="primary", use_container_width=True):
        methodology()
        return

    try:
        started = time.perf_counter()
        components = None
        if method == "cud":
            if not checkpoint.strip():
                raise ValueError("Enter the path to a trained CUD checkpoint")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = cached_model(checkpoint.strip(), device)
            fused, components = fuse_with_model(model, image_a, image_b)
        else:
            fused = fuse(image_a, image_b, method)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = fusion_metrics(image_a, image_b, fused)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Fusion failed: {error}")
        return

    result, indicators = st.columns([2, 1])
    with result:
        st.image(fused, caption=f"Fused result - {METHOD_LABELS[method]}")
        st.download_button(
            "Download fused PNG",
            data=encode_png(fused),
            file_name=f"fused_{method}.png",
            mime="image/png",
            use_container_width=True,
        )
    with indicators:
        st.subheader("Run summary")
        st.write(f"**Application:** {task}")
        st.write(f"**Method:** {METHOD_LABELS[method]}")
        st.write(f"**Fusion time:** {elapsed_ms:.1f} ms")
        st.caption("Time is one local run, not a benchmark or FPS claim.")
        st.metric("Entropy", f"{metrics['entropy_bits']:.4f} bits")
        st.metric("Spatial frequency", f"{metrics['spatial_frequency']:.4f}")
        st.metric("Mutual information sum", f"{metrics['mutual_information_sum']:.4f}")
        st.metric("Source-SSIM proxy", f"{metrics['source_ssim_proxy']:.4f}")
        st.caption(
            "Source-SSIM proxy is the average SSIM to both inputs; it is not ground-truth SSIM."
        )

    if components:
        with st.expander("Common and unique projections"):
            common, unique_a, unique_b = st.columns(3)
            common.image(components["common"], caption="Common projection")
            unique_a.image(components["unique1"], caption="Unique projection A")
            unique_b.image(components["unique2"], caption="Unique projection B")
    methodology()


def methodology() -> None:
    with st.expander("Methodology and academic-use notice"):
        st.markdown(
            """
            The CUD model is an independent educational implementation inspired by Liang et al.,
            *Fusion from Decomposition* (ECCV 2022). During self-supervised training, one clean
            image is converted into two mask-and-noise views. The network predicts their shared
            region, each view's unique region, and a reconstruction of the clean scene.

            The classical methods run without training. The CUD method requires a trained local
            checkpoint. This repository does not claim that the report's SSIM or 60 FPS figures
            have been reproduced; use the evaluation protocol and report measured results only.
            """
        )


if __name__ == "__main__":
    main()
