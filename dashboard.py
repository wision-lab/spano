"""
Run as: streamlit run dashboard.py
"""

from pathlib import Path
import argparse
import os

import streamlit as st
from natsort import natsorted


def show_video(path, subheader=None, **kwargs):
    if Path(path).exists():
        if subheader:
            st.subheader(subheader)
        st.video(str(path), **kwargs)
        return True
    return False


def main(root):
    st.set_page_config(layout="wide")
    st.title("\nPanoramas from Photons")

    with st.sidebar:
        sequence = Path(
            st.selectbox(
                "Select Sequence:",
                sorted(
                    [
                        str(i)
                        for i in Path(root).glob("*")
                        if i.is_dir() and not i.stem.startswith(".")
                    ],
                    key=os.path.getmtime,
                    reverse=True,
                ),
            )
        )

    with st.expander("**Captured Data**", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            bin_paths = list(sequence.glob("binary*.mp4"))

            if bin_paths:
                show_video(
                    bin_paths[0],
                    subheader="**Raw Data**",
                    loop=True,
                    autoplay=True,
                )
        with col2:
            avg_paths = list(sequence.glob("avg*.mp4"))

            if avg_paths:
                show_video(
                    avg_paths[0],
                    subheader="**Naive Reconstruction**",
                    loop=True,
                    autoplay=True,
                )

    with st.expander("**Stabilized Video**", expanded=False):
        lvls = natsorted(sequence.glob("lvl-*.mp4"))
        tabs = st.tabs([f"   Level #{i+1}   " for i in range(len(lvls))])
        for tab, video in zip(tabs, lvls):
            with tab:
                show_video(video, loop=True, autoplay=True)

    with st.expander("**Reconstructed Panorama**", expanded=False):
        if (pano_path := sequence / "out.png").exists():
            # See: https://discuss.streamlit.io/t/how-can-i-center-a-picture/30995/3
            st.markdown(
                """
                <style>
                    button[title^=Exit]+div [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.image(str(pano_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    main(args.root)
