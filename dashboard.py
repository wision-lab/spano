"""
Run as: streamlit run dashboard.py
"""
import json
from pathlib import Path
import argparse

import numpy as np
import streamlit as st
from natsort import natsorted


def main(root):
    st.title("\nPanoramas from Photons")
    sequence = st.selectbox("Select Panorama:", natsorted([str(i) for i in Path(root).glob("*")]))
        
    for i, video in enumerate(natsorted(Path(sequence).glob("lvl-*.mp4"))):
        st.subheader(f"**Stabilized video, iteration={i+1}**")
        st.video(str(video), loop=True, autoplay=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    main(args.root)
