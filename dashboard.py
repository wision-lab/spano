"""
Run as: streamlit run dashboard.py
"""

from pathlib import Path
import argparse
import os

import streamlit as st
from streamlit_image_comparison import *
from natsort import natsorted


# Override the original `image_comparison` to include a vertical arg.
def image_comparison(
	img1: Union[Image.Image, str, np.ndarray],
	img2: Union[Image.Image, str, np.ndarray],
	label1: str = "1",
	label2: str = "2",
	width: int = 704,
	show_labels: bool = True,
	starting_position: int = 50,
	make_responsive: bool = True,
	in_memory: bool = False,
    vertical: bool = False
) -> components.html:
	"""
	Create a comparison slider for two images.
	
	Parameters
	----------
	img1: str, PIL Image, or numpy array
		Data for the first image.
	img2: str, PIL Image, or numpy array
		Data for the second image.
	label1: str, optional
		Label for the first image. Default is "1".
	label2: str, optional
		Label for the second image. Default is "2".
	width: int, optional
		Width of the component in pixels. Default is 704.
	show_labels: bool, optional
		Whether to show labels on the images. Default is True.
	starting_position: int, optional
		Starting position of the slider as a percentage (0-100). Default is 50.
	make_responsive: bool, optional
		Whether to enable responsive mode. Default is True.
	in_memory: bool, optional
		Whether to handle pillow to base64 conversion in memory without saving to local. Default is False.

	Returns
	-------
	components.html
		Returns a static component with a timeline
	"""
	# Prepare images
	img1_pillow = read_image_as_pil(img1)
	img2_pillow = read_image_as_pil(img2)

	img_width, img_height = img1_pillow.size
	h_to_w = img_height / img_width
	height = int((width * h_to_w) * 0.95)

	if in_memory:
		# Convert images to base64 strings
		img1 = pillow_to_base64(img1_pillow)
		img2 = pillow_to_base64(img2_pillow)
	else:
		# Create base64 strings from temporary files
		os.makedirs(TEMP_DIR, exist_ok=True)
		for file_ in os.listdir(TEMP_DIR):
			if file_.endswith(".jpg"):
				os.remove(os.path.join(TEMP_DIR, file_))
		img1 = pillow_local_file_to_base64(img1_pillow, TEMP_DIR)
		img2 = pillow_local_file_to_base64(img2_pillow, TEMP_DIR)

	# Load CSS and JS
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# write html block
	htmlcode = f"""
		<style>body {{ margin: unset; }}</style>
		{css_block}
		{js_block}
		<div id="foo" style="height: {height}; width: {width or '100%'};"></div>
		<script>
		slider = new juxtapose.JXSlider('#foo',
			[
				{{
					src: '{img1}',
					label: '{label1}',
				}},
				{{
					src: '{img2}',
					label: '{label2}',
				}}
			],
			{{
				animate: true,
				showLabels: {'true' if show_labels else 'false'},
				showCredits: true,
				startingPosition: "{starting_position}%",
				makeResponsive: {'true' if make_responsive else 'false'},
                mode: "{'vertical' if vertical else 'horizontal'}"
			}});
		</script>
		"""
	static_component = components.html(htmlcode, height=height, width=width)

	return static_component


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
        latest = sorted(
            [
                str(i)
                for i in Path(root).glob("spano*")
                if i.is_dir() and not i.stem.startswith(".") and i.name != "spano"
            ],
            key=os.path.getmtime,
            reverse=True,
        )
        samples = natsorted(
            str(i)
            for i in Path(root).glob("samples/")
            if i.is_dir() and not i.stem.startswith(".")
        )
        short_names = {str(Path(i).name.lstrip("spano_")): i for i in samples + latest}
        short_name = st.selectbox("Select Sequence:", short_names.keys())
        sequence = Path(short_names[short_name])

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
        if (pano_path := sequence / "panorama.png").exists():
            # # See: https://discuss.streamlit.io/t/how-can-i-center-a-picture/30995/3
            # st.markdown(
            #     """
            #     <style>
            #         button[title^=Exit]+div [data-testid=stImage]{
            #             text-align: center;
            #             display: block;
            #             margin-left: auto;
            #             margin-right: auto;
            #             width: 100%;
            #         }
            #     </style>
            #     """,
            #     unsafe_allow_html=True,
            # )
            # st.image(str(pano_path))

            component = image_comparison(
                img1=str(sequence / "baseline.png"),
                img2=str(pano_path),
                vertical=True
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    main(args.root)
