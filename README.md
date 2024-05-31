# Getting Started

## Installing

First make sure you have an adequate rust toolchain installed ([install from here](https://rustup.rs/)), and gcc installed. 
Due to limitations in OpalKelly, we are currently limited to py36.

It's best practice to create a virtual environment, you can use micromamba to do so: 
```shell
micromamba create --name py36 python=3.6
```

Then activate your environment, and in it install `spano`, either directly form this repo:
```shell
pip install -v git+https://github.com/WISION-Lab/spano
```

Or you can clone the project and, in the project's directory, run:
```shell 
pip install -v .
```

This will compile the library from source so it will take some time. 

## Examples

### Registration of two images

Register two images using Inverse Compositional Lucas-Kanade at a fixed scale, and then again at multiple scales (image pyramid). 

On a Ryzen 5600G, single scale takes 3956 steps and ~6.5 seconds while multi scales takes 262 steps and ~0.15 seconds (best of 3 runs, without visualization or params dump).

```shell
spano lk -i assets/madison1.png -i assets/madison2.png -o out.png --downscale=2 --iterations=5000 --early-stop=1e-3 --viz-output=out.mp4 --params-path=params.json

spano lk -i assets/madison1.png -i assets/madison2.png -o out-multi.png --downscale=2 --iterations=5000 --early-stop=5e-3 --viz-output=out-multi.mp4 --params-path=params-multi.json --multi

# Plot convergence of the two runs
inv plot --path params.json --path params-multi.json

# Here's a harder example with different sized images and a large warp (takes ~6.4 seconds):
spano lk -i assets/skyline1.png -i assets/skyline2.png -o out-multi.png --downscale=2 --iterations=5000 --early-stop=1e-3 --multi
```

### Panorama from photoncube

You can create a panorama using a photoncube (in npy format) as well. Using [this photoncube](https://drive.google.com/file/d/1rTTD6wBLveElNyb_xNtfgPw3trQCY9tN/view?usp=sharing) and the appropriate inpainting and color filter arrays [found here](https://github.com/WISION-Lab/photoncube2video) you can create a panorama and the stabilized videos at each level like so:
```shell
spano pano -i binary.npy -s 0 -e 64000 -t rot90 -t flip-ud --colorspad-fix --granularity=32 --viz-step=10 --early-stop=1e-4 --viz-output=out.mp4 --inpaint-path ../photoncube2video/colorspad_inpaint_mask.npy --cfa-path ../photoncube2video/rgbw_oh_bn_color_ss2_corrected.png
```
On a 5600G, this takes about 1min 20s. 


# Development

We use [maturin](https://www.maturin.rs/) as a build system, which enables this package to be built as a python extension using [pyo3](https://pyo3.rs) bindings. Some other tools are needed for development work, which can be installed using `pip install -v .[dev]`.

#### Code Quality

We use `rustfmt` to format the codebase, we use some customizations (i.e for import sorting) which require nightly. First ensure you have the nightly toolchain installed with:
```shell
rustup toolchain install nightly
```

Then you can format the code using:

```shell
cargo +nightly fmt 
```

Similarly we use `black` to format the python parts of the project. 


To keep the project lean, it's recommended to check for unused dependencies [using this tool](https://github.com/est31/cargo-udeps), or [this one](https://github.com/bnjbvr/cargo-machete), like so: 

```shell
cargo +nightly udeps
cargo machete --with-metadata
```


## Benchmarking

To run benchmarks simply run `cargo bench`. You can also create flamegraphs (only on linux) like so:
```shell
cargo bench --bench benchmarks -- --profile-time 10
```

The graphs will be in `target/criterion/*/profile/flamegraph.svg`.
