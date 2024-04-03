# Getting Started

## Building

To compile it locally, simply make sure you have an adequate rust toolchain installed ([install from here](https://rustup.rs/)).

You can install directly using
```
cargo install --git="https://github.com/WISION-Lab/spano"
```

Which will create an executable and put in on your path (i.e: in ~/.cargo/bin on linux by default). Or you can clone this project, `cd` into it, and run:

```
cargo install --path .
```

Which will behave like the above command, just not using a tmp folder, or with

```
cargo build --release
```

Which will put the executable in `target/release`.

## Examples

Register two images using Inverse Compositional Lucas-Kanade at a fixed scale, and then again at mutiple scales (image pyramid). 

On a Ryzen 5600G, single scale takes 3956 steps and ~6.5 seconds while multi scales takes 262 steps and ~0.15 seconds (best of 3 runs, without visualization or params dump).

```
spano lk -i assets/madison1.png -i assets/madison2.png -o out.png --downscale=2 --iterations=5000 --early-stop=1e-3 --viz-output=out.mp4 --params-path=params.json

spano lk -i assets/madison1.png -i assets/madison2.png -o out-multi.png --downscale=2 --iterations=5000 --early-stop=5e-3 --viz-output=out-multi.mp4 --params-path=params-multi.json --multi

# Plot convergence of the two runs
inv plot --path params.json --path params-multi.json

# Here's a harder example with different sized images and a large warp (takes ~6.4 seconds):
spano lk -i assets/skyline1.png -i assets/skyline2.png -o out-multi.png --downscale=2 --iterations=5000 --early-stop=1e-3 --multi
```

# Benchmarking

To run benchmarks simply run `cargo bench`. You can also create flamegraphs (only on linux) like so:
```
cargo bench --bench benchmarks -- --profile-time 10
```

The graphs will be in `target/criterion/*/profile/flamegraph.svg`.
