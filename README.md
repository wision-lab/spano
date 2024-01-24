# Getting Started

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


# Benchmarking

To run benchmarks simply run `cargo bench`. You can also create flamegraphs (only on linux) like so:
```
cargo bench --bench benchmarks -- --profile-time 10
```

The graphs will be in `target/criterion/*/profile/flamegraph.svg`.
