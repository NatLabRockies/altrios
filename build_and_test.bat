call conda activate ./envs
cargo test --release
cd ..
maturin develop --release
pytest -v tests
