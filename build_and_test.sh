# assumes a python environment has been created and activated
echo "Testing rust" && \
cargo test --workspace && \
echo "Building python API" && \
pip install -qe ".[dev]" && \ 
echo "Running python tests" && \
pytest -v python/altrios/tests && \
echo "Verifying that demos run" && \
pytest -v python/altrios/demos && \
echo "Complete success!"
