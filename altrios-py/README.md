# How to update .pyi file
1. Install mypy: `pip install mypy`
2. Get the html documentation: 
```bash
    cd ALTRIOS/altrios-core
    cargo doc --open
```
3. Go to the python folder in Altrios: 
```bash
    cd ALTRIOS/altrios-py/python/
```
4. Generate a new pyi file:
```bash
    stubgen altrios_pyo3/
```
5. You should see a new file `out/altrios_pyo3.pyi` and within it there will be stubs for all the classes and functions in the `altrios_pyo3` module

6. Go to the html documentation that was opened and search the classes, update the `out/altrios_pyo3.pyi` accordingly.
