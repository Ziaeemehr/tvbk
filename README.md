# tvbk

Computational kernels for tvb.

## setup

My local dev setup is
in VS Code w/ Python, C/C++ extensions, and a venv setup for incremental rebuilds like so
```bash
rm -rf build env
uv venv env
source env/bin/activate
uv pip install nanobind 'scikit-build-core[pyproject]' pytest pytest-benchmark numpy cibuildwheel scipy 
uv pip install --no-build-isolation -Ceditable.rebuild=true -ve .
```
following https://nanobind.readthedocs.io/en/latest/packaging.html#step-5-incremental-rebuilds.
This enables editing and running the tests directly, with changes to the C++ automatically
taken into account, just running
```
pytest
```
will rebuild the C++ if required.  This also occurs on import in e.g. a Jupyter kernel.

## next

- add sde heun steppers
- all the neural mass models
- make first release to start integrating w/ TVB
- refactor buffers
- rm scipy dep for sparsity
- cuda/hip/webgpu or something
