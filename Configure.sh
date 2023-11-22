#!/bin/sh

poetry install && \
    poetry run pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
