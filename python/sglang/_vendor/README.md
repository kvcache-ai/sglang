# Private runtime dependencies

`compressed_tensors` is the released **0.15.0.1** implementation, privately
vendored as `sglang._vendor.compressed_tensors`. Its Apache-2.0 LICENSE, source
wheel URL, SHA256, and per-file transformation hashes accompany the package.

Why: the upstream distribution requires `transformers`, whereas KT's runtime
is distributed as `transformers-kt`. Installing both distributions into one
environment would give them overlapping `transformers/` files. SGLang instead
owns this private copy and declares its runtime dependencies explicitly. We do
not install upstream `compressed-tensors`, add a second `.dist-info`, manipulate
`sys.path`, or alias `sys.modules['compressed_tensors']`.

Only absolute internal imports and Loguru's module filter names are relocated.
Quantization algorithms, model format names, and Transformers imports are
unchanged. The latter resolve to the installed `transformers-kt` implementation.

To reproduce, download the wheel URL in `compressed_tensors/vendor-lock.json`
and run from the repository root:

```sh
python scripts/vendor_compressed_tensors.py --wheel /path/to/compressed_tensors-0.15.0.1-py3-none-any.whl
python scripts/vendor_compressed_tensors.py --wheel /path/to/compressed_tensors-0.15.0.1-py3-none-any.whl --check
```

Wheel builds use this checked-in tree and do not download dependencies at build
time. Updating the pinned upstream version requires a reviewed lock/transform
update plus fresh import, quantization and model-serving regression tests.
