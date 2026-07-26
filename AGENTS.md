# Development environment

Use the Conda environment named `morse` for all Python commands, tests,
training, inference, and model conversion:

```shell
conda activate morse
```

For non-interactive commands, use:

```shell
conda run -n morse <command>
```

## macOS OpenMP test workaround

On macOS, importing PyTorch before NumPy can abort the Python process during
test collection because the Conda OpenMP runtimes are loaded in the wrong
order. Run pytest with NumPy imported first:

```shell
conda run -n morse python -c 'import numpy, pytest, sys; sys.exit(pytest.main(["-q"]))'
```

Add test paths or other pytest arguments to the list passed to
`pytest.main()`. Do not use a plain `python -m pytest` invocation in this
environment.
