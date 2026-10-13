# Original training environment

`requirements.txt` and `setup_environment.sh` preserve the environment recipe
used with the paper's original Unsloth training scripts. They are not a lockfile
or the installation instructions for every subsequent BF16/v9 experiment.

The setup script installs packages into its current Python environment, applies
version-specific Unsloth patches and clears its listed caches. Review it in an
isolated environment before use; repository setup does not run it automatically.
Use each experiment's recorded dependency versions when reproducing that run.

MobileWorld support dependencies are managed separately by
[`MobileWorld/pyproject.toml`](../../MobileWorld/pyproject.toml) and its lockfile.
The OPSD tests use their [recorded native environment](../docs/opsd.md).
