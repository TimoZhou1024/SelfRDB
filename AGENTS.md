# Repository Guidelines

## Project Structure & Module Organization

SelfRDB is a compact PyTorch Lightning project. `main.py` defines `BridgeRunner` and the Lightning CLI entrypoint for `fit` and `test`. `diffusion.py` contains the diffusion bridge logic, `datasets.py` contains `BaseDataset`, `NumpyDataset`, and `DataModule`, and `utils.py` handles metrics and image output. Network implementations live in `backbones/`; custom CUDA/C++ ops are under `backbones/op/`. Dataset preparation helpers are in `data/`, visual assets are in `figures/`, and runtime defaults are in `config.yaml`. The conda environment is defined by `requirements.yaml`.

## Build, Test, and Development Commands

- `conda env create --file requirements.yaml`: create the supported Python 3.8/CUDA 11.7 environment.
- `conda activate selfrdb`: activate the project environment.
- `python main.py fit --config config.yaml --trainer.logger.name EXP --data.dataset_dir DATA --data.source_modality T1 --data.target_modality T2`: train or resume a model; add `--ckpt_path PATH` to resume.
- `python main.py test --config config.yaml --data.dataset_dir DATA --data.source_modality T1 --data.target_modality T2 --ckpt_path PATH`: run inference and evaluation.
- `python data/create_brats_dataset.py`: run the BRATS conversion helper after checking the hard-coded paths in `main()`.

## Coding Style & Naming Conventions

Use Python 3.8-compatible code with 4-space indentation. Follow existing naming: `snake_case` for functions, variables, and config keys; `PascalCase` for classes; uppercase only for constants. Keep Lightning configuration in `config.yaml` and prefer CLI overrides for experiments. New dataset implementations should inherit from `BaseDataset` in `datasets.py` and return `(target, source, index)` like `NumpyDataset`.

## Testing Guidelines

No standalone test suite is currently checked in. Validate changes with the smallest reproducible Lightning run, for example `--trainer.max_epochs 1` on a small local dataset, then run `test` against a known checkpoint when touching inference, metrics, or sampling. If adding automated tests, place them under `tests/`, name files `test_*.py`, and use small synthetic NumPy arrays rather than real medical data.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Add seed parameter...`, `Fix metric computation...`, and `Update README.md`. Keep that style and scope commits narrowly. Pull requests should describe the change, list commands run, note dataset/config assumptions, and include metrics or sample-output paths when behavior changes.

## Security & Configuration Tips

Do not commit generated `logs/`, checkpoints, local datasets, or patient data. Keep machine-specific paths out of `config.yaml` unless they are safe defaults, and document required overrides in the PR.
