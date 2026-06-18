## [Unreleased]
### Added
- `pricer/vol_surface_nn.py`: feedforward neural network vol surface interpolator with walk-forward split, StandardScaler normalisation, early stopping, and comparison utilities against the bicubic spline
- `tests/test_vol_surface_nn.py`: 48 tests, 100% coverage
- `notebooks/05_vol_surface_nn.ipynb`: end-to-end pipeline on SPY call data
