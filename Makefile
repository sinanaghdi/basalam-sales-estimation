.PHONY: install install-real smoke test prepare-data audit-real

install:
	pip install -e ".[dev]"

# For real-data work: adds pyarrow (Parquet conversion).
# RAR extraction additionally needs 7zz/7z/unrar/bsdtar on PATH.
install-real:
	pip install -e ".[dev,real]"

smoke:
	python -m basalam.run configs/experiments/smoke.yaml

test:
	python -m pytest -q

prepare-data:
	python scripts/prepare_data.py

audit-real:
	python -m basalam.run configs/experiments/audit_real.yaml
