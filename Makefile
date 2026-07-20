.DEFAULT_GOAL := help
PROFILE ?= smoke
RUN := uv run
export MPLBACKEND := Agg
export JAX_PLATFORMS := cpu
export XLA_PYTHON_CLIENT_PREALLOCATE := false
export PYTHONHASHSEED := 0

help:            ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

setup:           ## Install deps + hooks
	uv sync --extra dev && $(RUN) pre-commit install
lock:            ## Refresh the lockfile
	uv lock

lint:            ## ruff check + format check
	$(RUN) ruff check . && $(RUN) ruff format --check .
typecheck:       ## mypy
	$(RUN) mypy
test:            ## Fast unit tests
	$(RUN) pytest -m "not slow" -n auto
test-slow:       ## Scientific gates
	$(RUN) pytest -m slow

data:            ## Stage 1
	$(RUN) python scripts/generate_world.py profile=$(PROFILE) && $(RUN) python scripts/make_data.py profile=$(PROFILE)
graphs:          ## Stage 2
	$(RUN) python scripts/build_graphs.py profile=$(PROFILE)
abm:             ## Stage 3 (+3b)
	$(RUN) python scripts/run_abm.py profile=$(PROFILE)
calibrate:       ## Stage 4  (JAX in a subprocess)
	$(RUN) python scripts/calibrate.py profile=$(PROFILE)
causal:          ## Stage 5
	$(RUN) python scripts/causal_analysis.py profile=$(PROFILE) && $(RUN) python scripts/validate_simulator.py profile=$(PROFILE)
emulator:        ## Stages 6-7
	$(RUN) python scripts/train_emulator.py profile=$(PROFILE)
eval-emulator:   ## Stage 6-7 §11 evaluation suite
	$(RUN) python scripts/eval_emulator.py profile=$(PROFILE)
rl:              ## Stage 10
	$(RUN) python scripts/train_rl.py profile=$(PROFILE)
ensemble:        ## Stage 11
	$(RUN) python scripts/run_ensemble.py profile=$(PROFILE)
sensitivity:     ## Stage 14
	$(RUN) python scripts/sensitivity.py profile=$(PROFILE)
figures:         ## Stage 15
	$(RUN) python scripts/make_figures.py profile=$(PROFILE)
report:          ## Stage 17
	$(RUN) python scripts/build_report.py profile=$(PROFILE)

all:             ## Full pipeline, profile=dev
	$(RUN) python scripts/run_pipeline.py profile=dev
smoke:           ## Full pipeline, CPU, < 6 min  (CI gate)
	$(RUN) python scripts/run_pipeline.py profile=smoke && $(RUN) pytest -m slow -q
paper:           ## Full pipeline, profile=full (cluster)
	$(RUN) python scripts/run_pipeline.py profile=full compute=ray_cluster

sweep:           ## Hydra multirun example
	$(RUN) python scripts/run_pipeline.py -m profile=dev seed=0,1,2,3,4 emulator=rssm_gnn,rssm_flat,gru_baseline
slurm:           ## Same sweep via submitit
	$(RUN) python scripts/run_pipeline.py -m hydra/launcher=submitit_slurm profile=full seed=0,1,2,3,4

dashboard:       ## Streamlit
	$(RUN) streamlit run scripts/dashboard.py --server.headless true --server.port 8501

docker-build:
	docker build -f docker/Dockerfile -t regworld:latest .
docker-run:
	docker run --rm -v $$PWD/artifacts:/work/artifacts regworld:latest python scripts/run_pipeline.py profile=$(PROFILE)

clean:
	rm -rf experiments/* artifacts/* reports/figures/* .pytest_cache .mypy_cache

.PHONY: help setup lock lint typecheck test test-slow data graphs abm calibrate causal emulator eval-emulator \
        rl ensemble sensitivity figures report all smoke paper sweep slurm dashboard \
        docker-build docker-run clean
