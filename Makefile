# Rerun the searches and checks behind the paper.
# Override any variable on the command line, e.g. make sweep-static NODE=0 NODES=3

PYTHON  ?= venv/bin/python
TRACE   ?= data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt
NODE    ?= 0
NODES   ?= 1
JOBS    ?= 1

.PHONY: help test mechanism placement budget2000 noise timing \
        sweep-static sweep-dynamic

help:
	@echo "targets:"
	@echo "  make test           unit tests"
	@echo "  make mechanism      decision-space coherence"
	@echo "  make placement      placement distinctness (edge-cloud)"
	@echo "  make budget2000     2000-call run on test day 3"
	@echo "  make noise          CV of the objectives vs K"
	@echo "  make timing         oracle call time"
	@echo "  make sweep-static   NODE=i NODES=n: raw and SLA day rotation, NSGA-II, MOEA/D, MORL"
	@echo "  make sweep-dynamic  NODE=i NODES=n: dynamic and edge-cloud day rotation, plus"
	@echo "                      random search for raw and SLA"
	@echo "the searches need the Azure trace at TRACE=$(TRACE) (see README)"

test:
	$(PYTHON) -m pytest tests/ -q

mechanism:
	mkdir -p results_mechanism_check
	$(PYTHON) -m experiments.mechanism_check --trace $(TRACE) --seeds 1,2,3,4,5 \
	    --spaces static,dynamic,hpa --skip-local \
	    --out results_mechanism_check/coherence_with_hpa.json

placement:
	mkdir -p results_placement_distinctness
	for s in 1 2 3 4 5 6 7 8 9 10; do \
	  $(PYTHON) -m experiments.placement_distinctness --seed $$s \
	    > results_placement_distinctness/seed$$s.txt; done

budget2000:
	PYTHON=$(PYTHON) WORKLOAD=azure TRACE=$(TRACE) BASE_SEED=42 TRAIN_DAYS=0,1,2 \
	    TEST_DAY=3 EVALS=2000 POP=50 OUT=results_2000 JOBS=$(JOBS) ./reproduce.sh

noise:
	$(PYTHON) -m experiments.cov_replications --configs 20 --kmax 40 --target 0.05 \
	    --workload azure --trace $(TRACE)

timing:
	$(PYTHON) -m experiments.timing_benchmark --k 5 --samples 100 --workload azure --trace $(TRACE)
	$(PYTHON) -m experiments.timing_benchmark --k 5 --samples 20 --workload azure --trace $(TRACE) --dynamic

sweep-static:
	bash experiments/day_rotation_sweep.sh $(NODE) $(NODES)

sweep-dynamic:
	bash experiments/dynamic_day_rotation_sweep.sh $(NODE) $(NODES)
