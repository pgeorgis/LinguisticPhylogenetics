# Define the default target
.DEFAULT_GOAL := help
SHELL := /bin/bash
TEST_DATASETS = romance germanic baltoslavic sinitic
COVERAGE_REPORT_CONFIG = datasets/BaltoSlavic/config/test_config_minimal.yml

# Usage help
help:
	@echo "Usage:"
	@echo "  make init          # Run ./setup.sh"
	@echo "  make classify CONFIG=<path to config.yml> [LOGLEVEL=<desired log level>]"

# Run the setup script to install dependencies, clone submodules, and activate virtual environment
init:
	./setup.sh

init-silent:
	$(MAKE) init > /dev/null 2>&1

sync-submodules:
	git submodule update --init --recursive

classify:
ifdef CONFIG
	@source venv/bin/activate && \
		OUTPUT_LOG_DIR=$$(python3 -c "import os; from phyloLing.utils.utils import load_config; config = load_config('$(CONFIG)'); print(config.get('family', {}).get('outdir', os.path.dirname(config['family']['file'])))") && \
		OUTPUT_LOG_PATH=$$OUTPUT_LOG_DIR/logs/classify.log && \
		mkdir -p $$OUTPUT_LOG_DIR/logs && \
		python3 phyloLing/classifyLangs.py $(CONFIG) $(if $(LOGLEVEL),--loglevel $(LOGLEVEL)) 2>&1|tee $$OUTPUT_LOG_PATH
else
	@echo "Error: Please provide a path to config.yml using 'make classify CONFIG=<path> [LOGLEVEL=<desired log level>]'"
endif

classify-romance:
	$(MAKE) classify CONFIG=datasets/Romance/config/romance_config.yml

classify-germanic:
	$(MAKE) classify CONFIG=datasets/Germanic/config/germanic_config.yml

classify-baltoslavic:
	$(MAKE) classify CONFIG=datasets/BaltoSlavic/config/baltoslavic_config.yml

classify-sinitic:
	$(MAKE) classify CONFIG=datasets/Sinitic/config/sinitic_config.yml

test:
ifndef DATASET
	@echo "Error: Please provide a dataset to test using 'make test DATASET=<dataset>'"
	exit 1
endif

ifneq ($(filter $(DATASET),$(TEST_DATASETS)),)
	@source venv/bin/activate && \
		export PYTHONPATH=$(shell pwd):$$PYTHONPATH && \
		python3 -m xmlrunner phyloLing.test.test_$(DATASET)$(if $(TESTSET),.$(TESTSET),) -o tests
else
	@echo "Error: Invalid dataset. Please choose from $(TEST_DATASETS)" &&
		exit 1
endif

test-minimal:
ifndef DATASET
	@echo "Error: Please provide a dataset to test using 'make test-determinism DATASET=<dataset>'"
	exit 1
endif
	$(MAKE) test DATASET=$(DATASET) TESTSET=TestDeterminism
	$(MAKE) test DATASET=$(DATASET) TESTSET=TestMinimalTreeDistance

test-romance:
	$(MAKE) test-minimal DATASET=romance

test-germanic:
	$(MAKE) test-minimal DATASET=germanic

test-baltoslavic:
	$(MAKE) test-minimal DATASET=baltoslavic

test-sinitic:
	$(MAKE) test-minimal DATASET=sinitic

test-all:
	$(MAKE) -j test-romance test-germanic test-baltoslavic test-sinitic

test-tree-distance:
ifndef DATASET
	@echo "Error: Please provide a dataset to test using 'make test-tree-distance DATASET=<dataset>'"
	exit 1
endif
	$(MAKE) test DATASET=$(DATASET) TESTSET=TestTreeDistance

test-tree-distance-romance:
	$(MAKE) test-tree-distance DATASET=romance

test-tree-distance-germanic:
	$(MAKE) test-tree-distance DATASET=germanic

test-tree-distance-baltoslavic:
	$(MAKE) test-tree-distance DATASET=baltoslavic

test-tree-distance-sinitic:
	$(MAKE) test-tree-distance DATASET=sinitic

coverage: init-silent
	@source venv/bin/activate && \
		export PYTHONPATH=$(shell pwd):$$PYTHONPATH && \
		coverage erase && \
		coverage run phyloLing/classifyLangs.py $(COVERAGE_REPORT_CONFIG) && \
		coverage xml && \
		coverage html && \
		coverage report
