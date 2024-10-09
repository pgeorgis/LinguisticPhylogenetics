# Define the default target
.DEFAULT_GOAL := help
SHELL := /bin/bash

# Usage help
help:
	@echo "Usage:"
	@echo "  make init          # Run ./setup.sh"
	@echo "  make classify CONFIG=<path to config.yml> [LOGLEVEL=<desired log level>]"

# Run the setup script to install dependencies, clone submodules, and activate virtual environment
init:
	./setup.sh

# Run classification with nohup
classify: init
ifdef CONFIG
	@source venv/bin/activate && \
		OUTPUT_LOG_DIR=$$(python3 -c "import yaml, os; config = yaml.safe_load(open('$(CONFIG)', 'r')); print(config.get('family', {}).get('outdir', os.path.dirname(config['family']['file'])))") && \
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

classify-slavic:
	$(MAKE) classify CONFIG=datasets/Slavic/config/slavic_config.yml

# TODO score against a gold tree
