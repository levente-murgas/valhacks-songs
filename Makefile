# ------------------------------- #
#	Valhacks Songs Hackathon
# ------------------------------- #

#---COLORS---#
GRAY="\033[1;30m"
RED="\033[1;31m"
GREEN="\033[32m"
HGREEN="\033[1;32m"
YELLOW="\033[1;33m"
LYELLOW="\033[33m"
BLUE="\033[1;34m"
LBLUE="\033[34m"
PINK="\033[35m"
CYAN="\033[1;36m"
WHITE="\033[0;37m"
NOCOLOR="\033[0m\033[K"
#------------#

VENV_PATH=./.venv/
VENV_PYTHON=${VENV_PATH}/bin/python3
VENV_PIP=${VENV_PATH}/bin/pip

SOURCE_PATH=./src/


help: ## Display this help message
	@echo
	@printf $(HGREEN)=$(BLUE)-------------------------------------------------$(HGREEN)=$(NOCOLOR)"\n"
	@printf $(HGREEN)="            Valhacks Song Hackathon              "=$(NOCOLOR)"\n"
	@printf $(HGREEN)=$(BLUE)-------------------------------------------------$(HGREEN)=$(NOCOLOR)"\n"
	@echo
	@printf $(LYELLOW)"Please use \`make <target>\` where <target> is one of\n\n"$(NOCOLOR)
	@for makefile in $(MAKEFILE_LIST); do \
		grep -E '(^[0-9a-zA-Z_-]+:.*?##.*$$)|(^##)' "$$makefile" ; \
	done | awk 'BEGIN {FS = ":.*?## "}; {printf $(GREEN)"%-30s"$(NOCOLOR)"%s\n", $$1, $$2}' | sed -e "s/\[32m##/[36m/"

.PHONY: help

venv: ## Create a .venv
	@python3 -m venv ${VENV_PATH}

requirements: requirements.txt ## Install Requirements
	@${VENV_PIP} install -r requirements.txt

pip: ## Run pip commands
	@${VENV_PIP} $(ARGS)

run: ## Run main.py
	@${VENV_PYTHON} ${SOURCE_PATH}/main.py

evaluator: ## Run evaluator.py
	@${VENV_PYTHON} ${SOURCE_PATH}/evaluator.py


.PHONY: venv requirements pip run