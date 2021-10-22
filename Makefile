DEFAULT_VENV_DIR=.venv

.PHONY: all py38

all: data py38

py38: $(DEFAULT_VENV_DIR)/py38/

$(DEFAULT_VENV_DIR)/py38/:
	python3.8 -m venv $@
	$@/bin/pip install -U pip
	$@/bin/pip install -r pip-requirements.txt
	$@/bin/pip install -r pip-requirements-dev.txt
	$@/bin/python3 -m ipykernel install --user --name=pred-as-os
	@echo "Run \`source $@/bin/activate\` to start the virtual env."

data:
	wget https://zenodo.org/record/5618882/files/UCC2021-data.tar.bz2
	tar --use-compress-program=pbzip2 -xvf UCC2021-data.tar.bz2