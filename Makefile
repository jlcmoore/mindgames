.PHONY: init test build jslint pylint release-% get-data

init: build
	python3.11 -m venv env-mindgames
	env-mindgames/bin/pip install -r requirements.txt
	env-mindgames/bin/python -m ipykernel install --user --name "env-mindgames"
	env-mindgames/bin/pip install --editable .

release-%:    ## e.g. `make release-mindgames` or `make release-mindgames-public`
	git pull
	$(MAKE) build
	- rm database.db
	- rm logs/main.log
	sudo systemctl restart $(@:release-%=%)

# TODO: do not include this in the public release
get-data:
	scp mindgames@68.183.47.5:/srv/mindgames/logs/main.log main.log
	scp mindgames@68.183.47.5:/srv/mindgames/database.db database.db

init-conda:
	conda env create --file environment.yml
	conda activate env-mindgames && \
	pip install --editable .

build:
	cd frontend && npm install
	cd frontend && npm run build

jslint:
	cd frontend &&  npx eslint --cache --fix

pylint:
	env-mindgames/bin/black src
	source env-mindgames/bin/activate && darker --lint "pylint --jobs 0" src

# PYTHONPATH=src so that the working directory for the commands is the top level directory above the package.
# This might be hacky
# For local resource-light tests
test:
	env-mindgames/bin/pytest src/tests -W error

# For networked resources such as API or LLM calls.
test-query:
	RUN_QUERY_TESTS=True env-mindgames/bin/pytest src/tests -W error

# For networked resources such as API or LLM calls that have batching -- can take ages!
test-batch:
	RUN_BATCH_TESTS=True env-mindgames/bin/pytest src/tests -W error

# For GPU resources such as vllm calls
test-gpu:
	RUN_GPU_TESTS=True env-mindgames/bin/pytest src/tests -W error

# For all calls, including the CSP.
test-all:
	RUN_ALL_TESTS=True \
	RUN_QUERY_TESTS=True \
	RUN_GPU_TESTS=True \
	RUN_BATCH_TESTS=True \
	env-mindgames/bin/pytest src/tests -W error
