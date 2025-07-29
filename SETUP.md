# Technical set up

## Set-up

### For macOS/Linux - Install:

- `Make` (e.g. `brew install make`)

- `python>=3.10` (e.g. `brew install python@3.10`)

- [`jupyter`](https://jupyter.org/install).

- Install [XQuartz](https://www.xquartz.org/) if on a mac an trying to run the Rscripts

Run `make init`

Run `source env-mindgames/bin/activate`

### On a GPU machine

`$ make init-conda`


(For testing `vllm` on a machine with 8, 25Gb GPUs:

```
vllm serve \
meta-llama/Llama-2-70b-chat-hf  \
--dtype auto \
--trust-remote-code \
--tensor-parallel-size 8 \
--gpu-memory-utilization .8
```
)

### For Windows - Install:

- `python>=3.10`: Download and install from python.org (check "Add Python to PATH")

- [`jupyter`](https://jupyter.org/install).

Run `setup.bat` (run as administrator if you encounter permission issues)

Run `\env-mindgames\Scripts\activate`

### Environment variables

Store the following as environment variables or pass them in as arguments.

- [HF_HOME](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables#hfhome)
- [HF_TOKEN](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables#hftoken)
- [OPENAI_API_KEY](https://platform.openai.com/api-keys)
- [TOGETHER_API_KEY](https://docs.together.ai/docs/quickstart)
- [ANTHROPIC_API_KEY](https://docs.anthropic.com/en/api/getting-started)

E.g.:
 - macOS/Linux: `echo export HF_HOME="<path>"' >> ~/.zshrc`
 - Windows: System > "Advanced system settings" > "Environment variables" > Under "User variables" click "New" and add each variable

Or pass as arguments: `OPENAI_API_KEY=<key> make test`

### Web App Setup

To set up the frontend to run locally, you will need to have Node.js and npm installed.

1. **Install Node.js and npm**:
   - macOS/Linux: `brew install node`
   - Windows: [Download Node.js](https://nodejs.org) (this will also install npm).

2. (*Install and build the frontend*:)
   Only necessary if `make build` fails. After cloning the repository, navigate to the `frontend` directory and run:

   ```
   npm install
   npm run build
   ```
### Merging

If you have a conflict in a `*.ipynb` file run `nbdime mergetool`

### Example Game

### Generating the payoff matrices

We have already generated the relevant matricies. This step is only necessary if you would like to regenerate them.

```
$ make_games --save-games --max-solutions 10000 --n-games-to-save 100
$ make_games --save-games --non-solutions --difficulty can-win --max-solutions 10000 --n-games-to-save 100
$ make_games --save-games --non-solutions --difficulty always-win --max-solutions 10000 --n-games-to-save 100
$ make_games --save-games --non-solutions --difficulty never-win --max-solutions 10000 --n-games-to-save 100
```

### Running an Example Game (rational target only)

`$ play_rational_target`

### Running the Human-{Human, Rational Target, LLM} Experiments

We use [FastAPI](https://fastapi.tiangolo.com/):

`$ fastapi run main.py`

This will start a local server, usually accessible at `http://0.0.0.0:8000/` (or `localhost` if you run `dev`).

(On our personal server run `make release`.)

NB: To use the development only features (such as typing in  `'/decide'` as a message to force a round to conclude), pass `DEV_ENVIRONMENT=True fastapi {run, dev} main.py`.

Save the data from the database in `results/` with this command:

`$ read_database [--database <path_to_database>] save-rounds`

NB: Run a command like this to review sent messages before pushing to version control (for IRB). (Further validate by running `python scripts/filter_pii.py`.)

- `$ read_database --log debug --database database.db save-rounds > temp.txt`

Get bonus information from the database in `results/` with this command:

`$ read_database [--database <path_to_database>] get-bonuses` 

(Then open the file `bonuses.csv` copy it and upload it to the "Bulk bonus payment" dialogue on Prolific.)

To extract just the surveys run `read_database --database <databse> save-surveys`. These will be saved to a file `completed-surveys_<date>.csv` inside of the condition directory.

#### Example from figure

To find the model id of the example from the figure run the following.

`sqlite3 database.db`

```sql
select id from model where data == '{"utilities": {"A": {"x": -1, "y": -1, "z": 0}, "B": {"x": -1, "y": -1, "z": 1}, "C": {"x": 0, "y": 1, "z": 1}}, "hidden": {"A": {"x": false, "y": true, "z": false}, "B": {"x": false, "y": true, "z": true}, "C": {"x": false, "y": true, "z": false}}, "ideal_revealed": {"A": {"x": false, "y": true, "z": false}, "B": {"x": false, "y": false, "z": false}, "C": {"x": false, "y": true, "z": false}}, "target_coefficients": {"x": 0, "y": -1, "z": 1}, "persuader_coefficients": {"x": 0, "y": 0, "z": -1}, "proposals": ["A", "B", "C"], "attributes": ["x", "y", "z"], "max_hidden_utilities": 4}';
```


### Running the LLM-LLM Experiments

This command runs the configuration file in `config/llmllm.yaml`, creating a series of games and collecting the responses of various LLMs.

`$ llmllm --config main_experiment.yaml`

(Also run `$ random_baseline` for an emprical measure of what an agent would score if they randomly disclose `n` pieces of information over the game.)

(Run `$ llmllm --config config/validation.yaml` for various other optional validation experiments. These run on a reduced sample.)

Killing the job with `CTRL-C`. The script will output where it has saved the intermediary results. You may re-run the script passing in the intermediary result as an argument `—-previous-condition <FILENAME>`. Then you just keep running that script if the run fails.

Adjust the parameters at the top of your config file to run fewer trials just as a test. 

e.g.:
```
num_unique_payoffs_per_round_condition: 1 # was `20`
num_unique_scenarios_per_payoff: 1
```

## Analysis

In the `analysis` directory.

Run commands like the following to compile Rmarkdown notebooks

`Rscript -e "rmarkdown::render('e1_analysis.Rmd')"`

Or run `jupyter notebook` for the iPython notebooks.

Run the following to estimate and plot the random baseline probability:

`python scripts/random_choice_baseline.py`

## Tests

For basic tests, run:

- macOS/Linux: `make test`
- Windows: # TODO

For basic tests which include querying LLMs (and require you to have the API keys), run:

- macOS/Linux: `make test-query`
- Windows: # TODO

For basic tests which include batch LLM calls (only OpenAI and VLLM) (This takes a while and requires you to have the API keys), run:

- macOS/Linux: `make test-batch`
- Windows: # TODO

For all tests, including the long-running csp and LLM queries, run:

- macOS/Linux: `make test-all`
- Windows: # TODO

## Contributing

Aim to [`black`]( https://black.readthedocs.io) your code (e.g. `black src`).

Also use `pylint` (e.g. `pylint src` or just `darker --lint pylint src` which only applies pylint to the changed files, although it takes a while to run).

For the frontend run `make jslint` before committing. 

For large changes submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

## Repository Structure

- `Makefile`
    - Defines various project level shell script utilities.
- `README.md`
- *`env-mindgames`*
    - Built by `make init`. Your local python virtual environment.
- `environment.yml`
    - Package installs and machine configuration for use with `conda`.
- `requirements.txt`
    - Package installs for use with `pip`.
- `.pylintrc`
    - Defines flags to turn off or on for default pylint code checking.
- `setup.py`
- `scripts`
    - `download_scenarios_survey.py`
- `.server_settings`
    - A configuration file to define settings for the FastAPI server in `src/api.py`
- `config`
    - Arguments for various pacakge executables.
    - `llmllm.yaml`
- `src`
    - `data`
        - `payoffs`
            - A variety of files of payoff matricies generated by `src/mindgames/make_games`.
        - `scenarios.jsonl`
            - The cover stories (scenarios) for use in the games.
            - Do not edit manually. Instead run, `scripts/download_scenarios_survey.py` to update from the web.
        - `survey.jsonl`
            - The survey questions to use in the games.
            - Do not edit manually. Instead run, `scripts/download_scenarios_survey.py` to update from the web.            
    - `mindgames`
        - `classify_messages.py`
        - `game.py`
        - `known_models.py`
        - `make_games.py`
        - `model.py`
        - `play_rational_target.py`        
        - `query_models.py`
        - `run_game.py`
        - `utils.py`
    - `api`
        - `api.py`
        - `message_processing.py`
        - `sql_model.py`
        - `sql_queries.py`
        - `utils.py`
    - `experiments`
        - `llmllm.py`
        - `utils.py`
    - `tests`
        - `mindgames`
            - ...
        - `api`
            - ...
        - `experiments`
            - ...
        - `.pytest.ini`
