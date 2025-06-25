@echo off

echo Creating virtual environment...
python -m venv env-mindgames

echo Installing requirements...
.\env-mindgames\Scripts\pip install -r requirements.txt

echo Installing ipykernel...
.\env-mindgames\Scripts\python -m ipykernel install --user --name "env-mindgames"

echo Installing the project in editable mode...
.\env-mindgames\Scripts\pip install --editable .

echo Setup completed successfully!
echo Run '.\env-mindgames\Scripts\activate' to activate the virtual environment.

pause