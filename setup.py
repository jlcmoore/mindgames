import setuptools

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="mindgames",
    version="0.0.1",
    author="Jared Moore",
    author_email="jared@jaredmoore.org",
    description="MindGames",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={
        "data": ["*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "make_games = mindgames.make_games:main",
            "play_rational_target = mindgames.play_rational_target:main",
            "llmllm = experiments.llmllm:main",
            "read_database = api.read_database:read_database",
            "random_baseline = experiments.random_baseline:main",
        ]
    },
)
