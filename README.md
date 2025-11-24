# TensorFlow TEST of FKI MSU 2025

🚀 This is a uv python monorepo project, so make sure u know how to run a uv project.

🤓 If u don't know how to run it, it doesn't matter, just follow my steps and u will make it.

## Build the project 🚧
📦 As the first step, let's install the required packages.
```bash
uv sync --all-packages
```

## Run the script 🦾
Choosing a project script, then runing the following command.

For example, run the project **car-detector's** script **main.py**.
```bash
uv run --package car-detector src/car-detector/main.py
```

## Create a new project 🏗️
If u want to create a new project based the workspace. Just runing the following command.
```bash
uv init src/example
```
This command will help u create a standard uv project `example` in the directory `src/`. Of course, the uv scarford will add your project setting infomation into the file `pyproject.toml`.