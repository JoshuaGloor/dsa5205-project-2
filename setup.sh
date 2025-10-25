#!/bin/bash
set -euo pipefail

ENV="p2"

# Check if conda exists in PATH
if ! command -v conda &> /dev/null
then
    echo "Error: conda is not installed or not in PATH."
    exit 1
fi

# Make 'conda activate' work inside scripts
eval "$(conda shell.bash hook)"

# Create or update the conda environment
# [manual] run `conda env create -f environment.yaml` to create the env
if conda env list | grep -qE "^${ENV}\s"; then
    echo "Conda environment '${ENV}' already exists. Will update it..."
    conda env update -n "$ENV" -f environment.yaml --prune
else
    echo "Creating new conda environment '${ENV}'..."
    conda env create -f environment.yaml
fi

# Activate environment
# [manual] run below command with the env name found in the `environment.yaml` file.
conda activate "$ENV"

# Register Jupyter kernel
# [manual] run below command with the env name found in the `environment.yaml` file.
python -m ipykernel install --user --name="$ENV" --display-name "Python ($ENV)"

echo "Setup completed: Environment '$ENV' created and Jupyter kernel registered."

