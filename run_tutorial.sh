#!/usr/bin/env bash
# Open the tutorial, from a fresh clone, with one command:
#   ./run_tutorial.sh
# Makes .venv if it is missing, installs what the notebook needs, opens JupyterLab.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "Creating .venv ..."
    python3 -m venv .venv
fi

if ! .venv/bin/python -c "import jupyterlab, landlab_grid_validation" 2>/dev/null; then
    echo "Installing the package and the notebook tools (a few minutes, once) ..."
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e ".[io,plot,notebook]"
fi

echo
echo "JupyterLab is starting. Open the address below in your browser."
echo "Press Ctrl-C here when you are done."
echo
exec .venv/bin/jupyter lab examples/tutorial.ipynb
