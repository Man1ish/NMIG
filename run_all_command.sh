#!/bin/bash

echo "Stopping all running whisk containers ..."
# docker ps --format "{{.ID}} {{.Names}}" | grep 'wsk0.*guest' | awk '{print $1}' | xargs -r docker stop
echo "Finished stopping all running containers."

echo "Building docker images ..."
# bash dockerbuild.sh
bash dockerbuild.sh -p -v v1.2
echo "Building images complete."

echo "Creating actions ..."
# bash run_script.sh update --version=v1.2
# bash run_script.sh update
echo "Actions created."

# echo "Invoking actions ..."
# python app.py
# echo "Actions invoked."
# echo "All tasks completed."
