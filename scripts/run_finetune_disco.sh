#!/bin/bash
source "$(dirname "$0")/env_setup.sh"

# Pass the data directory path as $1
python -m physics_jepa.disco --data_path $1
