#!/bin/bash
set -e
apt-get update && apt-get install -y libtbb-dev libtbb12 ispc unzip file
# fetch weights using git LFS
git submodule update --init --recursive weights
cd weights && git lfs pull && cd ..
