#!/bin/bash
ver=$1
cd "$(dirname "$0")" || exit 1
cd ../../

cp tests/amit/avg_test.py bin/
# run

cd bin
python3 avg_test.py
python3 avg_test.py > ../tests/amit/$(basename $ver).txt
