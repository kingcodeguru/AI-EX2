#!/bin/bash
ver=$1
cd "$(dirname "$0")" || exit 1
cd ../../


cp original/*.py bin/
cp tests/david/ex2_check.py bin/
cp my/versions/${ver} bin/ex2.py



# run

cd bin
python3 ex2_check.py