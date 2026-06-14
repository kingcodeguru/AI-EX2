#!/bin/bash
ver=$1
cd "$(dirname "$0")" || exit 1
cd ../

cp original/*.py bin/
cp my/versions/${ver} bin/ex2.py
cp simulations/ex2_gui.py bin/



# run

cd bin
python3 ex2_gui.py