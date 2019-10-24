#!/usr/bin/env bash

echo "Uninstall delta-didi if exist ..."
pip3 uninstall -y delta-didi

echo "Build binary distribution wheel file ..."
rm -rf build/ delta-didi.egg-info/ dist/
python3 setup.py bdist_wheel
