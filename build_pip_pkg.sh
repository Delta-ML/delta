#!/usr/bin/env bash

echo "Uninstall tf_jieba if exist ..."
pip3 uninstall -y delta-didi

echo "Generate whl file ..."
rm -rf build/ delta-didi.egg-info/ dist/
python3 setup.py bdist_wheel
