#!/bin/sh
if command -v glide >/dev/null 2>&1; then
  glide install
else
  echo 'no exists glide'
  echo 'install glide....'
  curl https://glide.sh/get | sh
fi
