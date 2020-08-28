#!/bin/bash

rm dockerfile.delta*

bash build.sh ci cpu  dockerfile
bash build.sh delta cpu  dockerfile
bash build.sh delta gpu  dockerfile
bash build.sh deltann cpu  dockerfile 
bash build.sh deltann gpu  dockerfile 
