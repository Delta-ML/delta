#!/bin/bash

valgrind --leak-check=full --log-file=valgrind.txt --show-leak-kinds=all  ./tflite.out
