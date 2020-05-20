#! /usr/bin/env bash

mkdir -p build_linux
cd build_linux
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 8 --target install
cd ..
