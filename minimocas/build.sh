#! /usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* && "$1" == "gcc" ]]; then
  export CC="/usr/local/bin/gcc-8"
  export CXX="/usr/local/bin/g++-8"
fi

if [[ "$OSTYPE" == "linux"* ]]; then
  if [[ $(hostname) == *"osmino"* ]]; then
    export BOOST_ARGS="-DBOOST_ROOT:PATHNAME=/disk01/boost/boost_1.68.0_gcc_8.1/ -DBoost_NO_BOOST_CMAKE=TRUE"
    export FLTK_ARGS="-DFLTK_INCLUDE_DIR:PATH=/disk01/fltk/fltk-1.3.4_cmake/include -DFLTK_ROOT:PATHNAME=/disk01/fltk/fltk-1.3.4_cmake/"
  else
    export BOOST_ARGS="-DBOOST_ROOT:PATHNAME=$HOME/boost -DBoost_NO_BOOST_CMAKE=TRUE"
  fi
fi

if [[ "$*" == *"doc"* ]]; then
  export DOC_MODE="-DDOC_MODE:BOOL=TRUE"
else
  export DOC_MODE="-DDOC_MODE:BOOL=FALSE"
fi

mkdir -p build_linux
cd build_linux
cmake .. ${BOOST_ARGS} ${FLTK_ARGS} ${DOC_MODE} -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 12
cd ..

if [[ "$*" == *"docpdf"* ]]; then
  cd doc/build/latex
  make
  [ -f refman.pdf ] && cp refman.pdf ../../minimocas.pdf
  cd -
fi
