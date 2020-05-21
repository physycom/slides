#! /usr/bin/env bash

declare -a cities=(
  bari
  dubrovnik
  ferrara
  sybenik
  venezia
)

for c in ${cities[@]}; do
  curl -s -S http://127.0.0.1:8000/poly?city=$c | jq '.geojson' >> polyline_${c}.geojson
done
