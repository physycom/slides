#! /usr/bin/env bash

declare -a cities=(
  bari
  dubrovnik
  ferrara
  sybenik
  venezia
)

mkdir -p output

#curl -s -S --header "Content-Type: application/json" --request POST --data @body.json http://127.0.0.1:8000/sim?city=ferrara
for c in ${cities[@]}; do
  curl -s -S \
    --header "Content-Type: application/json" \
    --request POST \
    --data @body.json \
    http://127.0.0.1:8000/sim?city=${c} \
    | jq '.' > output/curl_response_${c}.json
done
