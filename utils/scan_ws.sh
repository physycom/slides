#! /usr/bin/env bash

echo '{
  "start_date"  : "2020-04-25 05:00:00",
  "stop_date"   : "2020-04-25 15:00:00",
  "sampling_dt" : 600,
  "last" : {}
}' > body.json

declare -a cities=(
  bari
  dubrovnik
  ferrara
  sybenik
  venezia
)

mkdir -p output

hostport="localhost:9999"
for c in ${cities[@]}; do
  echo "SIM city : ${c}"
  curl -s -S \
    --header "Content-Type: application/json" \
    --request POST \
    --data @body.json \
    http://${hostport}/sim?citytag=${c} \
    | jq '.' > output/sim_${c}.json
done

for c in ${cities[@]}; do
  echo "GEOJSON city : ${c}"
  curl -s -S \
    --request GET \
    http://${hostport}/poly?citytag=${c} \
    | jq '.' > output/geojson_${c}.json
done
