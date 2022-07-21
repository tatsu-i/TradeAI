#!/bin/bash
cd /plugin/kibana-extra/${PLUGIN_NAME}
yarn start --oss --dev  --no-base-path --config /kibana.yml
