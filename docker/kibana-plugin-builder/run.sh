#!/bin/bash
cd /plugin/kibana-extra/${PLUGIN_NAME}
yarn start --dev  --no-base-path --config /kibana.yml
