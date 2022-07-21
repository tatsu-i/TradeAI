#!/bin/bash
cd /plugin/kibana-extra/${PLUGIN_NAME}
yarn kbn bootstrap
yarn build
