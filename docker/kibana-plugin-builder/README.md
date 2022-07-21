# Kibana プラグイン開発用

## build initial
```
$ PLUGIN_NAME=example-plugin
$ docker run --rm -e PLUGIN_NAME=${PLUGIN_NAME} \
  -v $(pwd)/target-kibana-plugin:/plugin/kibana-extra/fregata \
  -it dtr.poc.cdi/poc/kibana-dev build.sh
```

## run kibana
```
$ PLUGIN_NAME=example-plugin
$ docker run --rm -e PLUGIN_NAME=${PLUGIN_NAME} \
  -v $(pwd)/target-kibana-plugin:/plugin/kibana-extra/fregata \
  -v $(pwd)/kibana.yml:/kibana.yml \
  -it dtr.poc.cdi/poc/kibana-dev run.sh
```
