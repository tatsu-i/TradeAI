#!/bin/bash
kibana_url=${1}
dashboard=${2}

curl -XPOST ${kibana_url}/api/kibana/dashboards/import -H 'kbn-xsrf:true' -H 'Content-type:application/json' -d @${dashboard}
