#!/bin/sh
kibana_url=${1}
dashboard_uuid=${2}
dashboard_name=${3}
curl -XGET ${kibana_url}/api/kibana/dashboards/export?dashboard=${dashboard_uuid} -o ${dashboard_name}.json
