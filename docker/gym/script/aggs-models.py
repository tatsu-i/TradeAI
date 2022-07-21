import csv
import json
import click
import sys,os
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search


@click.command()
@click.option('--host', '-h', default='elasticsearch')
@click.option('--port', '-p', default=9200, type=int)
@click.option('--rename_group', '-r', default='{}')
@click.option('--log_dir', '-l', default='.')
@click.option('--days', '-d', default=7, type=int)
@click.option('--exclude_group', '-e', multiple=True)
@click.option('--require_profit', '-n', default=100000, type=int)
def main(host, port, rename_group, log_dir, days, exclude_group, require_profit):
    rename_group = json.loads(rename_group)
    client = Elasticsearch(host=host, port=port)
    s = Search(using=client, index="backtest*").query("range", **{"@timestamp": {"gte": datetime.today() - timedelta(days=days), "lte": datetime.today()}})
    s.aggs.bucket('model_name', 'terms', field='model_name.keyword', size=100000) \
        .bucket('expname', 'terms', field='expname.keyword', size=10000) \
        .metric('max_netprofit', 'max', field='netprofit')

    response = s.execute()

    model_table = {}

    for model in response.aggregations.model_name.buckets:
        for exp in model.expname.buckets:
            if exp.key in exclude_group:
                continue
            model_table[model.key] = model_table.get(
                model.key,
                {"expname": exp.key, "netprofit": exp.max_netprofit.value}
            )
            if model_table[model.key]["netprofit"] < exp.max_netprofit.value:
                model_table[model.key] = {"expname": exp.key, "netprofit": exp.max_netprofit.value}


    model_table = dict(sorted(model_table.items(), key=lambda x:x[1].get('netprofit', 0), reverse=True))
    pf = open(os.path.join(log_dir, 'poor.csv'), 'w')
    rf = open(os.path.join(log_dir, 'rich.csv'), 'w')
    af = open(os.path.join(log_dir, 'all.csv'), 'w')
    for model_name in model_table.keys():
        expname = model_table[model_name]["expname"]
        expname = rename_group.get(expname, expname)
        netprofit = model_table[model_name]["netprofit"]
        profit = '{:,}'.format(int(netprofit))
        print(f"{model_name} {expname} {profit}")
        if netprofit < require_profit:
            writer = csv.writer(pf)
            writer.writerow([model_name, expname, netprofit])
        if netprofit >= require_profit:
            writer = csv.writer(rf)
            writer.writerow([model_name, expname, netprofit])
        writer = csv.writer(af)
        writer.writerow([model_name, expname, netprofit])
    pf.close()
    rf.close()

if __name__ == '__main__':
    main()


