import requests

def rest_queue_list(user='guest', password='guest', host='rabbitmq', port=15672, virtual_host=None):
    queues = []
    try:
        url = 'http://%s:%s/api/queues/%s' % (host, port, virtual_host or '')
        response = requests.get(url, auth=(user, password))
        queues = [q['name'] for q in response.json()]
    except:
        pass
    return queues
