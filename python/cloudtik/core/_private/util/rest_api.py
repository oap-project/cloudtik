import json
import urllib
import urllib.request
import urllib.error

REST_API_REQUEST_TIMEOUT = 60


def rest_api_open(url_or_req, timeout=None):
    if timeout is None:
        timeout = REST_API_REQUEST_TIMEOUT
    response = urllib.request.urlopen(
        url_or_req, timeout=timeout)
    return response.read()


def rest_api_get(endpoint_url, timeout=None):
    # disable all proxy on 127.0.0.1
    proxy_support = urllib.request.ProxyHandler({"no": "127.0.0.1"})
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)
    return rest_api_open(endpoint_url, timeout=timeout)


def rest_api_post(endpoint_url, data, data_format=None, timeout=None):
    data_in_bytes = data.encode('utf-8')  # needs to be bytes
    req = urllib.request.Request(endpoint_url, data=data_in_bytes)
    if data_format:
        req.add_header('Content-Type', 'application/{}; charset=utf-8'.format(
            data_format))
    return rest_api_open(req, timeout=timeout)


def rest_api_put(endpoint_url, data, data_format=None, timeout=None):
    data_in_bytes = data.encode('utf-8')  # needs to be bytes
    req = urllib.request.Request(endpoint_url, data=data_in_bytes, method="PUT")
    if data_format:
        req.add_header('Content-Type', 'application/{}; charset=utf-8'.format(
            data_format))
    return rest_api_open(req, timeout=timeout)


def rest_api_delete(endpoint_url, timeout=None):
    req = urllib.request.Request(url=endpoint_url, method="DELETE")
    return rest_api_open(req, timeout=timeout)


def rest_api_get_json(endpoint_url, timeout=None):
    response = rest_api_get(endpoint_url, timeout=timeout)
    return json.loads(response)


def rest_api_post_json(endpoint_url, body, timeout=None):
    data = json.dumps(body)
    response = rest_api_post(
        endpoint_url, data, "json", timeout=timeout)
    return json.loads(response)


def rest_api_put_json(endpoint_url, body, timeout=None):
    data = json.dumps(body)
    response = rest_api_put(
        endpoint_url, data, "json", timeout=timeout)
    return json.loads(response)


def rest_api_delete_json(endpoint_url, timeout=None):
    response = rest_api_delete(
        endpoint_url, timeout=timeout)
    return json.loads(response)
