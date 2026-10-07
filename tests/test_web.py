import json
import threading
import urllib.error
import urllib.request

import pytest

from gatekeep.web import make_server


def fake_ask(question, index, mode):
    if question == "boom":
        raise RuntimeError("no network")
    yield {"type": "step", "node": "route", "text": f"route -> retrieve ({mode}, {index})"}
    yield {"type": "done", "seconds": 0.1}


CORPORA = {"book": {"label": "Book", "index": "book-index", "prepare": "", "examples": ["q1"]},
           "sklearn": {"label": "scikit-learn docs", "index": None, "prepare": "python scripts/fetch_sklearn_docs.py",
                       "examples": []}}


@pytest.fixture
def url():
    srv = make_server(CORPORA, fake_ask, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url, path):
    return urllib.request.urlopen(url + path, timeout=5)


def post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url + "/api/ask", data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)


def status(fn, *a):
    try:
        return fn(*a).status
    except urllib.error.HTTPError as e:
        return e.code


def test_serves_the_page_and_its_files(url):
    page = get(url, "/")
    assert page.headers["Content-Type"].startswith("text/html") and b"GateKeep" in page.read()
    assert get(url, "/static/app.js").headers["Content-Type"].startswith("text/javascript")
    assert status(get, url, "/static/../web.py") == 404 and status(get, url, "/static/nope.css") == 404


def test_info_lists_corpora_and_which_are_ready(url):
    info = json.load(get(url, "/api/info"))
    assert [(c["name"], c["ready"]) for c in info["corpora"]] == [("book", True), ("sklearn", False)]
    assert info["corpora"][1]["prepare"].startswith("python scripts/fetch")


def test_ask_streams_json_lines_ending_in_done(url):
    res = post(url, {"question": "what is bagging?", "mode": "fast", "corpus": "book"})
    assert res.headers["Content-Type"] == "application/x-ndjson"
    events = [json.loads(line) for line in res.read().decode().splitlines()]
    assert events[0]["text"] == "route -> retrieve (fast, book-index)" and events[-1]["type"] == "done"


@pytest.mark.parametrize("body", [{"question": "  ", "mode": "fast", "corpus": "book"},
                                  {"question": "q", "mode": "turbo", "corpus": "book"},
                                  {"question": "q", "mode": "fast", "corpus": "sklearn"},
                                  {"question": "q", "mode": "fast", "corpus": "nope"},
                                  {"mode": "fast"}, b"not json"])
def test_bad_requests_get_400(url, body):
    assert status(post, url, body) == 400


def test_a_failing_backend_becomes_an_error_event(url):
    events = [json.loads(line) for line in post(url, {"question": "boom", "mode": "careful", "corpus": "book"}).read().decode().splitlines()]
    assert events == [{"type": "error", "message": "RuntimeError: no network"}]
