import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi.testclient import TestClient
import agent, app as app_module

client = TestClient(app_module.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_calculator_works():
    assert agent.calculator.invoke({"expression": "(5000) / (1200 + 800 + 5000) * 100"}) == "71.4286"


def test_calculator_refuses_code():
    out = agent.calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert out.startswith("error")


def test_ask_uses_the_agent(monkeypatch):
    monkeypatch.setattr(app_module, "get_graph", lambda: object())
    monkeypatch.setattr(agent, "run_agent", lambda g, q, t, c=None: ("fake answer", ["calculator"]))
    r = client.post("/ask", json={"question": "what is 2+2?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "fake answer"
    assert r.json()["tools_used"] == ["calculator"]


def test_ask_rejects_empty_question():
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 400
