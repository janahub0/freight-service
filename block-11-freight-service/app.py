# the freight/document assistant as a web service.
# GET /health says the service is alive, POST /ask sends a question to the agent.

import os, time, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import agent as agent_module

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("service")

app = FastAPI(title="Document assistant")
_graph = None


class Question(BaseModel):
    question: str
    thread_id: str = "default"


def get_graph():
    global _graph
    if _graph is None:
        _graph = agent_module.build_agent()
    return _graph


def get_callbacks():
    # langfuse tracing turns on only when its keys are set as environment variables
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    return []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    start = time.time()
    try:
        answer, tools_used = agent_module.run_agent(get_graph(), q.question, q.thread_id, get_callbacks())
    except Exception as e:
        log.exception("agent failed")
        raise HTTPException(status_code=502, detail="the model call failed: " + str(e)[:200])
    seconds = round(time.time() - start, 2)
    log.info("ask thread=%s tools=%s seconds=%s", q.thread_id, tools_used, seconds)
    return {"answer": answer, "tools_used": tools_used, "seconds": seconds}
