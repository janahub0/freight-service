# the agent from block 9, now as a module so the API, the tests and the MCP server can all import it.
# nothing here calls the internet when it is imported, the index and the model are made on first use.

import os, time, ast, operator, sqlite3, logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import rag_core

load_dotenv()
log = logging.getLogger("service")

# settings come from environment variables, nothing secret is in the code
DATA_DIR = os.getenv("DATA_DIR", ".")
MODEL = os.getenv("GEN_MODEL", "gemini-3.5-flash-lite")
os.makedirs(DATA_DIR, exist_ok=True)
rag_core.CACHE_PATH = os.path.join(DATA_DIR, "embedding_cache.json")

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg}


def safe_eval(expr):
    # only plain arithmetic, no eval() on text from the user
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError("only basic arithmetic is allowed")
    return ev(ast.parse(expr, mode="eval").body)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(40 - 10) / 40 * 100'. Use this for any calculation."""
    try:
        return str(round(safe_eval(expression), 4))
    except Exception as e:
        return "error: " + str(e)


_index = None


def get_index():
    global _index
    if _index is None:
        from rag_pipeline import build_index
        _index = build_index()
    return _index


# drift alert: if the best passage is not similar enough to the question, the notes may not cover it
# (or the embeddings changed). the threshold is a starting guess, tune it from the logs.
DRIFT_MIN_SCORE = float(os.getenv("DRIFT_MIN_SCORE", "0.55"))


def search(query, k=3):
    from rag_pipeline import retrieve
    passages = retrieve(get_index(), query, k)
    top = passages[0]["score"] if passages else 0.0
    log.info("retrieval top_score=%.3f query=%s", top, query[:80])
    if top < DRIFT_MIN_SCORE:
        log.warning("RETRIEVAL ALERT: top score %.3f is below %.2f for query: %s", top, DRIFT_MIN_SCORE, query[:80])
    return passages


@tool
def search_documents(query: str) -> str:
    """Search the user's notes on urban flooding and ESG reporting. Returns the most relevant passages with their section names."""
    return "\n\n".join("[" + p["section"] + "] " + p["text"] for p in search(query, 3))


SYSTEM_AGENT = ("You answer questions about the user's notes on urban flooding and ESG reporting. "
                "For anything about the notes call search_documents first and answer only from what it returns, mentioning the section names. "
                "For arithmetic use the calculator. If the notes do not contain the answer say you could not find it in the notes. "
                "Questions about the conversation itself can be answered from the earlier messages. Keep answers short.")

tools = [search_documents, calculator]
_last_call = [0.0]


def text_of(msg):
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def call_llm(model, messages, min_gap=6.0):
    # the free tier is slow, so wait between calls and retry when it says busy
    for attempt in range(5):
        wait = min_gap - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            return model.invoke(messages)
        except Exception as e:
            msg = str(e)
            busy = any(x in msg for x in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")) or "timed out" in msg.lower()
            if busy and attempt < 4:
                time.sleep(15 * (attempt + 1))
                continue
            raise


def build_agent(db_path=None):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=os.getenv("GEMINI_API_KEY"), timeout=60, max_retries=2)
    model = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        return {"messages": [call_llm(model, [SystemMessage(content=SYSTEM_AGENT)] + state["messages"])]}

    conn = sqlite3.connect(db_path or os.path.join(DATA_DIR, "agent_memory.sqlite"), check_same_thread=False)
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=SqliteSaver(conn))


def run_agent(graph, question, thread, callbacks=None):
    """Run one question. Returns the answer and the names of the tools the agent called."""
    config = {"configurable": {"thread_id": thread}, "callbacks": callbacks or []}
    used, answer = [], ""
    for update in graph.stream({"messages": [HumanMessage(content=question)]}, config, stream_mode="updates"):
        for node, data in update.items():
            for m in data["messages"]:
                if node == "agent":
                    if m.tool_calls:
                        used += [tc["name"] for tc in m.tool_calls]
                    else:
                        answer = text_of(m)
    return answer, used