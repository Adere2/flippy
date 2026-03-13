import json
import threading
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from agents.knowledge_agent import agent_executor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler


class StreamlitToolCallbackHandler(BaseCallbackHandler):
    """Renders live tool-call status widgets inside the active Streamlit context.

    Keyed by run_id so concurrent tool calls don't clobber each other.
    """

    def __init__(self):
        # Capture the session context NOW (main thread) so worker threads can
        # re-attach it before touching any Streamlit state.
        self._ctx = get_script_run_ctx()
        # Dict[run_id -> {"status": ..., "name": ...}] to support parallel calls.
        self._runs: dict = {}

    def _attach_ctx(self):
        if self._ctx:
            add_script_run_ctx(threading.current_thread(), self._ctx)

    @staticmethod
    def _format_input(input_str) -> str:
        """Return a pretty-printed JSON string regardless of input type."""
        if isinstance(input_str, dict):
            return json.dumps(input_str, indent=2)
        try:
            return json.dumps(json.loads(input_str), indent=2)
        except Exception:
            return str(input_str)

    def on_tool_start(self, serialized, input_str, run_id=None, **kwargs):
        self._attach_ctx()
        name = serialized.get("name", "unknown_tool")
        status = st.status(f"🔧 `{name}`", expanded=True, state="running")
        status.code(self._format_input(input_str), language="json")
        self._runs[run_id] = {"status": status, "name": name}

    def on_tool_end(self, output, run_id=None, **kwargs):
        self._attach_ctx()
        run = self._runs.pop(run_id, None)
        if not run:
            return
        clean = str(output)
        if len(clean) > 1200:
            clean = clean[:1200] + "\n… [output truncated]"
        run["status"].code(clean)
        run["status"].update(label=f"✅ `{run['name']}`", state="complete", expanded=False)

    def on_tool_error(self, error, run_id=None, **kwargs):
        self._attach_ctx()
        run = self._runs.pop(run_id, None)
        if not run:
            return
        run["status"].error(str(error))
        run["status"].update(label=f"❌ `{run['name']}` failed", state="error", expanded=True)


st.set_page_config(page_title="Flippy Assistant", page_icon="🤖")
st.title("🤖 Flippy: Fuzzball HPC Assistant")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# React to user input
if prompt := st.chat_input("Ask me about Fuzzball or to run a command..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            lc_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                else:
                    lc_messages.append(AIMessage(content=msg["content"]))

            handler = StreamlitToolCallbackHandler()
            response = agent_executor.invoke(
                {"messages": lc_messages},
                config={"callbacks": [handler]},
            )
            raw_content = response["messages"][-1].content
            if isinstance(raw_content, list) and raw_content and "text" in raw_content[0]:
                output_text = raw_content[0]["text"]
            else:
                output_text = str(raw_content)

            st.markdown(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})
        except Exception as e:
            st.error(f"Error running agent: {e}")