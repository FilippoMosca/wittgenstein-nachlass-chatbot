from __future__ import annotations

import os
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, Set

import streamlit as st
from azure.storage.blob import BlobServiceClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

st.set_page_config(
    page_title="Wittgenstein Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PASSWORD PROTECTION ---
def check_password():
    def password_entered():
        if st.session_state.get("password", "") == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("Wittgenstein Chatbot")
    st.write("Access restricted. Please enter the password.")

    st.text_input(
        "Password",
        type="password",
        on_change=password_entered,
        key="password",
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password")

    return False


if not check_password():
    st.stop()

# Copy Streamlit secrets into environment variables
# so that HistoryBot and LangChain/Azure clients can read them via os.getenv()
SECRET_KEYS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_VERSION",
    "MODEL_AZURE_DEPLOYMENT",
    "MODEL_AZURE_CODE_DEPLOYMENT",
    "MODEL_AZURE_CODE_DEPLOYMENT_NAME",
    "EMBED_AZURE_DEPLOYMENT",
    "AZURE_AI_SEARCH_SERVICE_NAME",
    "AZURE_AI_SEARCH_INDEX_NAME",
    "AZURE_AI_SEARCH_API_KEY",
]

try:
    secrets_dict = dict(st.secrets)
except Exception:
    secrets_dict = {}

for key in SECRET_KEYS:
    if key in secrets_dict:
        os.environ[key] = str(secrets_dict[key])

required_env = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_VERSION",
    "MODEL_AZURE_DEPLOYMENT",
    "MODEL_AZURE_CODE_DEPLOYMENT",
    "MODEL_AZURE_CODE_DEPLOYMENT_NAME",
    "EMBED_AZURE_DEPLOYMENT",
    "AZURE_AI_SEARCH_SERVICE_NAME",
    "AZURE_AI_SEARCH_INDEX_NAME",
    "AZURE_AI_SEARCH_API_KEY",
]

missing = [k for k in required_env if not os.getenv(k)]
if missing:
    st.error("Missing Streamlit secrets: " + ", ".join(missing))
    st.stop()

from witt_histochat_jupyter import HistoryBot

BASE_DIR = Path(__file__).resolve().parent
WITTGENSTEIN_IMG_PATH = BASE_DIR / "other" / "wittgenstein_pic.png"

DEFAULT_TEMPLATE = (
    " You are an assistant for question-answering tasks, primarily for answering questions "
    "about Ludwig Wittgenstein. The context will primarily be in German, with some parts in English. "
    "Use only the provided retrieved context to answer the question. Keep the answer accurate and "
    "well-explained. Only respond with 'I don't know', if the provided retrieved context is completely irrelevant to the question. "
)

ACCENT_RED = "#ff4b4b"


@st.cache_resource
def get_private_json_path() -> str:
    connection_string = st.secrets["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = st.secrets["AZURE_STORAGE_CONTAINER"]
    blob_name = st.secrets["AZURE_STORAGE_BLOB"]

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    temp_dir = Path(tempfile.gettempdir())
    local_path = temp_dir / blob_name

    if not local_path.exists():
        with open(local_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())

    return str(local_path)


try:
    DEFAULT_JSON_PATH = get_private_json_path()
except Exception as e:
    st.error(f"Failed to download metadata JSON from Azure Blob: {e}")
    st.stop()

if not os.path.exists(DEFAULT_JSON_PATH):
    st.error(f"Metadata JSON not found: {DEFAULT_JSON_PATH}")
    st.stop()


@st.cache_resource
def get_bot(
    json_searchindex_file_path: str,
    default_temperature: float,
    default_k_num: int,
    retrieval_min_query_chars: int,
    debug: bool,
) -> HistoryBot:
    return HistoryBot(
        json_searchindex_file_path=json_searchindex_file_path,
        default_temperature=default_temperature,
        default_k_num=default_k_num,
        retrieval_min_query_chars=retrieval_min_query_chars,
        debug=debug,
    )


if "last_out" not in st.session_state:
    st.session_state.last_out = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


def get_used_sigla(sources: Any) -> Set[str]:
    sigla_used: Set[str] = set()

    if not sources or sources == ["No source"]:
        return sigla_used

    if isinstance(sources, dict):
        sigla_used.update(sources.get("exact_siglum", []))
        sigla_used.update(sources.get("partial_siglum", []))
    elif isinstance(sources, list):
        sigla_used.update(sources)

    return {s for s in sigla_used if s}


def render_sources_text(out: Dict[str, Any]) -> None:
    docs = out.get("docs", []) or []
    sources = out.get("sources", [])

    exact_sigla = set()
    rejected_sigla = set()

    if not docs:
        st.warning("No documents were retrieved.")
        return

    if not sources or sources == ["No source"]:
        st.info("No exact sources declared by LLM2.")
        return

    if isinstance(sources, dict):
        exact_sigla.update(sources.get("exact_siglum", []))
        rejected_sigla.update(sources.get("rejected_siglum", []))
    elif isinstance(sources, list):
        exact_sigla.update(sources)

    shown = 0

    for doc in docs:
        sig = doc.metadata.get("siglum", "")
        if sig not in exact_sigla:
            continue

        shown += 1
        datefrom = doc.metadata.get("datefrom", "")
        dateto = doc.metadata.get("dateto", "")
        text = (doc.page_content or "").replace("\n", " ").strip()

        with st.container(border=True):
            st.markdown(f"### `{sig}`")
            st.markdown(f"**Period:** {datefrom}–{dateto}")
            st.markdown(f"> {text}")

    if shown == 0:
        st.info("No exact declared sources were found among the retrieved documents.")

    if rejected_sigla:
        with st.expander("Rejected source identifiers returned by the model"):
            st.code(str(sorted(rejected_sigla)))


def render_debug_panel(out: Dict[str, Any], bot: HistoryBot) -> None:
    with st.expander("Debug info", expanded=True):
        st.write("Index:", os.getenv("AZURE_AI_SEARCH_INDEX_NAME"))
        st.write("Metadata JSON path:", DEFAULT_JSON_PATH)
        st.write("Metadata JSON exists:", os.path.exists(DEFAULT_JSON_PATH))
        st.write("DF rows:", len(getattr(bot, "DF_wittgenstein", [])))
        st.write("Valid refs:", out.get("valid_refs_df"))
        st.write("Invalid refs:", out.get("invalid_refs_df"))
        st.write("Document-level refs:", out.get("document_level_refs"))
        st.write("Date normalization mode:", out.get("date_normalization_mode"))
        st.write("Filter expression:", out.get("filter_expression"))
        st.write("Retrieval query:", out.get("retrieval_query"))
        st.write("Retrieval fallback used:", out.get("retrieval_query_fallback_used"))
        st.write("Retrieved docs:", len(out.get("docs", []) or []))
        st.write("Top 5 retrieved sigla:", (out.get("docs_siglum") or [])[:5])
        st.write("Candidate pool size:", getattr(bot, "last_candidate_pool_size", 0) or 0)
        st.write("Candidate pool preview:", (getattr(bot, "last_candidate_pool_preview", []) or [])[:5])


def image_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


st.markdown(
    f"""
    <style>
        a.anchor-link {{
            display: none !important;
        }}

        .block-container {{
            padding-top: 3.5rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }}

        div.stButton > button {{
            width: 56px;
            height: 56px;
            border-radius: 999px;
            font-size: 1.3rem;
            font-weight: 700;
            padding: 0;
        }}

        .witt-header-wrap {{
            margin-bottom: 1rem;
        }}

        .witt-subtitle {{
            margin-top: -0.1rem;
            opacity: 0.9;
        }}

        .witt-portrait-card {{
            width: 96px;
            height: 96px;
            border-radius: 14px;
            background: {ACCENT_RED};
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            box-sizing: border-box;
            padding: 6px;
            margin-top: 0.65rem;
        }}

        .witt-portrait-card img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 10px;
            display: block;
        }}

        .witt-title-box {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 96px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


header_col1, header_col2 = st.columns([1.0, 9.0], gap="medium")

with header_col1:
    img_b64 = image_to_base64(WITTGENSTEIN_IMG_PATH)
    if img_b64:
        st.markdown(
            f"""
            <div class="witt-portrait-card">
                <img src="data:image/png;base64,{img_b64}" alt="Wittgenstein portrait" />
            </div>
            """,
            unsafe_allow_html=True,
        )

with header_col2:
    st.markdown(
        """
        <div class="witt-title-box">
            <div class="witt-header-wrap">
                <h1 style="margin-bottom: 0.2rem;">Wittgenstein Nachlass Chatbot</h1>
                <div class="witt-subtitle">A RAG-based Q&A interface for exploring Wittgenstein through his philosophical Nachlass.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Configuration")

    k_num = st.slider(
        "Number of retrieved remarks (k)",
        min_value=10,
        max_value=500,
        value=500,
        step=10,
    )

    show_debug = st.checkbox("Show debug panel", value=False)


try:
    bot = get_bot(
        json_searchindex_file_path=DEFAULT_JSON_PATH,
        default_temperature=0.6,
        default_k_num=k_num,
        retrieval_min_query_chars=6,
        debug=False,
    )
except Exception as e:
    st.error(f"Failed to initialize HistoryBot: {e}")
    st.stop()


question = st.text_area(
    "Question",
    placeholder="Ask your query on Wittgenstein Nachlass",
    height=120,
    label_visibility="collapsed",
    key="question_input",
)

ask_clicked = st.button("➜", help="Run query")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running retrieval and answer generation..."):
            try:
                out = bot.ask(
                    question,
                    user_template_text=DEFAULT_TEMPLATE,
                    k=k_num,
                )
                st.session_state.last_out = out
                st.session_state.last_question = question
            except Exception as e:
                st.error(f"Error while running the chatbot: {e}")
                st.stop()


if st.session_state.last_out is not None:
    out = st.session_state.last_out

    st.markdown("## Answer")
    answer = (out.get("answer") or "").strip()
    st.write(answer if answer else "_(empty)_")

    st.markdown("## Sources")
    st.code(str(out.get("sources", [])))

    with st.expander("Sources – text", expanded=True):
        render_sources_text(out)

    if show_debug:
        render_debug_panel(out, bot)
