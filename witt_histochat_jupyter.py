from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

import os
import re
import uuid
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Azure Search field mapping ---
os.environ["AZURESEARCH_FIELDS_ID"] = "chunk_id"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"

# --- Azure Search SDK imports ---
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# --- LangChain imports ---
from typing_extensions import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_community.vectorstores.azuresearch import AzureSearch
import langchain_community.vectorstores.azuresearch as azuresearch


# -----------------------------
# Helper functions
# -----------------------------

def formatdate_yyyy_mm_dd(date_str: Any) -> str:
    date_str = str(date_str) if not isinstance(date_str, str) else date_str
    return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")


def _normalize_ms_ts_variants(text: str) -> str:
    """Normalize MS-/ms- -> Ms- and TS-/ts- -> Ts-."""
    text = re.sub(r"\bms-", "Ms-", text, flags=re.IGNORECASE)
    text = re.sub(r"\bts-", "Ts-", text, flags=re.IGNORECASE)
    return text


def _unescape_backslashes(s: str) -> str:
    """Reverse re.escape-style escaping used in referenceList/listrefs strings."""
    return re.sub(r"\\(.)", r"\1", s)


def _user_siglum_to_df_siglum(sig: str) -> str:
    """
    Normalize user ref to DF-style (underscore).
    """
    s = str(sig)
    s = re.sub(r",\s+(?=[A-Z0-9\[])", ",", s)
    s = re.sub(r",(?=[A-Za-z0-9])", "_", s)
    return s


def process_sentence_in_pattern(sentence: str) -> Tuple[str, List[str]]:
    """
    Robust extraction of Ms-/Ts- sigla.
    """
    sentence = _normalize_ms_ts_variants(sentence)
    sentence = re.sub(r",\s+(?=[A-Z0-9\[])", ",", sentence)
    sentence = re.sub(r"\s*([?.!;:\"(){}])\s*", r" \1 ", sentence)
    sentence = re.sub(r"\s*(['])\s*", r" \1 ", sentence)

    ref_pattern = re.compile(
        r"\b(?:Ms-|Ts-)\d{3}[A-Za-z0-9_,\[\]]*(?:et[A-Za-z0-9_,\[\]]+)*"
    )

    found_refs = [m.group(0) for m in ref_pattern.finditer(sentence)]

    referenceList: List[str] = []
    for ref in found_refs:
        ref_norm = _user_siglum_to_df_siglum(ref)
        ref_norm = ref_norm.rstrip(",")
        ref_esc = re.escape(ref_norm)

        sentence = sentence.replace(ref, ref_esc)
        referenceList.append(ref_esc)

    return sentence, referenceList


# -----------------------------
# Main Bot class
# -----------------------------

class HistoryBot:
    """Core chatbot logic without Streamlit UI."""

    class AnswerWithSources(TypedDict):
        answer: str
        sources: Annotated[
            List[str],
            ...,
            "List of sources copied exactly from the retrieved context. Return [] if no sources are used.",
        ]

    def __init__(
        self,
        json_searchindex_file_path: str = os.getenv(
            "WITT_DF_PATH",
            "assets-json/DF-wittgenstein_dates_fixed.json"
        ),
        default_temperature: float = 0.0,
        default_k_num: int = 500,
        retrieval_min_query_chars: int = 6,
        default_retrieval_mode: str = "vector",
        use_multilingual_query_expansion: bool = True,
        debug: bool = False,
    ) -> None:

        os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME", "")
        os.environ["AZURE_AI_SEARCH_INDEX_NAME"] = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "")
        os.environ["AZURE_AI_SEARCH_API_KEY"] = os.getenv("AZURE_AI_SEARCH_API_KEY", "")

        self.default_temperature = default_temperature
        self.default_k_num = default_k_num
        self.retrieval_min_query_chars = retrieval_min_query_chars
        self.default_retrieval_mode = (default_retrieval_mode or "vector").strip().lower()
        self.use_multilingual_query_expansion = use_multilingual_query_expansion
        self.debug = debug

        self.default_user_template = (
            " You are an assistant for question-answering tasks, primarily for answering questions about Ludwig Wittgenstein. "
            "The context will primarily be in German, with some parts in English. Use only the provided retrieved context "
            "to answer the question. Keep the answer accurate and well-explained. Only respond with 'I don't know', "
            "if the provided retrieved context is completely irrelevant to the question. "
        )

        self.json_searchindex_file_path = json_searchindex_file_path
        if json_searchindex_file_path and os.path.exists(json_searchindex_file_path):
            self.DF_wittgenstein = self._load_df(json_searchindex_file_path)
        else:
            self.DF_wittgenstein = pd.DataFrame(columns=["siglum", "datefrom", "dateto", "refcontent"])

        self.known_sigla = set(self.DF_wittgenstein["siglum"].astype(str).tolist())
        self.known_siglum_prefixes = set(str(s).split("_", 1)[0] for s in self.known_sigla)

        self.azure_code_llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_deployment=os.getenv("MODEL_AZURE_CODE_DEPLOYMENT"),
            model=os.getenv("MODEL_AZURE_CODE_DEPLOYMENT_NAME"),
            temperature=0,
        )

        self.embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("EMBED_AZURE_DEPLOYMENT"),
            chunk_size=1,
        )

        self.azure_llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_deployment=os.getenv("MODEL_AZURE_DEPLOYMENT"),
            temperature=1.0,
        )

        self.structured_azure_llm = self.azure_llm.with_structured_output(
            self.AnswerWithSources,
            method="function_calling",
        )

        self.json_parser = JsonOutputParser()
        self.filter_prompt = self._build_filter_prompt()
        self.filter_chain = (
            {
                "user_query": lambda x: x["user_query"],
                "referenceList": lambda x: x["referenceList"],
            }
            | self.filter_prompt
            | self.azure_code_llm
            | self.json_parser
        )

        self.prompt = self._build_final_prompt()
        self.chain = (
            RunnablePassthrough()
            | {
                "user_template_text": lambda x: x["user_template_text"],
                "question": lambda x: x["question"],
                "context": lambda x: x["context"],
            }
            | self.prompt
            | self.structured_azure_llm
        )

        vector_store_address: str = f"https://{os.getenv('AZURE_AI_SEARCH_SERVICE_NAME')}.search.windows.net"
        index_name: str = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
        search_key: str = os.getenv("AZURE_AI_SEARCH_API_KEY")

        self.azure_search: AzureSearch = azuresearch.AzureSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=search_key,
            index_name=index_name,
            embedding_function=self.embeddings.embed_query,
            additional_search_client_options={"retry_total": 4},
        )

        self.search_client = SearchClient(
            endpoint=vector_store_address,
            index_name=index_name,
            credential=AzureKeyCredential(search_key),
        )

        self.last_candidate_pool_size = 0
        self.last_candidate_pool_preview: List[str] = []
        self.last_valid_refs: List[str] = []
        self.last_invalid_refs: List[str] = []
        self.last_document_level_refs: List[str] = []
        self.last_date_normalization_mode: str = "none"
        self.last_retrieval_query_plain: str = ""
        self.last_retrieval_query_expanded: str = ""
        self.last_query_expansion_applied: bool = False
        self.last_query_translation_targets: List[str] = []
        self.last_query_translations: Dict[str, str] = {}

    # -----------------------------
    # Prompts
    # -----------------------------

    def _build_filter_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["user_query", "referenceList"],
            template=r"""
You are a helpful assistant who provides structured output for a search system from a user's query.

User Query:
{user_query}

Extract structured output based on the following context:

Focus on two Python lists: 'listrefs', 'listdates'.
If {referenceList} is empty, listrefs = [].

If the user query related to 'listrefs' implies a range, extract all values within that range into a list called 'listrefs'.

If the user query related to 'listrefs' implies discrete values, the discrete values should be items of the list {referenceList}.
Include all discrete values into the list called 'listrefs'.

The values for 'datefrom' and 'dateto' must be in yyyymmdd format.
Ignore 'datefrom' and 'dateto' if no specific date, year, or time period is mentioned in the user query.

Extract 'datefrom' and 'dateto' into the list called 'listdates':
listdates = ['datefrom', 'dateto']

If 'listrefs' is empty, listrefs = [].
If 'listdates' is empty, listdates = [].
Both 'listrefs' and 'listdates' must be valid Python lists.

Must return a JSON object having:
- 'listrefs': listrefs
- 'listdates': listdates
- 'modified_user_query': modified User Query trimming filtered expression without explanation
""",
        )

    def _build_final_prompt(self) -> PromptTemplate:
        prompt_template = """You are a helpful assistant.

{user_template_text}

User question: {question}

The context consists of three fields: 'source', 'period', and 'context'.

Important rules for the 'sources' field:
- Return only source identifiers copied EXACTLY from the 'source' field in the provided context.
- Do not shorten, normalize, paraphrase, simplify, or merge any source identifier.
- Do not omit brackets, page markers, remark markers, or suffixes.
- Do not return document-level identifiers if the context only contains remark-level identifiers.
- If you are not sure of an exact source identifier, do not include it.
- Every item in 'sources' must be an exact string that appears in the provided context.

Context:
{context}
"""
        return PromptTemplate.from_template(prompt_template)

    # -----------------------------
    # Data / filtering
    # -----------------------------

    def _load_df(self, json_path: str) -> pd.DataFrame:
        df = pd.read_json(json_path, lines=True)
        usecols = ["siglum", "datefrom", "dateto", "refcontent"]
        return df[usecols]

    def get_filter_from_df(self, filter_val_dct: Dict[str, Any]) -> Optional[str]:
        self.last_candidate_pool_size = 0
        self.last_candidate_pool_preview = []

        listrefs_esc = filter_val_dct.get("listrefs") or []
        listdates = filter_val_dct.get("listdates") or []

        has_refs = bool(listrefs_esc)
        has_dates = bool(listdates)

        listrefs: List[str] = [_unescape_backslashes(r) for r in listrefs_esc] if has_refs else []

        if has_dates:
            try:
                d_start = int(listdates[0])
                d_end = int(listdates[1])
            except Exception:
                has_dates = False
                d_start = d_end = None
        else:
            d_start = d_end = None

        def _search_in_siglum(values: List[str]) -> str:
            safe = [v.replace("'", "''") for v in values]
            return "search.in(siglum, '{}', '|')".format("|".join(safe))

        if has_dates and not has_refs:
            filtered_df = self.DF_wittgenstein[
                (self.DF_wittgenstein["datefrom"] <= d_end)
                & (self.DF_wittgenstein["dateto"] >= d_start)
            ]
            if not filtered_df.empty:
                sigla = list(filtered_df["siglum"].astype(str))
                self.last_candidate_pool_size = len(sigla)
                self.last_candidate_pool_preview = sigla[:5]

            return f"datefrom le {d_end} and dateto ge {d_start}"

        if has_refs and not has_dates:
            filtered_df = self.DF_wittgenstein[
                self.DF_wittgenstein["siglum"].astype(str).str.startswith(tuple(listrefs))
            ]
            if filtered_df.empty:
                return None

            sigla = list(filtered_df["siglum"].astype(str))
            self.last_candidate_pool_size = len(sigla)
            self.last_candidate_pool_preview = sigla[:5]

            return _search_in_siglum(sigla)

        if has_refs and has_dates:
            filtered_df = self.DF_wittgenstein[
                (self.DF_wittgenstein["siglum"].astype(str).str.startswith(tuple(listrefs)))
                & (self.DF_wittgenstein["datefrom"] <= d_end)
                & (self.DF_wittgenstein["dateto"] >= d_start)
            ]
            if filtered_df.empty:
                return None

            sigla = list(filtered_df["siglum"].astype(str))
            self.last_candidate_pool_size = len(sigla)
            self.last_candidate_pool_preview = sigla[:5]

            return _search_in_siglum(sigla)

        return None

    # -----------------------------
    # Guard-rails helpers
    # -----------------------------

    def _extract_and_validate_refs(self, referenceList_escaped: List[str]) -> Tuple[List[str], List[str]]:
        candidates = [_unescape_backslashes(r) for r in referenceList_escaped]

        valid: List[str] = []
        invalid: List[str] = []

        for c in candidates:
            if c in self.known_sigla:
                valid.append(c)
            elif c in self.known_siglum_prefixes:
                valid.append(c)
            else:
                invalid.append(c)

        return valid, invalid

    @staticmethod
    def _is_yyyymmdd(s: Any) -> bool:
        return bool(re.fullmatch(r"\d{8}", str(s)))

    def _normalize_dates_from_text(self, question_text: str) -> List[str]:
        text = (question_text or "").strip().lower()
        if not text:
            return []

        month_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
            "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
            "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5,
            "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10,
            "novembre": 11, "dicembre": 12,
            "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "mai": 5,
            "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
            "mars": 3, "desember": 12,
        }

        def _last_day_of_month(year: int, month: int) -> int:
            if month == 2:
                leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
                return 29 if leap else 28
            if month in {4, 6, 9, 11}:
                return 30
            return 31

        month_names_sorted = sorted(month_map.keys(), key=len, reverse=True)
        month_alt = "|".join(re.escape(m) for m in month_names_sorted)

        cross_year_connectors = r"(?:to|and|e|a|al|alla|fino\s+a|bis|und|og)"
        cross_year_prefixes = r"(?:from|between|da|fra|tra|von)?\s*"

        cross_year_match = re.search(
            rf"\b{cross_year_prefixes}({month_alt})\s+(19\d{{2}}|20\d{{2}})\s+{cross_year_connectors}\s+({month_alt})\s+(19\d{{2}}|20\d{{2}})\b",
            text,
            flags=re.IGNORECASE,
        )

        if cross_year_match:
            month1_name = cross_year_match.group(1).lower()
            year1 = int(cross_year_match.group(2))
            month2_name = cross_year_match.group(3).lower()
            year2 = int(cross_year_match.group(4))

            m1 = month_map[month1_name]
            m2 = month_map[month2_name]

            if (year2, m2) < (year1, m1):
                year1, year2 = year2, year1
                m1, m2 = m2, m1

            last_day = _last_day_of_month(year2, m2)
            return [f"{year1}{m1:02d}01", f"{year2}{m2:02d}{last_day:02d}"]

        range_connectors = r"(?:to|and|e|a|al|alla|fino\s+a|bis|und|og)"
        optional_prefixes = r"(?:from|between|da|fra|tra|von)?\s*"

        range_match = re.search(
            rf"\b{optional_prefixes}({month_alt})\s+{range_connectors}\s+({month_alt})\s+(19\d{{2}}|20\d{{2}})\b",
            text,
            flags=re.IGNORECASE,
        )

        if range_match:
            month1_name = range_match.group(1).lower()
            month2_name = range_match.group(2).lower()
            year = int(range_match.group(3))

            m1 = month_map[month1_name]
            m2 = month_map[month2_name]

            if m2 < m1:
                m1, m2 = m2, m1

            last_day = _last_day_of_month(year, m2)
            return [f"{year}{m1:02d}01", f"{year}{m2:02d}{last_day:02d}"]

        single_match = re.search(
            rf"\b({month_alt})\s+(19\d{{2}}|20\d{{2}})\b",
            text,
            flags=re.IGNORECASE,
        )

        if single_match:
            month_name = single_match.group(1).lower()
            year = int(single_match.group(2))
            month_num = month_map[month_name]
            last_day = _last_day_of_month(year, month_num)
            return [f"{year}{month_num:02d}01", f"{year}{month_num:02d}{last_day:02d}"]

        years = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
        if not years:
            return []
        if len(years) == 1:
            y = years[0]
            return [f"{y}0101", f"{y}1231"]

        y1, y2 = years[0], years[1]
        if y2 < y1:
            y1, y2 = y2, y1
        return [f"{y1}0101", f"{y2}1231"]

    def _patch_llm1_dates(self, llm1_out: Dict[str, Any], original_question: str) -> Dict[str, Any]:
        out = dict(llm1_out)
        ld = out.get("listdates") or []
        years_in_q = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", original_question)]

        if (
            isinstance(ld, list)
            and len(ld) == 2
            and self._is_yyyymmdd(ld[0])
            and self._is_yyyymmdd(ld[1])
        ):
            start = str(ld[0])
            end = str(ld[1])

            if years_in_q and end.endswith("0101"):
                end = end[:4] + "1231"
                out["listdates"] = [start, end]
                out["listdates_patched"] = [start, end]
                self.last_date_normalization_mode = "kept_llm1_yearend"
                return out

            out["listdates"] = [start, end]
            out["listdates_patched"] = [start, end]
            self.last_date_normalization_mode = "kept_llm1"
            return out

        norm = self._normalize_dates_from_text(original_question)
        if norm:
            out["listdates"] = norm
            out["listdates_patched"] = norm
            self.last_date_normalization_mode = "fallback_dates"
        else:
            out["listdates"] = []
            out["listdates_patched"] = []
            self.last_date_normalization_mode = "empty"

        return out

    @staticmethod
    def _cleanup_semantic_query(text: str) -> str:
        mq = text or ""
        mq = re.sub(r"\(\s*\)", " ", mq)
        mq = re.sub(r"\[\s*\]", " ", mq)
        mq = re.sub(r"\b(in|with)\s+(and|with|in)\b", r"\1", mq, flags=re.IGNORECASE)
        mq = re.sub(r"\b(in|with)\s+and\b", " ", mq, flags=re.IGNORECASE)
        mq = re.sub(r"\b(in|with|and|to|from|between|of)\b\s*([?.!,;:]*)$", r"\2", mq, flags=re.IGNORECASE)
        mq = re.sub(r"\s+", " ", mq).strip()
        mq = re.sub(r"\s+([?.!,;:])", r"\1", mq)
        mq = re.sub(r"^[?.!,;:]+", "", mq).strip()
        mq = re.sub(r"^[)\]]+", "", mq).strip()
        mq = re.sub(r"^[,;:]\s*", "", mq).strip()
        mq = re.sub(r"\s+", " ", mq).strip()
        mq = re.sub(r"[?.!,;:]+$", "", mq).strip()
        return mq

    def _strip_sigla_tokens(self, text: str) -> str:
        t = _unescape_backslashes(text)
        token_pat = re.compile(
            r"\b(?:Ms-|Ts-)\d{3}[A-Za-z0-9_,\[\]]*(?:et[A-Za-z0-9_,\[\]]+)*\b"
        )
        t = token_pat.sub(" ", t)
        return self._cleanup_semantic_query(t)

    def _strip_years(self, text: str) -> str:
        return re.sub(r"\b(19\d{2}|20\d{2})\b", " ", text)

    def _strip_dates_scaffolding(self, text: str) -> str:
        t = text or ""
        t = self._strip_years(t)

        month_words = [
            "january", "jan", "february", "feb", "march", "mar", "april", "apr",
            "may", "june", "jun", "july", "jul", "august", "aug",
            "september", "sep", "sept", "october", "oct", "november", "nov",
            "december", "dec", "gennaio", "febbraio", "marzo", "aprile", "maggio",
            "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
            "januar", "februar", "märz", "maerz", "mai", "juni", "juli", "oktober",
            "dezember", "mars", "desember",
        ]

        month_alt = "|".join(re.escape(m) for m in sorted(set(month_words), key=len, reverse=True))
        t = re.sub(rf"\b({month_alt})\b", " ", t, flags=re.IGNORECASE)
        t = re.sub(
            r"\b(between|from|to|and|in|during|da|a|al|alla|nel|del|della|di|fra|tra|e|im|am|vom|von|bis|und|i|mellom|og)\b",
            " ",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(r"\b(month|mese|monat)\b", " ", t, flags=re.IGNORECASE)
        return self._cleanup_semantic_query(t)

    def _strip_listrefs_from_text(self, text: str, listrefs_escaped: List[str], valid_refs_siglum: List[str]) -> str:
        t = text or ""
        for ref_esc in listrefs_escaped or []:
            t = t.replace(ref_esc, " ")
            t = t.replace(_unescape_backslashes(ref_esc), " ")

        for r in valid_refs_siglum or []:
            variants = {r, r.replace("_", ","), r.replace(",", "_")}
            for v in variants:
                t = t.replace(v, " ")
                t = t.replace(re.escape(v), " ")

        return self._cleanup_semantic_query(t)

    # -----------------------------
    # Retrieval helpers
    # -----------------------------
    def _strip_corpus_author_boilerplate(self, text: str) -> str:
        """
        Remove author/corpus and request boilerplate from the retrieval query before embedding/search.
        This affects ONLY retrieval_query, not the original user question.
        """
        t = text or ""

        author_patterns = [
            r"\bludwig\s+wittgenstein\b",
            r"\bl\.\s*wittgenstein\b",
            r"\bl\.\s*w\.\b",
            r"\blw\b",
            r"\bwittgenstein\b",
            r"\bludwig\b",
        ]

        passage_terms = (
            r"passages?|remarks?|observations?|notes?|entries?|"
            r"passaggi|osservazioni|note|"
            r"stellen|bemerkungen?|notizen|"
            r"passasjer|bemerkninger"
        )

        boilerplate_patterns = [
            # English
            rf"\bwhat\s+does\s+(?:ludwig\s+)?wittgenstein\s+(?:say|write|think)\s+about\b",
            rf"\bwhat\s+does\s+(?:he|wittgenstein)\s+(?:say|write|think)\s+about\b",
            rf"\bhow\s+does\s+(?:ludwig\s+)?wittgenstein\s+(?:describe|characterize|define|present|explain|treat|use|uses)\b",
            rf"\bhow\s+does\s+(?:he|wittgenstein)\s+(?:describe|characterize|define|present|explain|treat|use|uses)\b",
            rf"\bwhat\s+is\s+(?:his|wittgenstein's)\s+(?:view|thought)\s+on\b",
            rf"\bwhat\s+does\s+(?:he|wittgenstein)\s+mean\s+by\b",
            rf"\b(?:give|show|list|find|retrieve)\s+(?:me\s+)?(?:all\s+)?(?:a\s+list\s+of\s+)?(?:the\s+)?{passage_terms}\s+(?:where|in\s+which|that|about|on)\b",
            rf"\b(?:give|show|list|find|retrieve)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?{passage_terms}\s+(?:where|in\s+which|that)?\s*(?:wittgenstein\s+)?(?:uses|mentions|discusses|speaks\s+about|writes\s+about)\b",
            rf"\b(?:all\s+)?{passage_terms}\s+(?:where|in\s+which|that|about|on)\b",
            rf"\bthe\s+(?:theme|topic)\s+of\b",

            # Italian
            rf"\b(?:che\s+cosa|cosa)\s+(?:dice|pensa)\s+(?:wittgenstein|ludwig\s+wittgenstein)?\s*(?:di|del|della|dello|dei|degli|delle|su|sul|sulla|sullo|sui|sugli|sulle)\b",
            rf"\bcome\s+(?:wittgenstein|ludwig\s+wittgenstein|egli|lui)?\s*(?:descrive|caratterizza|definisce|presenta|spiega|tratta|usa|utilizza)\b",
            rf"\bin\s+che\s+modo\s+(?:wittgenstein|ludwig\s+wittgenstein|egli|lui)?\s*(?:descrive|caratterizza|definisce|presenta|spiega|tratta|usa|utilizza)\b",
            rf"\bqual\s+è\s+(?:il\s+(?:suo\s+)?pensiero|la\s+(?:sua\s+)?posizione)\s+(?:su|sul|sulla|sullo|sui|sugli|sulle|riguardo\s+a|intorno\s+a)\b",
            rf"\b(?:dammi|fammi|trovami|cercami|recupera|mostrami|elenca)\s+(?:tutti\s+i\s+|tutte\s+le\s+)?(?:una\s+lista\s+di\s+)?{passage_terms}\s+(?:in\s+cui|che|su)\b",
            rf"\b(?:dammi|fammi|trovami|cercami|recupera|mostrami|elenca)\s+(?:tutti\s+i\s+|tutte\s+le\s+)?(?:{passage_terms}\s+)?(?:in\s+cui|che)?\s*(?:wittgenstein\s+)?(?:usa|utilizza|menziona|discute|parla\s+di|scrive\s+di)\b",
            rf"\b(?:tutti\s+i\s+|tutte\s+le\s+)?{passage_terms}\s+(?:in\s+cui|che|su)\b",
            rf"\bil\s+tema\s+(?:di|del|della)\b",
            rf"\ba\s+proposito\s+di\b",

            # German
            rf"\bwas\s+(?:sagt|schreibt|denkt)\s+(?:wittgenstein|er)\s+(?:über|zu)\b",
            rf"\bwie\s+(?:beschreibt|charakterisiert|definiert|stellt|erklärt|behandelt|verwendet|benutzt)\s+(?:wittgenstein|er)\b",
            rf"\bwie\s+(?:wittgenstein|er)\s+(?:beschreibt|charakterisiert|definiert|darstellt|erklärt|behandelt|verwendet|benutzt)\b",
            rf"\bwie\s+lautet\s+(?:seine|wittgensteins)\s+auffassung\s+(?:über|zu)\b",
            rf"\b(?:gib|zeige|liste|finde|suche)\s+(?:mir\s+)?(?:alle\s+)?{passage_terms}\s+(?:in\s+denen|wo|über|zu)\b",
            rf"\b(?:gib|zeige|liste|finde|suche)\s+(?:mir\s+)?(?:alle\s+)?(?:{passage_terms}\s+)?(?:in\s+denen|wo)?\s*(?:wittgenstein\s+)?(?:verwendet|benutzt|erwähnt|diskutiert|bespricht)\b",
            rf"\b(?:alle\s+)?{passage_terms}\s+(?:in\s+denen|wo|über|zu)\b",
            rf"\bdas\s+thema\s+(?:von|der|des)\b",

            # Norwegian
            rf"\bhva\s+(?:sier|skriver|tenker)\s+(?:wittgenstein|han)\s+om\b",
            rf"\bhvordan\s+(?:beskriver|karakteriserer|definerer|presenterer|forklarer|behandler|bruker)\s+(?:wittgenstein|han)\b",
            rf"\bhvordan\s+(?:wittgenstein|han)\s+(?:beskriver|karakteriserer|definerer|presenterer|forklarer|behandler|bruker)\b",
            rf"\b(?:gi|vis|list|finn|hent)\s+(?:meg\s+)?(?:alle\s+)?{passage_terms}\s+(?:der|som|om)\b",
            rf"\b(?:gi|vis|list|finn|hent)\s+(?:meg\s+)?(?:alle\s+)?(?:{passage_terms}\s+)?(?:der|som)?\s*(?:wittgenstein\s+)?bruker\b",
            rf"\b(?:alle\s+)?{passage_terms}\s+(?:der|som|om)\b",
            rf"\btemaet\s+(?:om|for)\b",
        ]

        generic_question_scaffold_patterns = [
            # English leftovers
            r"\bhow\s+does\s+(?:describe|characterize|define|present|explain|treat|use|uses)\b",
            r"\bwhat\s+does\s+(?:say|write|think)\s+about\b",
            r"\bwhat\s+is\s+(?:view|thought)\s+on\b",
            rf"\b(?:give|show|list|find|retrieve)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:uses|{passage_terms})\b",
            r"\buses\b",

            # Italian leftovers
            r"\bcome\s+(?:descrive|caratterizza|definisce|presenta|spiega|tratta|usa|utilizza)\b",
            r"\bin\s+che\s+modo\s+(?:descrive|caratterizza|definisce|presenta|spiega|tratta|usa|utilizza)\b",
            rf"\b(?:dammi|fammi|trovami|cercami|recupera|mostrami|elenca)\s+(?:tutti\s+i\s+|tutte\s+le\s+)?(?:usi|{passage_terms})\b",

            # German leftovers
            r"\bwie\s+(?:beschreibt|charakterisiert|definiert|stellt|erklärt|behandelt|verwendet|benutzt)\b",
            r"\bwas\s+(?:sagt|schreibt|denkt)\s+(?:über|zu)\b",
            rf"\b(?:gib|zeige|liste|finde|suche)\s+(?:mir\s+)?(?:alle\s+)?(?:verwendungen|{passage_terms})\b",

            # Norwegian leftovers
            r"\bhvordan\s+(?:beskriver|karakteriserer|definerer|presenterer|forklarer|behandler|bruker)\b",
            r"\bhva\s+(?:sier|skriver|tenker)\s+om\b",
            rf"\b(?:gi|vis|list|finn|hent)\s+(?:meg\s+)?(?:alle\s+)?(?:bruk|{passage_terms})\b",
        ]

        for pat in boilerplate_patterns:
            t = re.sub(pat, " ", t, flags=re.IGNORECASE)

        for pat in author_patterns:
            t = re.sub(pat, " ", t, flags=re.IGNORECASE)

        for pat in generic_question_scaffold_patterns:
            t = re.sub(pat, " ", t, flags=re.IGNORECASE)

        # Final cleanup for common leftover request scaffolding.
        leftover_request_patterns = [
            # English
            rf"\b(?:give|show|list|find|retrieve)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:uses|{passage_terms})\b",
            rf"\b(?:all\s+)?{passage_terms}\b",

            # Italian
            rf"\b(?:dammi|fammi|trovami|cercami|recupera|mostrami|elenca)\s+(?:tutti\s+i\s+|tutte\s+le\s+|tutti\s+|tutte\s+)?(?:gli\s+|le\s+)?(?:usi|{passage_terms})\b",
            rf"\b(?:tutti\s+i\s+|tutte\s+le\s+|tutti\s+|tutte\s+)?(?:gli\s+|le\s+)?{passage_terms}\b",
            r"\b(?:usa|utilizza|menziona|discute|parla\s+di|scrive\s+di)\b",

            # German
            rf"\b(?:gib|zeige|liste|finde|suche)\s+(?:mir\s+)?(?:alle\s+)?(?:verwendungen|{passage_terms})\b",
            rf"\b(?:alle\s+)?{passage_terms}\b",
            r"\b(?:verwendet|benutzt|erwähnt|diskutiert|bespricht)\b",

            # Norwegian
            rf"\b(?:gi|vis|list|finn|hent)\s+(?:meg\s+)?(?:alle\s+)?(?:bruk|{passage_terms})\b",
            rf"\b(?:alle\s+)?{passage_terms}\b",
            r"\bbruker\b",
        ]

        for pat in leftover_request_patterns:
            t = re.sub(pat, " ", t, flags=re.IGNORECASE)

        t = re.sub(
            r"^(?:about|on|of|su|sul|sulla|sullo|sui|sugli|sulle|di|del|della|dello|dei|degli|delle|per|riguardo\s+a|intorno\s+a|über|zu|om|for)\s+",
            " ",
            t,
            flags=re.IGNORECASE,
        )

        t = re.sub(
            r"\b(?:il|lo|la|i|gli|le|un|una|uno|di|del|della|dello|dei|degli|delle|tra|fra|in|cui)\b",
            " ",
            t,
            flags=re.IGNORECASE,
        )

        return self._cleanup_semantic_query(t)
    

    def _build_relevant_semantic_query(
        self,
        *,
        original_question: str,
        processed_q: str,
        llm1_out: Dict[str, Any],
        valid_refs_siglum: List[str],
        has_dates: bool,
    ) -> str:
        """
        Deterministically remove:
        - Nachlass refs
        - dates / date scaffolding
        - author/corpus boilerplate
        - request scaffolding

        Return only the semantically relevant content phrase.
        """
        q = processed_q or original_question or ""

        q = self._strip_listrefs_from_text(
            q,
            listrefs_escaped=(llm1_out.get("listrefs") or []),
            valid_refs_siglum=valid_refs_siglum,
        )

        if has_dates:
            q = self._strip_dates_scaffolding(q)

        # Extra safety: dates must not enter semantic retrieval even if LLM1 failed to detect them.
        q = self._strip_years(q)

        q = self._strip_sigla_tokens(q)
        q = self._strip_corpus_author_boilerplate(q)
        q = self._cleanup_semantic_query(q)

        return q

    @staticmethod
    def _dedupe_phrases_preserve_order(parts: List[str]) -> List[str]:
        seen = set()
        out = []

        for p in parts:
            clean = re.sub(r"\s+", " ", (p or "")).strip()
            if not clean:
                continue

            key = clean.lower()
            if key not in seen:
                seen.add(key)
                out.append(clean)

        return out

    def _translate_retrieval_query(
        self,
        plain_query: str,
        *,
        target_languages: List[str],
    ) -> Dict[str, str]:
        """
        Translate the clean retrieval query into compact multilingual equivalents.
        This is ONLY translation, not conceptual rewriting or keyword expansion.
        """
        clean_query = self._cleanup_semantic_query(plain_query)
        targets = [t for t in target_languages if t]

        if not clean_query or not targets:
            return {}

        prompt = f"""
Translate the following retrieval phrase into the requested target languages.

Retrieval phrase:
{clean_query}

Target languages:
{', '.join(targets)}

Rules:
- Translate literally.
- Do NOT interpret.
- Do NOT add related philosophical concepts.
- Do NOT add author names.
- Do NOT add "Wittgenstein".
- Do NOT add dates.
- Do NOT add "grammar", "description", "language", "experience", "logic", "analytic philosophy", unless these words are present in the retrieval phrase.
- If the phrase is a single term, return only the direct equivalent.
- Preserve technical distinctions exactly.
- Preserve proper names, titles, examples, games, and objects.
- Return compact search phrases, not full sentences.
- Return ONLY valid JSON with target language names as keys and translated phrases as values.

Examples:
Input: phenomenology
Output: {{"German": "Phänomenologie"}}

Input: sense and meaning
Output: {{"German": "Sinn und Bedeutung"}}

Input: tennis
Output: {{"German": "Tennis"}}

Input: understanding as mental process
Output: {{"German": "Verstehen als geistiger Vorgang"}}
""".strip()

        try:
            msg = self.azure_code_llm.invoke(prompt)
            content = getattr(msg, "content", msg)

            if isinstance(content, list):
                content = " ".join(str(x) for x in content)

            content = str(content).strip()

            # Remove accidental Markdown fences.
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)

            data = json.loads(content)

            if not isinstance(data, dict):
                return {}

            out: Dict[str, str] = {}

            for lang in targets:
                val = data.get(lang, "")
                val = str(val)

                # Remove dates from translations: dates are handled by metadata filters.
                val = re.sub(r"\b(19\d{2}|20\d{2})\b", " ", val)
                val = re.sub(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b", " ", val)

                val = self._cleanup_semantic_query(val)

                if val:
                    out[lang] = val

            return out

        except Exception as e:
            if self.debug:
                print("[DEBUG][WARN] Query translation failed:", repr(e))
            return {}

    def _build_multilingual_retrieval_query(
        self,
        retrieval_query: str,
        *,
        original_question: str,
    ) -> Tuple[str, bool, List[str], Dict[str, str]]:
        """
        Build a conservative multilingual retrieval query through translation.

        Policy:
        - If original user question is English, append German translation only.
        - If original user question is German, append English translation only.
        - If original user question is Italian or Norwegian, append English and German translations.
        - Answer language is NOT affected by this.
        """
        base = self._cleanup_semantic_query(retrieval_query)

        if not base:
            return base, False, [], {}

        # Remove dates from base as well: dates are already handled by metadata filters.
        base = re.sub(r"\b(19\d{2}|20\d{2})\b", " ", base)
        base = re.sub(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b", " ", base)
        base = self._cleanup_semantic_query(base)

        lang_code = self._detect_user_language(original_question)

        if lang_code == "EN":
            target_languages = ["German"]
        elif lang_code == "DE":
            target_languages = ["English"]
        else:
            target_languages = ["English", "German"]

        translations = self._translate_retrieval_query(
            base,
            target_languages=target_languages,
        )

        parts = [base]

        for t in target_languages:
            val = translations.get(t, "")
            val = re.sub(r"\b(19\d{2}|20\d{2})\b", " ", val)
            val = re.sub(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b", " ", val)
            val = self._cleanup_semantic_query(val)

            if val and val.lower() != base.lower():
                parts.append(val)

        parts = self._dedupe_phrases_preserve_order(parts)

        expanded = self._cleanup_semantic_query(" ".join(parts))
        applied = expanded != base

        return expanded, applied, target_languages, translations

    def _vector_search_sdk(
        self,
        query: str,
        *,
        k: int,
        filters: Optional[str] = None,
    ) -> List[Document]:
        clean_query = (query or "").strip()

        if not clean_query:
            clean_query = "context"

        vector = self.embeddings.embed_query(clean_query)

        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=k,
            fields="text_vector",
        )

        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filters,
            top=k,
        )

        docs: List[Document] = []

        for r in results:
            metadata = {
                "siglum": r.get("siglum", ""),
                "datefrom": r.get("datefrom", ""),
                "dateto": r.get("dateto", ""),
                "chunk_id": r.get("chunk_id", ""),
                "parent_id": r.get("parent_id", ""),
                "score": r.get("@search.score", None),
            }

            docs.append(
                Document(
                    page_content=r.get("chunk", ""),
                    metadata=metadata,
                )
            )

        return docs

    def _retrieve_documents(
        self,
        *,
        retrieval_query: str,
        k: int,
        filter_expression: Optional[str],
        retrieval_mode: Optional[str] = None,
    ) -> List[Document]:
        mode = (retrieval_mode or self.default_retrieval_mode or "vector").strip().lower()

        if mode == "vector":
            return self._vector_search_sdk(
                query=retrieval_query,
                k=k,
                filters=filter_expression,
            )

        if mode == "hybrid":
            return self.azure_search.similarity_search(
                query=retrieval_query,
                k=k,
                filters=filter_expression,
                vector_filter_mode="preFilter",
                search_type="hybrid",
            )

        if mode == "bm25":
            return self.azure_search.similarity_search(
                query=retrieval_query,
                k=k,
                filters=filter_expression,
                search_type="similarity",
            )

        raise ValueError(
            "Unknown retrieval_mode: {}. Use 'vector', 'hybrid', or 'bm25'.".format(retrieval_mode)
        )

    # -----------------------------
    # Ref-only intent routing for LLM2
    # -----------------------------

    @staticmethod
    def _is_ref_only_query(original_question: str, valid_refs_siglum: List[str]) -> bool:
        if not valid_refs_siglum:
            return False

        s = (original_question or "").strip()
        for r in valid_refs_siglum:
            for v in {r, r.replace("_", ","), r.replace(",", "_")}:
                s = s.replace(v, " ")

        s = re.sub(r"[^\w]+", " ", s).strip()
        return len(s) < 6

    def _ref_kind(self, valid_refs_siglum: List[str]) -> str:
        if not valid_refs_siglum:
            return "none"

        all_exact = all(r in self.known_sigla for r in valid_refs_siglum)
        all_doc = all((r in self.known_siglum_prefixes) and (r not in self.known_sigla) for r in valid_refs_siglum)

        if all_exact:
            return "remark"
        if all_doc:
            return "doc"
        return "mixed"

    def _build_llm2_question_for_ref_only(self, *, original_question: str, valid_refs_siglum: List[str]) -> str:
        kind = self._ref_kind(valid_refs_siglum)

        if kind == "remark" and len(valid_refs_siglum) == 1:
            ref = valid_refs_siglum[0]
            return (
                f"The user provided only this Wittgenstein Nachlass remark identifier: {ref}.\n"
                f"Start with ONE short sentence: '{ref} is a remark from Wittgenstein's Nachlass.'\n"
                f"Then explain what Wittgenstein seems to be saying in this remark, using ONLY the provided context.\n"
                f"Do not focus on explaining the identifier itself beyond that first sentence.\n"
                f"In the 'sources' field, return a Python list of the source identifiers you used (siglum strings).\n"
            )

        if kind == "doc" and len(valid_refs_siglum) == 1:
            doc = valid_refs_siglum[0]
            span = self._date_span_for_ref(doc)

            if span:
                y1 = int(str(span["min_datefrom"])[:4])
                y2 = int(str(span["max_dateto"])[:4])
                dating_line = f"roughly dated to {y1}." if y1 == y2 else f"roughly composed between {y1} and {y2}."
            else:
                dating_line = "with an uncertain dating based on available metadata."

            return (
                f"The user provided only this Wittgenstein Nachlass document identifier: {doc}.\n"
                f"Start with ONE short sentence: '{doc} is a document in Wittgenstein's Nachlass, {dating_line}'\n"
                f"Then summarize the main themes that emerge in the retrieved remarks, using ONLY the provided context. "
                f"Bullet points are allowed.\n"
                f"Do not invent themes beyond what is supported by the retrieved context.\n"
                f"Do not use technical phrasing like 'metadata dating' or counts; keep it natural.\n"
                f"In the 'sources' field, return a Python list of the source identifiers you used (siglum strings).\n"
            )

        refs_disp = ", ".join(valid_refs_siglum)
        return (
            f"The user provided only these Wittgenstein Nachlass identifiers: {refs_disp}.\n"
            f"For each identifier, briefly explain the content supported by the retrieved context, "
            f"then summarize any common themes.\n"
            f"In the 'sources' field, return a Python list of the source identifiers you used (siglum strings).\n"
        )

    @staticmethod
    def _should_boost_k_for_doc_query(original_question: str, document_level_refs: List[str], is_ref_only: bool) -> bool:
        if not document_level_refs:
            return False
        if is_ref_only:
            return True

        q = (original_question or "").strip().lower()
        overview_markers = [
            "what does wittgenstein say in", "what does he say in", "summarize", "summary", "overview",
            "main topics", "most important topics", "key themes", "main themes", "what are the topics",
            "what are the main", "give me an overview", "in general", "overall",
        ]
        return any(m in q for m in overview_markers)

    # -----------------------------
    # Language control for LLM2
    # -----------------------------

    # -----------------------------
    # Language control for LLM2
    # -----------------------------

    @staticmethod
    def _detect_user_language(q: str) -> str:
        s = (q or "").strip()
        if not s:
            return "EN"

        s_lower = s.lower()

        # Strong signals (umlauts etc.)
        if re.search(r"[äöüß]", s_lower):
            return "DE"
        if re.search(r"[æøå]", s_lower):
            return "NO"

        tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", s_lower)

        en_sw = {
            "the", "a", "an", "and", "or", "but", "if", "then",
            "what", "why", "how", "where", "when", "does", "do", "did",
            "say", "mean", "about", "in", "on", "of", "to", "with",
            "between", "explain", "compare", "main", "topic", "themes",
            "give", "list", "passages", "role",
        }

        de_sw = {
            "der", "die", "das", "und", "oder", "aber", "wenn", "dann",
            "was", "warum", "wie", "wo", "wann",
            "sagt", "meint", "schreibt", "denkt",
            "über", "in", "auf", "von", "zu", "mit", "zwischen",
            "erkläre", "vergleiche", "thema", "themen", "haupt",
            "alle", "denen", "bemerkung", "bemerkungen", "stellen",
            "sprache", "verwendet", "benutzt", "erwähnt",
        }

        it_sw = {
            "il", "lo", "la", "i", "gli", "le", "e", "o", "ma", "se",
            "allora", "che", "cosa", "perché", "come", "dove", "quando",
            "dice", "significa", "su", "in", "di", "a", "con", "tra",
            "spiega", "confronta", "tema", "temi", "principale",
        }

        no_sw = {
            "og", "eller", "men", "hvis", "da", "hva", "hvorfor",
            "hvordan", "hvor", "når", "sier", "betyr", "om", "i", "på",
            "av", "til", "med", "mellom", "forklar", "sammenlign",
            "tema", "temaer", "hoved", "det", "den", "de", "er", "ikke",
            "dette", "disse", "hvilke", "kan", "har", "være", "som",
        }

        scores = {
            "EN": sum(1 for t in tokens if t in en_sw),
            "DE": sum(1 for t in tokens if t in de_sw),
            "IT": sum(1 for t in tokens if t in it_sw),
            "NO": sum(1 for t in tokens if t in no_sw),
        }

        best_lang = max(scores, key=scores.get)
        best_val = scores[best_lang]
        second_val = sorted(scores.values(), reverse=True)[1]

        # Strong decision
        if best_val >= 2 and best_val >= second_val + 1:
            return best_lang

        # 👇 FIX CHIAVE: anche con 1 solo segnale chiaro
        if best_val >= 1 and second_val == 0:
            return best_lang

        # Fallback (ASCII → English)
        ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(len(s), 1)
        if ascii_ratio >= 0.95:
            return "EN"

        return "EN"

    @staticmethod
    def _language_instruction(lang_code: str) -> str:
        if lang_code == "DE":
            return "Answer in German."
        if lang_code == "IT":
            return "Answer in Italian."
        if lang_code == "NO":
            return "Answer in Norwegian."
        return "Answer in English."
    
    # -----------------------------
    # Conflict detection
    # -----------------------------

    def _date_span_for_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        if not ref:
            return None

        if ref in self.known_sigla:
            df = self.DF_wittgenstein[self.DF_wittgenstein["siglum"] == ref]
        else:
            df = self.DF_wittgenstein[self.DF_wittgenstein["siglum"].astype(str).str.startswith(ref)]

        if df.empty:
            return None

        min_df = int(df["datefrom"].min())
        max_dt = int(df["dateto"].max())
        return {"ref": ref, "n": int(len(df)), "min_datefrom": min_df, "max_dateto": max_dt}

    def _detect_metadata_conflict(self, llm1_out: Dict[str, Any]) -> Dict[str, Any]:
        listrefs_esc = llm1_out.get("listrefs") or []
        listdates = llm1_out.get("listdates") or []

        if not (listrefs_esc and isinstance(listdates, list) and len(listdates) == 2):
            return {"metadata_conflict": False, "message": "", "details": {}}

        listrefs = [_unescape_backslashes(r) for r in listrefs_esc]

        try:
            q_start = int(listdates[0])
            q_end = int(listdates[1])
        except Exception:
            return {"metadata_conflict": False, "message": "", "details": {}}

        spans: List[Dict[str, Any]] = []
        for r in listrefs:
            s = self._date_span_for_ref(r)
            if s:
                spans.append(s)

        if not spans:
            return {
                "metadata_conflict": True,
                "message": "Your filters returned no results. I couldn't verify the reference dates to explain the mismatch.",
                "details": {"listrefs": listrefs, "listdates": listdates, "spans": []},
            }

        non_overlapping = []
        for s in spans:
            if not (s["min_datefrom"] <= q_end and s["max_dateto"] >= q_start):
                non_overlapping.append(s)

        if len(non_overlapping) == len(spans):
            parts = []
            for s in non_overlapping:
                d1 = formatdate_yyyy_mm_dd(s["min_datefrom"])
                d2 = formatdate_yyyy_mm_dd(s["max_dateto"])
                if d1 == d2:
                    parts.append(f"{s['ref']} is dated {d1}")
                else:
                    parts.append(f"{s['ref']} spans {d1}–{d2}")

            q1 = formatdate_yyyy_mm_dd(q_start)
            q2 = formatdate_yyyy_mm_dd(q_end)
            msg = (
                f"Your metadata filters are contradictory: you asked for the period {q1}–{q2}, "
                f"but {', '.join(parts)}. Please adjust either the reference(s) or the date range."
            )
            return {
                "metadata_conflict": True,
                "message": msg,
                "details": {"listrefs": listrefs, "listdates": listdates, "spans": spans, "non_overlapping": non_overlapping},
            }

        return {
            "metadata_conflict": True,
            "message": "Your filters returned no results. The reference(s) partially overlap the requested dates, but no matching rows were found. Consider widening the date range or checking the reference.",
            "details": {"listrefs": listrefs, "listdates": listdates, "spans": spans},
        }

    def _base_return_payload(
        self,
        *,
        session_id: str,
        question: str,
        processed_q: str,
        referenceList: List[str],
        valid_refs_siglum: List[str],
        invalid_refs_siglum: List[str],
        document_level_refs: List[str],
        referenceList_for_llm1: List[str],
        k: int,
        retrieval_mode: str,
    ) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "input_question": question,
            "processed_question": processed_q,
            "referenceList_raw": referenceList,
            "valid_refs_df": valid_refs_siglum,
            "invalid_refs_df": invalid_refs_siglum,
            "document_level_refs": document_level_refs,
            "referenceList_for_llm1": referenceList_for_llm1,
            "filter_json": {},
            "llm1_drift_refs": [],
            "date_normalization_mode": "none",
            "filter_expression": None,
            "retrieval_query": "",
            "retrieval_query_plain": "",
            "retrieval_query_expanded": "",
            "retrieval_query_fallback_used": False,
            "query_expansion_applied": False,
            "query_translation_targets": [],
            "query_translations": {},
            "retrieval_mode": retrieval_mode,
            "k": k,
            "docs": [],
            "docs_siglum": [],
            "sources_raw": [],
            "sources": ["No source"],
            "metadata_conflict": False,
            "metadata_conflict_message": "",
            "metadata_conflict_details": {},
        }

    # -----------------------------
    # Core ask()
    # -----------------------------

    def ask(
        self,
        question: str,
        *,
        user_template_text: Optional[str] = None,
        k: Optional[int] = None,
        temperature: Optional[float] = None,
        retrieval_mode: Optional[str] = None,
        use_multilingual_query_expansion: Optional[bool] = None,
    ) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        _ = temperature

        retrieval_mode = (retrieval_mode or self.default_retrieval_mode or "vector").strip().lower()

        if use_multilingual_query_expansion is None:
            use_multilingual_query_expansion = self.use_multilingual_query_expansion

        if k is None:
            k = self.default_k_num

        if user_template_text is None:
            user_template_text = self.default_user_template

        processed_q, referenceList = process_sentence_in_pattern(question)

        valid_refs_siglum, invalid_refs_siglum = self._extract_and_validate_refs(referenceList)
        self.last_valid_refs = valid_refs_siglum
        self.last_invalid_refs = invalid_refs_siglum

        if referenceList and not valid_refs_siglum:
            self.last_document_level_refs = []
            invalid_disp = ", ".join(invalid_refs_siglum) if invalid_refs_siglum else "the provided reference(s)"
            out = self._base_return_payload(
                session_id=session_id,
                question=question,
                processed_q=processed_q,
                referenceList=referenceList,
                valid_refs_siglum=valid_refs_siglum,
                invalid_refs_siglum=invalid_refs_siglum,
                document_level_refs=[],
                referenceList_for_llm1=[],
                k=k,
                retrieval_mode=retrieval_mode,
            )
            out["answer"] = f"I could not find the reference(s): {invalid_disp}. Please check the identifier."
            return out

        if valid_refs_siglum and invalid_refs_siglum:
            self.last_document_level_refs = [
                r for r in valid_refs_siglum
                if r in self.known_siglum_prefixes and r not in self.known_sigla
            ]
            valid_disp = ", ".join(valid_refs_siglum)
            invalid_disp = ", ".join(invalid_refs_siglum)
            out = self._base_return_payload(
                session_id=session_id,
                question=question,
                processed_q=processed_q,
                referenceList=referenceList,
                valid_refs_siglum=valid_refs_siglum,
                invalid_refs_siglum=invalid_refs_siglum,
                document_level_refs=self.last_document_level_refs,
                referenceList_for_llm1=[re.escape(s) for s in valid_refs_siglum],
                k=k,
                retrieval_mode=retrieval_mode,
            )
            out["answer"] = (
                f"I found the following valid reference(s): {valid_disp}. "
                f"But I could not find: {invalid_disp}. "
                f"Please correct the invalid reference(s) if you want a reliable comparison or joint answer."
            )
            return out

        document_level_refs = [
            r for r in valid_refs_siglum
            if r in self.known_siglum_prefixes and r not in self.known_sigla
        ]
        self.last_document_level_refs = document_level_refs
        referenceList_for_llm1 = [re.escape(s) for s in valid_refs_siglum]

        llm1_out = self.filter_chain.invoke({"referenceList": referenceList_for_llm1, "user_query": processed_q})

        if valid_refs_siglum:
            llm1_out["listrefs"] = [re.escape(r) for r in valid_refs_siglum]

        llm1_out = self._patch_llm1_dates(llm1_out, question)

        drift_refs: List[str] = []
        if valid_refs_siglum:
            allowed = set(valid_refs_siglum)
            got = [_unescape_backslashes(x) for x in (llm1_out.get("listrefs") or [])]
            drift_refs = [x for x in got if x not in allowed]

        filter_expression = self.get_filter_from_df(llm1_out)

        is_ref_only = self._is_ref_only_query(question, valid_refs_siglum)
        if self._should_boost_k_for_doc_query(question, document_level_refs, is_ref_only):
            k = max(k, 25)

        metadata_conflict_info = {"metadata_conflict": False, "message": "", "details": {}}
        if (llm1_out.get("listrefs") and llm1_out.get("listdates")) and (filter_expression is None):
            metadata_conflict_info = self._detect_metadata_conflict(llm1_out)
            if metadata_conflict_info.get("metadata_conflict"):
                return {
                    "session_id": session_id,
                    "input_question": question,
                    "processed_question": processed_q,
                    "referenceList_raw": referenceList,
                    "valid_refs_df": valid_refs_siglum,
                    "invalid_refs_df": invalid_refs_siglum,
                    "document_level_refs": document_level_refs,
                    "referenceList_for_llm1": referenceList_for_llm1,
                    "filter_json": llm1_out,
                    "llm1_drift_refs": drift_refs,
                    "date_normalization_mode": self.last_date_normalization_mode,
                    "filter_expression": filter_expression,
                    "retrieval_query": "",
                    "retrieval_query_plain": "",
                    "retrieval_query_expanded": "",
                    "retrieval_query_fallback_used": False,
                    "query_expansion_applied": False,
                    "query_translation_targets": [],
                    "query_translations": {},
                    "retrieval_mode": retrieval_mode,
                    "k": k,
                    "docs": [],
                    "docs_siglum": [],
                    "answer": metadata_conflict_info.get("message", "Your filters returned no results."),
                    "sources_raw": [],
                    "sources": ["No source"],
                    "metadata_conflict": True,
                    "metadata_conflict_message": metadata_conflict_info.get("message", ""),
                    "metadata_conflict_details": metadata_conflict_info.get("details", {}),
                }

        has_dates = bool(llm1_out.get("listdates"))

        # --- Deterministic semantic query extraction ---
        retrieval_query_plain = self._build_relevant_semantic_query(
            original_question=question,
            processed_q=processed_q,
            llm1_out=llm1_out,
            valid_refs_siglum=valid_refs_siglum,
            has_dates=has_dates,
        )

        retrieval_query_fallback_used = False

        if len(retrieval_query_plain) < self.retrieval_min_query_chars:
            if is_ref_only:
                retrieval_query_plain = "context"
            else:
                fallback_q = self._cleanup_semantic_query(processed_q)
                fallback_q = self._strip_listrefs_from_text(
                    fallback_q,
                    listrefs_escaped=(llm1_out.get("listrefs") or []),
                    valid_refs_siglum=valid_refs_siglum,
                )
                if has_dates:
                    fallback_q = self._strip_dates_scaffolding(fallback_q)
                fallback_q = self._strip_years(fallback_q)
                fallback_q = self._strip_sigla_tokens(fallback_q)
                fallback_q = self._strip_corpus_author_boilerplate(fallback_q)
                retrieval_query_plain = fallback_q or "context"

            retrieval_query_fallback_used = True

        # --- Literal multilingual query translation ---
        retrieval_query_expanded = retrieval_query_plain
        query_expansion_applied = False
        query_translation_targets: List[str] = []
        query_translations: Dict[str, str] = {}

        if use_multilingual_query_expansion and retrieval_query_plain.strip():
            (
                retrieval_query_expanded,
                query_expansion_applied,
                query_translation_targets,
                query_translations,
            ) = self._build_multilingual_retrieval_query(
                retrieval_query_plain,
                original_question=question,
            )

        retrieval_query = retrieval_query_expanded

        self.last_retrieval_query_plain = retrieval_query_plain
        self.last_retrieval_query_expanded = retrieval_query_expanded
        self.last_query_expansion_applied = query_expansion_applied
        self.last_query_translation_targets = query_translation_targets
        self.last_query_translations = query_translations

        docs = self._retrieve_documents(
            retrieval_query=retrieval_query,
            k=k,
            filter_expression=filter_expression,
            retrieval_mode=retrieval_mode,
        )

        if self.debug:
            print("[DEBUG] Index:", os.getenv("AZURE_AI_SEARCH_INDEX_NAME"))
            print("[DEBUG] Retrieval mode:", retrieval_mode)
            print("[DEBUG] Valid refs (DF):", valid_refs_siglum)
            print("[DEBUG] Invalid refs (DF):", invalid_refs_siglum)
            print("[DEBUG] Document-level refs:", document_level_refs)
            if drift_refs:
                print("[DEBUG][WARN] listrefs drift:", drift_refs)
            print("[DEBUG] Date normalization mode:", self.last_date_normalization_mode)
            print("[DEBUG] Candidate pool size:", getattr(self, "last_candidate_pool_size", None))
            print("[DEBUG] Candidate pool preview:", getattr(self, "last_candidate_pool_preview", None))
            if filter_expression:
                print("[DEBUG] Filter chars:", len(filter_expression))
            print("[DEBUG] Retrieval query plain:", retrieval_query_plain)
            print("[DEBUG] Query expansion applied:", query_expansion_applied)
            print("[DEBUG] Query translation targets:", query_translation_targets)
            print("[DEBUG] Query translations:", query_translations)
            print("[DEBUG] Retrieval query:", retrieval_query)
            print("[DEBUG] Retrieval fallback used:", retrieval_query_fallback_used)
            print("[DEBUG] Retrieved docs:", len(docs), "(k requested:", k, ")")
            print("[DEBUG] Top 5 retrieved sigla:", [d.metadata.get("siglum") for d in docs[:5]])

        docs_siglumList = [doc.metadata.get("siglum", "") for doc in docs]

        docs_content = "\n\n".join(
            f"{{ source: {doc.metadata.get('siglum','')}, period: {doc.metadata.get('datefrom','')} to {doc.metadata.get('dateto','')}, context: {doc.page_content}}} \n\n"
            for doc in docs
        )

        question_for_llm2 = processed_q
        if is_ref_only:
            question_for_llm2 = self._build_llm2_question_for_ref_only(
                original_question=question,
                valid_refs_siglum=valid_refs_siglum,
            )

        if is_ref_only:
            lang_inst = "Answer in English."
        else:
            lang_code = self._detect_user_language(question)
            lang_inst = self._language_instruction(lang_code)

        if self.debug:
            print("[DEBUG] LLM2 language:", lang_inst)

        user_template_text = user_template_text + " " + lang_inst

        qa_prompt_edit = {
            "user_template_text": user_template_text,
            "question": question_for_llm2,
            "context": docs_content,
        }
        bot_response = self.chain.invoke(qa_prompt_edit)
        answer = bot_response.get("answer", "")

        meta_source = bot_response.get("sources", [])
        meta_source = [_unescape_backslashes(aref) for aref in meta_source]

        if (not meta_source) or all(x == "" for x in meta_source):
            meta_source_disp: Any = ["No source"]
        else:
            kval_exact: List[str] = []
            kval_rejected: List[str] = []

            for kval in meta_source:
                if kval != "":
                    if kval in docs_siglumList:
                        kval_exact.append(kval)
                    else:
                        kval_rejected.append(kval)

            kval_exact = list(dict.fromkeys(kval_exact))
            kval_rejected = list(dict.fromkeys(kval_rejected))

            if kval_exact:
                meta_source_disp = {"exact_siglum": kval_exact, "rejected_siglum": kval_rejected}
            else:
                meta_source_disp = ["No source"]

        return {
            "session_id": session_id,
            "input_question": question,
            "processed_question": processed_q,
            "referenceList_raw": referenceList,
            "valid_refs_df": valid_refs_siglum,
            "invalid_refs_df": invalid_refs_siglum,
            "document_level_refs": document_level_refs,
            "referenceList_for_llm1": referenceList_for_llm1,
            "filter_json": llm1_out,
            "llm1_drift_refs": drift_refs,
            "date_normalization_mode": self.last_date_normalization_mode,
            "filter_expression": filter_expression,
            "retrieval_query": retrieval_query,
            "retrieval_query_plain": retrieval_query_plain,
            "retrieval_query_expanded": retrieval_query_expanded,
            "retrieval_query_fallback_used": retrieval_query_fallback_used,
            "query_expansion_applied": query_expansion_applied,
            "query_translation_targets": query_translation_targets,
            "query_translations": query_translations,
            "retrieval_mode": retrieval_mode,
            "k": k,
            "docs": docs,
            "docs_siglum": docs_siglumList,
            "answer": answer,
            "sources_raw": meta_source,
            "sources": meta_source_disp,
            "metadata_conflict": metadata_conflict_info.get("metadata_conflict", False),
            "metadata_conflict_message": metadata_conflict_info.get("message", ""),
            "metadata_conflict_details": metadata_conflict_info.get("details", {}),
        }

    # -----------------------------
    # Convenience display
    # -----------------------------

    def pretty_print(self, out: Dict[str, Any], *, show_docs: int = 10) -> None:
        print("=" * 80)
        print("QUESTION:")
        print(out.get("input_question", ""))
        print("-" * 80)

        if out.get("metadata_conflict"):
            print("[METADATA CONFLICT]")
            print(out.get("metadata_conflict_message", ""))
            print("-" * 80)

        print("RETRIEVAL MODE:")
        print(out.get("retrieval_mode", ""))
        print("-" * 80)

        print("FILTER EXPRESSION (OData):")
        fe = out.get("filter_expression", None)
        print(fe)
        if fe:
            print("Filter chars:", len(fe))
        print("-" * 80)

        print("RETRIEVAL QUERY PLAIN:")
        print(out.get("retrieval_query_plain", out.get("retrieval_query", "")))
        print("-" * 80)

        print("QUERY EXPANSION APPLIED:")
        print(out.get("query_expansion_applied", False))
        print("-" * 80)

        print("QUERY TRANSLATION TARGETS:")
        print(out.get("query_translation_targets", []))
        print("-" * 80)

        print("QUERY TRANSLATIONS:")
        print(out.get("query_translations", {}))
        print("-" * 80)

        print("RETRIEVAL QUERY USED (embedder/search input):")
        print(out.get("retrieval_query", ""))
        print("Fallback used:", out.get("retrieval_query_fallback_used", False))
        print("-" * 80)

        print("ANSWER:")
        print(out.get("answer", ""))
        print("-" * 80)

        print("SOURCES:")
        print(out.get("sources", []))
        print("-" * 80)

        docs = out.get("docs", []) or []
        if docs:
            print(f"TOP DOCS (showing {min(show_docs, len(docs))} of {len(docs)}):")
            for i, d in enumerate(docs[:show_docs], start=1):
                sig = d.metadata.get("siglum", "")
                df_ = d.metadata.get("datefrom", "")
                dt_ = d.metadata.get("dateto", "")
                score = d.metadata.get("score", None)
                snippet = (d.page_content or "").replace("\n", " ")
                if len(snippet) > 220:
                    snippet = snippet[:220] + "..."
                if score is not None:
                    print(f"{i:02d}. {sig}  ({df_}–{dt_})  [score={score}]  ::  {snippet}")
                else:
                    print(f"{i:02d}. {sig}  ({df_}–{dt_})  ::  {snippet}")
        else:
            print("No docs retrieved.")

        print("=" * 80)


# -----------------------------
# CLI entry point
# -----------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python witt_histochat_jupyter.py "your question"')
        raise SystemExit(1)

    q = sys.argv[1]
    bot = HistoryBot(
        json_searchindex_file_path=os.getenv(
            "WITT_DF_PATH",
            "assets-json/DF-wittgenstein_dates_fixed.json"
        ),
        debug=True,
    )
    out = bot.ask(q, k=10)
    bot.pretty_print(out)
