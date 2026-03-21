from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Azure Search field mapping ---
os.environ["AZURESEARCH_FIELDS_ID"] = "chunk_id"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"

# --- LangChain imports ---
from typing_extensions import Annotated, TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

    We now enforce full comma->underscore normalization when the comma is part
    of the siglum token (i.e., comma followed by an alphanumeric char):
      - Ms-106,213[2] -> Ms-106_213[2]
      - Ms-101,IIr[1] -> Ms-101_IIr[1]
    Punctuation-comma at end should NOT become an underscore:
      - Ms-114,  (then later .rstrip(",") removes it)
    """
    s = str(sig)

    # Remove spaces after comma within sigla (Ms-101, IIr -> Ms-101,IIr)
    s = re.sub(r",\s+(?=[A-Z0-9\[])", ",", s)

    # Convert comma to underscore when followed by an alphanumeric char
    s = re.sub(r",(?=[A-Za-z0-9])", "_", s)

    return s


def process_sentence_in_pattern(sentence: str) -> Tuple[str, List[str]]:
    """
    Robust extraction of Ms-/Ts- sigla.

    Accepts both underscore-style and comma-style user input, but normalizes to DF-style (underscore)
    for downstream validation. Avoids punctuation-comma corruption: "Ms-114," -> "Ms-114".

    IMPORTANT PATCH:
    - Prevent false captures like "Ts-309_find" in sentences like:
        "In Ts-309, find remarks ..."
      The bug came from collapsing comma+space globally (", " -> ",") which turns
      "Ts-309, find" into "Ts-309,find", making "find" look like part of the siglum.

    We now collapse comma+spaces ONLY when what follows looks like a genuine siglum continuation:
    - uppercase letter (e.g., IIr)
    - digit (e.g., 213)
    - '['
    """
    sentence = _normalize_ms_ts_variants(sentence)

    # Collapse comma+spaces ONLY for siglum-like continuations (NOT for normal prose like ", find")
    sentence = re.sub(r",\s+(?=[A-Z0-9\[])", ",", sentence)

    # Punctuation spacing: IMPORTANT -> do NOT space commas, otherwise Ms-106,213[2] gets split.
    sentence = re.sub(r"\s*([?.!;:\"(){}])\s*", r" \1 ", sentence)
    sentence = re.sub(r"\s*(['])\s*", r" \1 ", sentence)

    # Siglum pattern: allow letters/digits/_/[,]/, plus optional repeated 'et...' segments.
    ref_pattern = re.compile(
        r"\b(?:Ms-|Ts-)\d{3}[A-Za-z0-9_,\[\]]*(?:et[A-Za-z0-9_,\[\]]+)*"
    )

    found_refs = [m.group(0) for m in ref_pattern.finditer(sentence)]

    referenceList: List[str] = []
    for ref in found_refs:
        ref_norm = _user_siglum_to_df_siglum(ref)
        ref_norm = ref_norm.rstrip(",")  # drop punctuation comma at end (e.g., "Ms-114,")
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
            "List of sources (each source starts with 'Ms\\-' or 'Ts\\-' and ends before the first whitespace) used to answer the question. Return an empty list [] if no sources are used.",
        ]

    def __init__(
        self,
        json_searchindex_file_path: str = os.getenv(
            "WITT_DF_PATH",
            "assets-json/DF-wittgenstein-nonNAComma_FULL.json"
        ),
        default_temperature: float = 1.0,
        default_k_num: int = 500,
        retrieval_min_query_chars: int = 6,
        debug: bool = False,
    ) -> None:

        # --- Environment variables ---
        os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME", "")
        os.environ["AZURE_AI_SEARCH_INDEX_NAME"] = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "")
        os.environ["AZURE_AI_SEARCH_API_KEY"] = os.getenv("AZURE_AI_SEARCH_API_KEY", "")

        self.default_temperature = default_temperature
        self.default_k_num = default_k_num
        self.retrieval_min_query_chars = retrieval_min_query_chars
        self.debug = debug

        self.default_user_template = (
            " You are an assistant for question-answering tasks, primarily for answering questions about Ludwig Wittgenstein. "
            "The context will primarily be in German, with some parts in English. Use only the provided retrieved context "
            "to answer the question. Keep the answer accurate and well-explained. Only respond with 'I don't know', "
            "if the provided retrieved context is completely irrelevant to the question. "
        )

        # --- Load DF ---
        self.json_searchindex_file_path = json_searchindex_file_path
        if json_searchindex_file_path and os.path.exists(json_searchindex_file_path):
            self.DF_wittgenstein = self._load_df(json_searchindex_file_path)
        else:
            self.DF_wittgenstein = pd.DataFrame(columns=["siglum", "datefrom", "dateto", "refcontent"])

        self.known_sigla = set(self.DF_wittgenstein["siglum"].astype(str).tolist())
        self.known_siglum_prefixes = set(str(s).split("_", 1)[0] for s in self.known_sigla)

        # --- Models/embeddings ---
        self.azure_code_llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_deployment=os.getenv("MODEL_AZURE_CODE_DEPLOYMENT"),
            model=os.getenv("MODEL_AZURE_CODE_DEPLOYMENT_NAME"),
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
        )

        self.structured_azure_llm = self.azure_llm.with_structured_output(
            self.AnswerWithSources,
            method="function_calling",
        )

        # --- Filter chain (LLM1) ---
        self.json_parser = JsonOutputParser()
        self.filter_prompt = self._build_filter_prompt()
        self.filter_chain = (
            {
                "user_query": RunnablePassthrough(),
                "referenceList": RunnablePassthrough(),
            }
            | self.filter_prompt
            | self.azure_code_llm
            | self.json_parser
        )

        # --- Final prompt chain (LLM2) ---
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

        # --- Azure Search vector store ---
        vector_store_address: str = f"https://{os.getenv('AZURE_AI_SEARCH_SERVICE_NAME')}.search.windows.net"
        index_name: str = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

        self.azure_search: AzureSearch = azuresearch.AzureSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
            index_name=index_name,
            embedding_function=self.embeddings.embed_query,
            additional_search_client_options={"retry_total": 4},
        )

        # Debug: last run metadata
        self.last_candidate_pool_size = 0
        self.last_candidate_pool_preview: List[str] = []
        self.last_valid_refs: List[str] = []
        self.last_invalid_refs: List[str] = []
        self.last_document_level_refs: List[str] = []
        self.last_date_normalization_mode: str = "none"

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
        """
        Build an Azure Search OData filter expression.

        Strategy (compatible, no startswith):
        - DATE-ONLY: use native Azure date filters (short, scalable)
        - REFS-ONLY: use DF -> siglum list -> search.in(siglum, ...)
        - REFS + DATES: use DF -> siglum list (already restricted by both) -> search.in(siglum, ...)
        """

        # --- reset debug counters ---
        self.last_candidate_pool_size = 0
        self.last_candidate_pool_preview = []

        listrefs_esc = filter_val_dct.get("listrefs") or []
        listdates = filter_val_dct.get("listdates") or []

        has_refs = bool(listrefs_esc)
        has_dates = bool(listdates)

        # --- normalize refs (escaped -> plain) ---
        listrefs: List[str] = [_unescape_backslashes(r) for r in listrefs_esc] if has_refs else []

        # --- normalize dates ---
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

        # ------------------------------------------------------------
        # DATE ONLY  (native, short)
        # ------------------------------------------------------------
        if has_dates and not has_refs:
            filtered_df = self.DF_wittgenstein[
                (self.DF_wittgenstein["datefrom"] >= d_start)
                & (self.DF_wittgenstein["dateto"] <= d_end)
            ]
            if not filtered_df.empty:
                sigla = list(filtered_df["siglum"].astype(str))
                self.last_candidate_pool_size = len(sigla)
                self.last_candidate_pool_preview = sigla[:5]

            return f"datefrom ge {d_start} and dateto le {d_end}"

        # ------------------------------------------------------------
        # REFS ONLY  (DF -> search.in)
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # REFS + DATES  (DF -> search.in)
        # ------------------------------------------------------------
        if has_refs and has_dates:
            filtered_df = self.DF_wittgenstein[
                (self.DF_wittgenstein["siglum"].astype(str).str.startswith(tuple(listrefs)))
                & (self.DF_wittgenstein["datefrom"] >= d_start)
                & (self.DF_wittgenstein["dateto"] <= d_end)
            ]
            if filtered_df.empty:
                return None

            sigla = list(filtered_df["siglum"].astype(str))
            self.last_candidate_pool_size = len(sigla)
            self.last_candidate_pool_preview = sigla[:5]

            return _search_in_siglum(sigla)

        # ------------------------------------------------------------
        # NO METADATA
        # ------------------------------------------------------------
        return None

    # -----------------------------
    # Guard-rails helpers
    # -----------------------------

    def _extract_and_validate_refs(self, referenceList_escaped: List[str]) -> Tuple[List[str], List[str]]:
        """
        Accept:
          - exact sigla present in DF (remark-level), e.g. Ms-105_2[1]
          - document-level prefixes present in DF, e.g. Ms-114 or Ts-201a1
        """
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
        years = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", question_text)]
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
        """
        Normalize date ranges deterministically.

        Goal:
        - If the user speaks in YEARS (e.g. "between 1932 and 1934"),
          interpret the range as inclusive of the whole final year.
        - Fix LLM outputs like: end = YYYY0101 -> YYYY1231
        """

        out = dict(llm1_out)
        ld = out.get("listdates") or []

        # Extract years explicitly mentioned by the user
        years_in_q = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", original_question)]

        # ------------------------------------------------------------
        # Case 1: LLM1 produced a valid yyyymmdd range
        # ------------------------------------------------------------
        if (
            isinstance(ld, list)
            and len(ld) == 2
            and self._is_yyyymmdd(ld[0])
            and self._is_yyyymmdd(ld[1])
        ):
            start = str(ld[0])
            end = str(ld[1])

            # If the user talked in years and the end date is YYYY0101,
            # interpret it as end-of-year (YYYY1231)
            if years_in_q and end.endswith("0101"):
                end = end[:4] + "1231"
                out["listdates"] = [start, end]
                out["listdates_patched"] = [start, end]
                self.last_date_normalization_mode = "kept_llm1_yearend"
                return out

            # Otherwise keep LLM1 dates as-is
            out["listdates"] = [start, end]
            out["listdates_patched"] = [start, end]
            self.last_date_normalization_mode = "kept_llm1"
            return out

        # ------------------------------------------------------------
        # Case 2: LLM1 failed -> fallback to deterministic year parsing
        # ------------------------------------------------------------
        norm = self._normalize_dates_from_text(original_question)
        if norm:
            out["listdates"] = norm
            out["listdates_patched"] = norm
            self.last_date_normalization_mode = "fallback_years"
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

        # Remove orphan punctuation at start (common after stripping dates/refs)
        mq = re.sub(r"^[?.!,;:]+", "", mq).strip()
        mq = re.sub(r"^[)\]]+", "", mq).strip()
        mq = re.sub(r"^[,;:]\s*", "", mq).strip()

        mq = re.sub(r"\s+", " ", mq).strip()

        return mq

    def _strip_sigla_tokens(self, text: str) -> str:
        """
        LAST-MILE GUARANTEE:
        Remove any Ms-/Ts- siglum tokens from the text to be embedded.
        Works even if tokens are still escaped (contain backslashes).
        """
        t = _unescape_backslashes(text)

        token_pat = re.compile(
            r"\b(?:Ms-|Ts-)\d{3}[A-Za-z0-9_,\[\]]*(?:et[A-Za-z0-9_,\[\]]+)*\b"
        )
        t = token_pat.sub(" ", t)

        return self._cleanup_semantic_query(t)

    def _strip_years(self, text: str) -> str:
        return re.sub(r"\b(19\d{2}|20\d{2})\b", " ", text)

    def _strip_dates_scaffolding(self, text: str) -> str:
        t = text
        t = self._strip_years(t)
        # remove range glue words often left after stripping years
        t = re.sub(r"\b(between|from)\b\s+\b(and|to)\b", " ", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(between|from|to|and|in)\b\s+(?=[?.!,;:])", " ", t, flags=re.IGNORECASE)
        return self._cleanup_semantic_query(t)

    def _strip_listrefs_from_text(self, text: str, listrefs_escaped: List[str], valid_refs_siglum: List[str]) -> str:
        """
        Deterministically remove references from a text:
        - remove escaped refs from llm1_out["listrefs"]
        - remove plain refs from valid_refs_siglum (and common variants)
        """
        t = text or ""

        # remove escaped refs directly
        for ref_esc in listrefs_escaped or []:
            t = t.replace(ref_esc, " ")
            t = t.replace(_unescape_backslashes(ref_esc), " ")

        # remove validated refs (and some variants)
        for r in valid_refs_siglum or []:
            variants = {r, r.replace("_", ","), r.replace(",", "_")}
            for v in variants:
                t = t.replace(v, " ")
                t = t.replace(re.escape(v), " ")

        return self._cleanup_semantic_query(t)

    # -----------------------------
    # NEW: ref-only intent routing for LLM2
    # -----------------------------

    @staticmethod
    def _is_ref_only_query(original_question: str, valid_refs_siglum: List[str]) -> bool:
        """
        True iff the user input is essentially only identifier(s).
        Conservative: triggers only if, after removing the recognized refs and punctuation, almost nothing remains.
        """
        if not valid_refs_siglum:
            return False

        s = (original_question or "").strip()

        for r in valid_refs_siglum:
            for v in {r, r.replace("_", ","), r.replace(",", "_")}:
                s = s.replace(v, " ")

        s = re.sub(r"[^\w]+", " ", s).strip()
        return len(s) < 6

    def _ref_kind(self, valid_refs_siglum: List[str]) -> str:
        """
        "remark" if ALL refs are exact sigla (in DF)
        "doc"    if ALL refs are doc-level prefixes (in prefixes but not exact sigla)
        "mixed"  otherwise
        """
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
        """
        Build an explicit task question for LLM2 when the user input is ref-only.
        - remark-level: identify as Nachlass remark + explain content
        - doc-level: identify as Nachlass document + brief thematic overview (from retrieved context only)
        """
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
                if y1 == y2:
                    dating_line = f"roughly dated to {y1}."
                else:
                    dating_line = f"roughly composed between {y1} and {y2}."
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

        # multiple or mixed ids
        refs_disp = ", ".join(valid_refs_siglum)
        return (
            f"The user provided only these Wittgenstein Nachlass identifiers: {refs_disp}.\n"
            f"For each identifier, briefly explain the content supported by the retrieved context, "
            f"then summarize any common themes.\n"
            f"In the 'sources' field, return a Python list of the source identifiers you used (siglum strings).\n"
        )

    # -----------------------------
    # NEW: doc-level k-boost only for overview intent
    # -----------------------------

    @staticmethod
    def _should_boost_k_for_doc_query(original_question: str, document_level_refs: List[str], is_ref_only: bool) -> bool:
        """
        Boost k only for document-level OVERVIEW intents.

        Boost if:
        - ref-only doc id (e.g. "Ms-114")
        - doc-level ref present AND the question looks like an overview request

        Do NOT boost for topic-focused questions like:
        - "about objects", "about grammar", "about ostensive definition", etc.
        """
        if not document_level_refs:
            return False
        if is_ref_only:
            return True

        q = (original_question or "").strip().lower()

        overview_markers = [
            "what does wittgenstein say in", "what does he say in",
            "summarize", "summary", "overview",
            "main topics", "most important topics", "key themes", "main themes",
            "what are the topics", "what are the main", "give me an overview",
            "in general", "overall"
        ]

        if any(m in q for m in overview_markers):
            return True

        return False

    # -----------------------------
    # NEW: language control for LLM2 (EN/DE/IT heuristic)
    # -----------------------------

    # -----------------------------
    # NEW: language control for LLM2 (EN/DE/IT heuristic)
    # -----------------------------

    @staticmethod
    def _detect_user_language(q: str) -> str:
        """
        Heuristic language detection for user query: EN / DE / IT / NO.

        Priority:
        1) Strong German chars (äöüß) => DE
        2) Strong Norwegian chars (æøå) => NO
        3) Stopword vote (EN/DE/IT/NO)
        4) ASCII ratio fallback => EN

        Returns: "EN", "DE", "IT", or "NO"
        """
        s = (q or "").strip()
        if not s:
            return "EN"

        s_lower = s.lower()

        # 1) Strong German characters
        if re.search(r"[äöüß]", s_lower):
            return "DE"

        # 2) Strong Norwegian characters (æ, ø, å)
        if re.search(r"[æøå]", s_lower):
            return "NO"

        # 3) Stopword vote (small, robust sets)
        tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", s_lower)

        en_sw = {
            "the","a","an","and","or","but","if","then","what","why","how","where","when","does","do","did",
            "say","mean","about","in","on","of","to","with","between","explain","compare","main","topic","themes"
        }
        de_sw = {
            "der","die","das","und","oder","aber","wenn","dann","was","warum","wie","wo","wann","sagt","meint",
            "über","in","auf","von","zu","mit","zwischen","erkläre","vergleiche","thema","themen","haupt"
        }
        it_sw = {
            "il","lo","la","i","gli","le","e","o","ma","se","allora","che","cosa","perché","come","dove","quando",
            "dice","significa","su","in","di","a","con","tra","spiega","confronta","tema","temi","principale"
        }
        no_sw = {
            "og","eller","men","hvis","da","hva","hvorfor","hvordan","hvor","når","sier","betyr","om","i","på",
            "av","til","med","mellom","forklar","sammenlign","tema","temaer","hoved","det","den","de","er","ikke",
            "dette","disse","hvilke","kan","har","være","som"
        }

        en_score = sum(1 for t in tokens if t in en_sw)
        de_score = sum(1 for t in tokens if t in de_sw)
        it_score = sum(1 for t in tokens if t in it_sw)
        no_score = sum(1 for t in tokens if t in no_sw)

        scores = {"EN": en_score, "DE": de_score, "IT": it_score, "NO": no_score}
        best_lang = max(scores, key=scores.get)
        best_val = scores[best_lang]
        second_val = sorted(scores.values(), reverse=True)[1]

        if best_val >= 2 and best_val >= second_val + 1:
            return best_lang

        # 4) ASCII ratio fallback
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
    ) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        _ = temperature

        if k is None:
            k = self.default_k_num

        if user_template_text is None:
            user_template_text = self.default_user_template

        # 1) preprocess
        processed_q, referenceList = process_sentence_in_pattern(question)

        # 2) validate refs (remark-level + document-level)
        valid_refs_siglum, invalid_refs_siglum = self._extract_and_validate_refs(referenceList)
        self.last_valid_refs = valid_refs_siglum
        self.last_invalid_refs = invalid_refs_siglum
        
        # --- HARD STOP: all refs invalid ---
        if referenceList and not valid_refs_siglum:
            self.last_document_level_refs = []
            invalid_disp = ", ".join(invalid_refs_siglum) if invalid_refs_siglum else "the provided reference(s)"

            return {
                "session_id": session_id,
                "input_question": question,
                "processed_question": processed_q,
                "referenceList_raw": referenceList,
                "valid_refs_df": valid_refs_siglum,
                "invalid_refs_df": invalid_refs_siglum,
                "document_level_refs": [],
                "referenceList_for_llm1": [],
                "filter_json": {},
                "llm1_drift_refs": [],
                "date_normalization_mode": "none",
                "filter_expression": None,
                "retrieval_query": "",
                "retrieval_query_fallback_used": False,
                "k": k,
                "docs": [],
                "docs_siglum": [],
                "answer": f"I could not find the reference(s): {invalid_disp}. Please check the identifier.",
                "sources_raw": [],
                "sources": ["No source"],
                "metadata_conflict": False,
                "metadata_conflict_message": "",
                "metadata_conflict_details": {},
            }

        # --- SOFT STOP: mixed valid + invalid refs ---
        if valid_refs_siglum and invalid_refs_siglum:
            self.last_document_level_refs = [
                r for r in valid_refs_siglum
                if r in self.known_siglum_prefixes and r not in self.known_sigla
            ]

            valid_disp = ", ".join(valid_refs_siglum)
            invalid_disp = ", ".join(invalid_refs_siglum)

            return {
                "session_id": session_id,
                "input_question": question,
                "processed_question": processed_q,
                "referenceList_raw": referenceList,
                "valid_refs_df": valid_refs_siglum,
                "invalid_refs_df": invalid_refs_siglum,
                "document_level_refs": self.last_document_level_refs,
                "referenceList_for_llm1": [re.escape(s) for s in valid_refs_siglum],
                "filter_json": {},
                "llm1_drift_refs": [],
                "date_normalization_mode": "none",
                "filter_expression": None,
                "retrieval_query": "",
                "retrieval_query_fallback_used": False,
                "k": k,
                "docs": [],
                "docs_siglum": [],
                "answer": (
                    f"I found the following valid reference(s): {valid_disp}. "
                    f"But I could not find: {invalid_disp}. "
                    f"Please correct the invalid reference(s) if you want a reliable comparison or joint answer."
                ),
                "sources_raw": [],
                "sources": ["No source"],
                "metadata_conflict": False,
                "metadata_conflict_message": "",
                "metadata_conflict_details": {},
            }

        document_level_refs = [
            r for r in valid_refs_siglum
            if r in self.known_siglum_prefixes and r not in self.known_sigla
        ]
        self.last_document_level_refs = document_level_refs

        referenceList_for_llm1 = [re.escape(s) for s in valid_refs_siglum]

        # 3) LLM1 extraction
        llm1_out = self.filter_chain.invoke({"referenceList": referenceList_for_llm1, "user_query": processed_q})

        # 3b) HARD GUARDRAIL: enforce listrefs from preprocess/DF validation (no drift)
        if valid_refs_siglum:
            llm1_out["listrefs"] = [re.escape(r) for r in valid_refs_siglum]

        # 4) patch dates deterministically if needed
        llm1_out = self._patch_llm1_dates(llm1_out, question)

        # 5) drift check (warn-only)
        drift_refs: List[str] = []
        if valid_refs_siglum:
            allowed = set(valid_refs_siglum)
            got = [_unescape_backslashes(x) for x in (llm1_out.get("listrefs") or [])]
            drift_refs = [x for x in got if x not in allowed]

        # 6) metadata -> filter
        filter_expression = self.get_filter_from_df(llm1_out)

        # 6c) OPTIONAL: doc-level k-boost ONLY for overview intent
        is_ref_only = self._is_ref_only_query(question, valid_refs_siglum)
        if self._should_boost_k_for_doc_query(question, document_level_refs, is_ref_only):
            k = max(k, 25)

        # 6b) metadata conflict: refs + dates but empty pool
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
                    "retrieval_query_fallback_used": False,
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

        # 7) Build retrieval_query (embedder input) with final guarantees
        has_refs = bool(llm1_out.get("listrefs"))
        has_dates = bool(llm1_out.get("listdates"))

        # NO-METADATA invariance: do not use LLM paraphrase
        if (not has_refs) and (not has_dates):
            retrieval_query = self._cleanup_semantic_query(processed_q)
            retrieval_query_fallback_used = False
        else:
            # start from LLM1 rewrite if present, else fallback to processed_q
            candidate = (llm1_out.get("modified_user_query_semantic_final") or "").strip()
            if not candidate:
                candidate = (llm1_out.get("modified_user_query") or processed_q).strip()

            # deterministic stripping of refs from the candidate
            candidate = self._strip_listrefs_from_text(
                candidate,
                listrefs_escaped=(llm1_out.get("listrefs") or []),
                valid_refs_siglum=valid_refs_siglum,
            )

            # deterministic stripping of dates/years if dates exist
            if has_dates:
                candidate = self._strip_dates_scaffolding(candidate)

            # LAST-MILE: remove any remaining sigla tokens regardless of upstream success
            candidate = self._strip_sigla_tokens(candidate)

            retrieval_query = candidate
            retrieval_query_fallback_used = False

            # If too short, fall back conservatively to processed_q stripped the same way
            if len(retrieval_query) < self.retrieval_min_query_chars:
                fb = processed_q
                fb = self._strip_listrefs_from_text(
                    fb,
                    listrefs_escaped=(llm1_out.get("listrefs") or []),
                    valid_refs_siglum=valid_refs_siglum,
                )
                if has_dates:
                    fb = self._strip_dates_scaffolding(fb)
                fb = self._strip_sigla_tokens(fb)
                retrieval_query = fb
                retrieval_query_fallback_used = True

        # Optional safety: avoid empty embedding input on ref-only queries (stabilizes hybrid call)
        if (not retrieval_query.strip()) and is_ref_only:
            retrieval_query = "context"
            retrieval_query_fallback_used = True

        # 8) retrieval
        docs = self.azure_search.similarity_search(
            query=retrieval_query,
            k=k,
            filters=filter_expression,
            vector_filter_mode="preFilter",
            search_type="hybrid",
        )

        if self.debug:
            print("[DEBUG] Index:", os.getenv("AZURE_AI_SEARCH_INDEX_NAME"))
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
            print("[DEBUG] Retrieval query:", retrieval_query)
            print("[DEBUG] Retrieval fallback used:", retrieval_query_fallback_used)
            print("[DEBUG] Retrieved docs:", len(docs), "(k requested:", k, ")")
            print("[DEBUG] Top 5 retrieved sigla:", [d.metadata.get("siglum") for d in docs[:5]])

        docs_siglumList = [doc.metadata.get("siglum", "") for doc in docs]

        # 9) build context string
        docs_content = "\n\n".join(
            f"{{ source: {doc.metadata.get('siglum','')}, period: {doc.metadata.get('datefrom','')} to {doc.metadata.get('dateto','')}, context: {doc.page_content}}} \n\n"
            for doc in docs
        )

        # 10) LLM2 answer (with deterministic ref-only routing)
        question_for_llm2 = processed_q
        if is_ref_only:
            question_for_llm2 = self._build_llm2_question_for_ref_only(
                original_question=question,
                valid_refs_siglum=valid_refs_siglum,
            )

        # Language control:
        # - REF-ONLY -> always English
        # - otherwise -> language of user query (heuristic)
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

        # 11) sources post-processing
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

            # remove duplicates while preserving order
            kval_exact = list(dict.fromkeys(kval_exact))
            kval_rejected = list(dict.fromkeys(kval_rejected))

            if kval_exact:
                meta_source_disp = {
                    "exact_siglum": kval_exact,
                    "rejected_siglum": kval_rejected,
                }
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
            "retrieval_query_fallback_used": retrieval_query_fallback_used,
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

        print("FILTER EXPRESSION (OData):")
        fe = out.get("filter_expression", None)
        print(fe)
        if fe:
            print("Filter chars:", len(fe))
        print("-" * 80)

        print("RETRIEVAL QUERY (embedder input):")
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
                snippet = (d.page_content or "").replace("\n", " ")
                if len(snippet) > 220:
                    snippet = snippet[:220] + "..."
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
            "assets-json/DF-wittgenstein-nonNAComma_FULL.json"
        ),
        debug=True,
    )
    out = bot.ask(q, k=10)
    bot.pretty_print(out)
