"""3-agent pipeline: Router → Analyst → Verifier."""
import contextlib, io, json, logging, os, re, textwrap
from typing import Callable
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Variable names exposed to generated code (LLM uses these names)
_DF_VARS = {
    "population_2017-2022.csv": "df_population",
    "finance_2017-2022.csv": "df_finance",
    "labor_2017-2022.csv": "df_labor",
}

DATA_DIR = os.environ.get(
    "TABLE_QA_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "data"),
)
SCHEMA_PATH = os.path.join(DATA_DIR, "schema.md")
TABLE_FILES = ["population_2017-2022.csv", "finance_2017-2022.csv", "labor_2017-2022.csv"]
_HEADER_SENTINEL = "調査年 コード"


def _load_df(filename: str) -> pd.DataFrame:
    """Read CSV, skip preamble rows, return DataFrame."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith(f'"{_HEADER_SENTINEL}"'))
    return pd.read_csv(path, skiprows=header_idx, encoding="utf-8")


def _parse_json(text: str, fallback: dict) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group()) if m else fallback
    except json.JSONDecodeError:
        return fallback


# EVOLVE-BLOCK-START
# NOTE: Available imports at module top: contextlib, io, json, logging, os, re, textwrap, pd, Callable.
# NOTE: Do NOT use typing.Dict/List/Tuple etc. Use built-in dict/list/tuple or add imports explicitly.
# NOTE: _exec_python uses exec(code, local_vars, local_vars). Helper functions defined in
#       exec'd code are visible to list comprehensions and nested scopes.
# NOTE: Prompt templates use string concatenation with parentheses, NOT triple-quoted strings.
#       This prevents unterminated string literal errors when editing prompts.
# NOTE: Each prompt section is delimited by "# ── ... ──" comment lines.
#       When inserting new code or constants, place them AFTER the section comment
#       and BEFORE the next section comment. All prompt constants are module-level
#       (zero indentation). Do NOT indent them.

# ── Router prompt ────────────────────────────────────────────────────────────
ROUTER_SYSTEM = "あなたは日本の地方統計データの質問を分析し、必要なテーブルと操作を特定する専門家です。"

ROUTER_TEMPLATE = (
    "以下のスキーマ情報があります。\n\n"
    "{schema}\n\n"
    "質問: {question}\n\n"
    "この質問に答えるために必要な情報をJSONで返してください。\n"
    '{{"tables": ["使うテーブルのファイル名リスト"], '
    '"operations": ["必要な操作 (集計/比較/時系列/JOIN等)"], '
    '"confidence": 0.0-1.0}}'
)

# ── Analyst prompt ───────────────────────────────────────────────────────────
ANALYST_SYSTEM = "あなたは日本の地方統計データの分析専門家です。正確なPythonコードを書いて質問に回答します。"

ANALYST_TEMPLATE = (
    "【スキーマ】\n{schema}\n\n"
    "【サンプル（先頭3行）】\n{sample_block}\n\n"
    "{retry_note}"
    "以下の変数はメモリ上にロード済みです。pd.read_csv() は使わないでください。\n"
    "{var_info}\n\n"
    "注意事項:\n"
    "- 都道府県比較の際は全国レコード（地域コードが '00000' の行）を除外すること\n"
    "- 単位に注意（千円、万人など。必要に応じて変換すること）\n"
    "- 年次データは質問が指定する年の列を正確に選ぶこと\n\n"
    "質問: {question}\n\n"
    "出力ルール:\n"
    "- 都道府県名は正式名称で出力（例: 東京都、大阪府、北海道）\n"
    "- 数値は適切な単位とともに明確に出力\n"
    "- ランキングは順位を明記して出力\n\n"
    "答えを print() で出力する Python コードを ```python ブロックで書いてください。"
)

# ── Verifier prompt ──────────────────────────────────────────────────────────
VERIFIER_SYSTEM = "あなたは統計データの検証専門家です。回答の正確性を厳密にチェックします。"

VERIFIER_TEMPLATE = (
    "質問: {question}\n\n"
    "回答: {answer}\n\n"
    "この回答を検証してください。チェック項目:\n"
    "- 質問に正確に答えているか\n"
    "- 単位は正しいか（千円/万人/倍 等）\n"
    "- 母数・分子の取り違えはないか\n"
    "- テーブル結合・時点の取り方は妥当か\n"
    "- 全国レコード（地域コード 00000）の除外が必要な場合にされているか\n"
    "- 都道府県名が正式名称で出力されているか\n\n"
    '結果をJSONで: {{"ok": true/false, "issues": ["問題点リスト"], "verdict": "コメント"}}'
)


class Agent:
    def __init__(self, query_llm: Callable, temperature: float = 0.0):
        self.query_llm = query_llm
        # Evolvable parameters
        self.router_temperature = 0.0
        self.analyst_temperature = 0.0
        self.verifier_temperature = 0.0
        self.max_retries = 2
        self.exclude_national_records = True
        self._dfs = {f: _load_df(f) for f in TABLE_FILES}
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            self._schema = f.read()

    def _get_df(self, table: str) -> pd.DataFrame:
        """Return DataFrame, optionally excluding national records."""
        df = self._dfs[table]
        if self.exclude_national_records and "地域コード" in df.columns:
            df = df[df["地域コード"] != "00000"]
        return df

    # ── Stage 1: Router ──────────────────────────────────────────────────────
    def _router(self, question: str) -> tuple[dict, float]:
        prompt = ROUTER_TEMPLATE.format(schema=self._schema, question=question)
        resp, cost = self.query_llm(
            prompt=prompt, system=ROUTER_SYSTEM, temperature=self.router_temperature,
        )
        plan = _parse_json(resp, {"tables": TABLE_FILES, "operations": [], "confidence": 0.5})
        return plan, cost

    # ── Stage 2: Analyst ─────────────────────────────────────────────────────
    def _exec_python(self, code: str, tables: list[str]) -> str:
        """Execute LLM-generated code with DataFrames in scope. Returns stdout or error."""
        local_vars = {"pd": pd, "json": json, **{_DF_VARS[t]: self._get_df(t) for t in tables}}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, local_vars, local_vars)  # noqa: S102
            return buf.getvalue() or "(出力なし)"
        except Exception as e:
            return f"[実行エラー] {e}"

    def _analyst(self, question: str, tables: list[str], hint: str = "") -> tuple[str, float]:
        var_info = "\n".join(f"  {_DF_VARS[t]} : {t}" for t in tables)
        sample_block = "\n\n".join(
            f"=== {t} ===\n{self._dfs[t].head(3).to_csv(index=False)}" for t in tables
        )
        retry_note = "前回の問題点: {}\n\n".format(hint) if hint else ""
        prompt = ANALYST_TEMPLATE.format(
            schema=self._schema,
            sample_block=sample_block,
            retry_note=retry_note,
            var_info=var_info,
            question=question,
        )
        resp, cost = self.query_llm(
            prompt=prompt, system=ANALYST_SYSTEM, temperature=self.analyst_temperature,
        )

        code_match = re.search(r"```python\n(.*?)```", resp, re.DOTALL)
        if code_match:
            exec_output = self._exec_python(code_match.group(1), tables)
            if not exec_output.startswith("[実行エラー]"):
                return exec_output, cost
            resp += "\n\n【コード実行結果】\n" + exec_output

        return resp, cost

    # ── Stage 3: Verifier ────────────────────────────────────────────────────
    def _verifier(self, question: str, answer: str) -> tuple[dict, float]:
        prompt = VERIFIER_TEMPLATE.format(question=question, answer=answer)
        resp, cost = self.query_llm(
            prompt=prompt, system=VERIFIER_SYSTEM, temperature=self.verifier_temperature,
        )
        result = _parse_json(resp, {"ok": True, "issues": [], "verdict": ""})
        return result, cost

    # ── Orchestration ─────────────────────────────────────────────────────────
    def forward(self, question: str) -> tuple[str, float]:
        total_cost = 0.0

        plan, cost = self._router(question)
        total_cost += cost
        tables = [t for t in plan.get("tables", TABLE_FILES) if t in self._dfs] or TABLE_FILES

        answer, cost = self._analyst(question, tables)
        total_cost += cost

        for i in range(self.max_retries):
            verdict, cost = self._verifier(question, answer)
            total_cost += cost
            if verdict.get("ok"):
                break
            issues = "; ".join(verdict.get("issues", []))
            answer, cost = self._analyst(question, tables, hint=issues)
            total_cost += cost

        return answer, total_cost

# EVOLVE-BLOCK-END


def run(question: str = "このデータを分析してください。", *, query_llm_fn=None, **kwargs):
    from utils import create_call_limited_query_llm

    if query_llm_fn is None:
        from utils import query_llm as _qlm
        query_llm_fn = _qlm

    max_calls = kwargs.get("max_calls", 10)

    limited_query_llm = create_call_limited_query_llm(query_llm_fn, max_calls)

    agent = Agent(query_llm=limited_query_llm)
    response, cost = agent.forward(question)
    print(response)
    print(f"\n[cost: ${cost:.4f}]")
    return response, cost


if __name__ == "__main__":
    run()
