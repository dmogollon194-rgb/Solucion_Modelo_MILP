import streamlit as st
import pandas as pd
import numpy as np
import itertools
import io
import ast
import re
from typing import Any
import pyomo.environ as pyo

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="MILP General", layout="wide")

# ============================================================
# STYLES + WATERMARK
# ============================================================
st.markdown("""
<style>
.watermark{position:fixed;top:150px;right:25px;opacity:.95;font-size:22px;font-weight:900;
    color:#ff4b4b;text-shadow:1px 1px 2px #000;z-index:2000}
.stApp{background:linear-gradient(180deg,#07101f 0%,#050b16 100%);color:#f3f7ff}
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1400px}
h1,h2,h3,h4,h5,h6,p,label,div,span{color:#f3f7ff}
.top-hero{background:linear-gradient(135deg,rgba(8,22,55,.95),rgba(3,10,28,.98));
    border:1px solid rgba(61,132,255,.22);border-radius:22px;padding:22px 26px;
    margin-bottom:16px;box-shadow:0 0 0 1px rgba(61,132,255,.06),0 10px 35px rgba(0,0,0,.35)}
.top-hero h2{margin:0 0 8px;font-size:1.55rem;font-weight:800;color:#fff}
.top-hero p{margin:0;font-size:1rem;color:#d7e6ff}
.kpi-card{background:linear-gradient(135deg,rgba(8,22,55,.95),rgba(3,10,28,.98));
    border:1px solid rgba(61,132,255,.22);border-radius:18px;padding:16px 18px;
    min-height:120px;display:flex;flex-direction:column;justify-content:center;
    box-shadow:0 0 0 1px rgba(61,132,255,.05),0 10px 28px rgba(0,0,0,.28);margin-bottom:10px}
.kpi-title{font-size:1.02rem;font-weight:700;color:#fff;margin-bottom:8px}
.kpi-value{font-size:2.25rem;font-weight:800;color:#fff;line-height:1.05;word-break:break-word}
.section-box{background:rgba(5,12,28,.78);border:1px solid rgba(61,132,255,.16);
    border-radius:18px;padding:18px 18px 14px}
.help-tip{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;
    margin-left:8px;border-radius:50%;border:1px solid rgba(150,190,255,.55);color:#cfe0ff;
    font-size:.78rem;font-weight:800;cursor:help;vertical-align:middle}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#08101f 0%,#050b16 100%);
    border-right:1px solid rgba(61,132,255,.18)}
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,#0c2b69,#0a1f49);
    color:#fff;border:1px solid rgba(100,162,255,.45);border-radius:12px;
    font-weight:700;padding:.6rem 1rem}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"]>div,
.stMultiSelect div[data-baseweb="select"]>div,.stTextArea textarea{
    background-color:rgba(7,16,35,.92)!important;color:#fff!important;border-radius:12px!important}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{background:rgba(8,22,55,.72);border-radius:12px 12px 0 0;
    padding:10px 18px;color:#dfeaff;font-weight:700}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#12306f,#0b1f4a);color:#fff!important}
hr{border-color:rgba(61,132,255,.18)}
</style>
<div class="watermark">by M.Sc. Dilan Mogollón</div>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
_EMPTY_SPEC = {
    "sets": {}, "parameters": {}, "decisions": {},
    "objective": None, "constraints": [], "results": None,
}

APP_SCHEMA_VERSION = 4

def _init():
    if st.session_state.get("app_schema_version") != APP_SCHEMA_VERSION:
        st.session_state.clear()
        st.session_state["app_schema_version"] = APP_SCHEMA_VERSION
    if "model_spec" not in st.session_state:
        st.session_state["model_spec"] = _EMPTY_SPEC.copy()
    if "constraint_family_expander_open" not in st.session_state:
        st.session_state["constraint_family_expander_open"] = None
    if "parameter_expander_open" not in st.session_state:
        st.session_state["parameter_expander_open"] = None
    if "objective_term_expander_open" not in st.session_state:
        st.session_state["objective_term_expander_open"] = 0

_init()
spec = st.session_state["model_spec"]

# ============================================================
# UTILITIES — SYMBOLS & SETS
# ============================================================
def valid_sym(name: str) -> bool:
    name = (name or "").strip()
    return bool(name) and (name[0].isalpha() or name[0] == "_") and all(c.isalnum() or c == "_" for c in name[1:])

def set_elements(size: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, size + 1)]

def combos(set_names: list[str], set_specs: dict) -> list[tuple]:
    if not set_names:
        return [tuple()]
    return list(itertools.product(*[set_specs[n]["elements"] for n in set_names]))

def total_elems(set_names: list[str], set_specs: dict) -> int:
    n = 1
    for k in set_names:
        n *= set_specs[k]["size"]
    return n if set_names else 1

def sig(name: str, sets_used: list[str]) -> str:
    return f"{name}[{', '.join(sets_used)}]" if sets_used else name

# ============================================================
# UTILITIES — VALUES SERIALIZATION
# ============================================================
def scalar_get(vals: dict, default=0.0) -> float:
    return float(vals.get("__scalar__", default))

def scalar_set(v: float) -> dict:
    return {"__scalar__": float(v)}

def df_to_vals(df: pd.DataFrame, set_names: list[str]) -> dict:
    out = {}
    if len(set_names) == 1:
        for _, row in df.iterrows():
            out[str((row["label"],))] = float(row["value"])
    else:
        for _, row in df.iterrows():
            key = tuple(str(row[i]) for i in set_names)
            out[str(key)] = float(row["value"])
    return out

def vals_to_df(set_names: list[str], combo_list: list[tuple], vals: dict) -> pd.DataFrame:
    rows = []
    for c in combo_list:
        row = {n: c[i] for i, n in enumerate(set_names)}
        row["value"] = float(vals.get(str(c), 0.0))
        rows.append(row)
    return pd.DataFrame(rows)

def vals_to_df1d(labels: list[str], vals: dict) -> pd.DataFrame:
    return pd.DataFrame([{"label": l, "value": float(vals.get(str((l,)), 0.0))} for l in labels])

def rand_vals(combo_list: list[tuple], lo: float, hi: float, integer: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    fn = (lambda: int(rng.integers(int(lo), int(hi) + 1))) if integer else (lambda: float(rng.uniform(lo, hi)))
    return {str(c): float(fn()) for c in combo_list}

def rand_scalar(lo: float, hi: float, integer: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    v = int(rng.integers(int(lo), int(hi) + 1)) if integer else float(rng.uniform(lo, hi))
    return {"__scalar__": float(v)}


# ============================================================
# UTILITIES — PARAMETER PERSISTENCE + EXCEL/CSV
# ============================================================
def _param_store_key(row_pos: int) -> str:
    return f"_param_values_store_{row_pos}"


def _param_signature_key(pname: str, set_names: list[str]) -> str:
    return f"{pname}__{'_'.join(set_names) if set_names else 'scalar'}"


def _empty_values_for_parameter(set_names: list[str], set_specs: dict) -> dict:
    if not set_names:
        return {"__scalar__": 0.0}
    return {str(c): 0.0 for c in combos(set_names, set_specs)}


def _values_match_structure(values: dict, set_names: list[str], set_specs: dict) -> bool:
    if not isinstance(values, dict):
        return False
    if not set_names:
        return "__scalar__" in values
    expected = {str(c) for c in combos(set_names, set_specs)}
    return expected.issubset(set(values.keys()))


def _initial_param_values(row_pos: int, pname: str, set_names: list[str], set_specs: dict, old_vals: dict) -> dict:
    """Return persistent values compatible with the current parameter structure."""
    store_key = _param_store_key(row_pos)

    # 1) Priority: live values from the editor in session_state.
    stored = st.session_state.get(store_key)
    if _values_match_structure(stored, set_names, set_specs):
        return dict(stored)

    # 2) Then: values saved in the previous spec.
    if _values_match_structure(old_vals, set_names, set_specs):
        st.session_state[store_key] = dict(old_vals)
        return dict(old_vals)

    # 3) If dimensionality changed, initialize zeros with the new structure.
    fresh = _empty_values_for_parameter(set_names, set_specs)
    st.session_state[store_key] = dict(fresh)
    return fresh


def _set_param_values(row_pos: int, values: dict) -> dict:
    st.session_state[_param_store_key(row_pos)] = dict(values)
    return dict(values)


def template_df_for_parameter(set_names: list[str], set_specs: dict, current_values: dict) -> pd.DataFrame:
    if not set_names:
        return pd.DataFrame([{"value": scalar_get(current_values, 0.0)}])
    if len(set_names) == 1:
        return vals_to_df1d(set_specs[set_names[0]]["elements"], current_values)
    return vals_to_df(set_names, combos(set_names, set_specs), current_values)


def dataframe_to_csv_bytes(df: pd.DataFrame, header: bool = True) -> bytes:
    return df.to_csv(index=False, header=header).encode("utf-8-sig")


def dataframe_to_xlsx_bytes(df: pd.DataFrame, header: bool = True) -> bytes | None:
    """Create XLSX only when xlsxwriter is installed; otherwise return None without breaking the app."""
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, header=header, sheet_name="parameter")
        return buffer.getvalue()
    except Exception:
        return None


def parameter_values_only_df(set_names: list[str], set_specs: dict, current_values: dict) -> pd.DataFrame:
    """
    Build a headerless values-only template.

    The upload format intentionally contains no row labels, column headers,
    set names, or other text. Values are assigned to parameter components
    in the same order used by `combos(...)`.
    """
    if not set_names:
        values = [scalar_get(current_values, 0.0)]
    else:
        values = [float(current_values.get(str(c), 0.0)) for c in combos(set_names, set_specs)]
    return pd.DataFrame(values)


def read_parameter_upload(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """
    Read a headerless numeric CSV/Excel file.

    CSV supports comma, semicolon, or tab separators. A parameter may be
    supplied horizontally or vertically; validation later flattens values
    from left to right and then top to bottom.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            raw = uploaded_file.getvalue()
            if not raw:
                return None, "The file is empty."

            # UTF-8 with BOM is common when exporting from Excel.
            try:
                content = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                content = raw.decode("latin-1")

            if not content.strip():
                return None, "The file is empty."

            # Detect the most common practical separators explicitly.
            # This avoids pandas interpreting an entire semicolon row as a header.
            sample = content[:4096]
            counts = {sep: sample.count(sep) for sep in (";", ",", "\\t")}
            sep = max(counts, key=counts.get) if max(counts.values()) > 0 else None

            if sep is None:
                return pd.read_csv(io.StringIO(content), header=None), None
            return pd.read_csv(io.StringIO(content), sep=sep, header=None), None

        if name.endswith((".xlsx", ".xls")):
            try:
                return pd.read_excel(uploaded_file, header=None), None
            except ImportError:
                return None, "To load .xlsx files, install `openpyxl` or use CSV."
            except ModuleNotFoundError:
                return None, "To load .xlsx files, install `openpyxl` or use CSV."

        return None, "Unsupported format. Use .csv, .xlsx, or .xls."
    except Exception as exc:
        return None, f"The file could not be read: {exc}"


def validate_and_convert_parameter_df(
    df: pd.DataFrame,
    set_names: list[str],
    set_specs: dict
) -> tuple[dict | None, list[str]]:
    """
    Validate a values-only parameter file.

    Rules:
      * no headers or row labels;
      * numeric values only;
      * the exact expected number of values;
      * values are read left-to-right, top-to-bottom.
    """
    if df is None or df.empty:
        return None, ["The file is empty."]

    work = df.copy()

    # Remove rows/columns that are completely empty, but do not silently
    # ignore holes inside the actual data region.
    work = work.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if work.empty:
        return None, ["The file is empty."]

    numeric = work.apply(pd.to_numeric, errors="coerce")

    # Any non-empty cell that cannot be converted to a number indicates
    # a header, row label, set name, or other invalid text.
    invalid_mask = work.notna() & numeric.isna()
    if invalid_mask.any().any():
        bad = []
        for r, c in zip(*np.where(invalid_mask.to_numpy())):
            bad.append(str(work.iloc[r, c]))
            if len(bad) >= 5:
                break
        shown = ", ".join(repr(x) for x in bad)
        return None, [
            "The upload must contain numeric values only — no column headers, "
            "row labels, set names, or text."
            + (f" Invalid cell(s): {shown}." if shown else "")
        ]

    if numeric.isna().any().any():
        return None, [
            "Empty cells are not allowed inside the data range. "
            "Provide one numeric value for every parameter element."
        ]

    flat_values = numeric.to_numpy(dtype=float).ravel(order="C").tolist()
    expected = total_elems(set_names, set_specs)

    if len(flat_values) != expected:
        signature = sig("parameter", set_names)
        return None, [
            f"Expected exactly {expected} numeric value(s), but the file contains "
            f"{len(flat_values)}. Do not include headers or row/column labels."
        ]

    if not set_names:
        return {"__scalar__": float(flat_values[0])}, []

    values = {}
    for combo, value in zip(combos(set_names, set_specs), flat_values):
        values[str(combo)] = float(value)
    return values, []


def parameter_template_controls(
    row_pos: int,
    pname: str,
    set_names: list[str],
    set_specs: dict,
    current_values: dict
):
    # The downloadable file contains values only; the labelled preview shown
    # in the app remains available so users can see the assignment order.
    df_template_raw = parameter_values_only_df(set_names, set_specs, current_values)
    df_preview = template_df_for_parameter(set_names, set_specs, current_values)
    file_base = f"template_{pname}"
    widget_suffix = _param_signature_key(pname, set_names)

    st.info(
        "**File format instructions**\\n\\n"
        "- Enter **numeric values only**.\\n"
        "- **Do not include column headers, row titles, set-element labels, or set names.**\\n"
        "- CSV files may use a **comma (`,`) or semicolon (`;`)** as separator.\\n"
        "- Values may be arranged in **one row or one column**. They are read "
        "**left to right, then top to bottom**.\\n"
        f"- This parameter requires exactly **{total_elems(set_names, set_specs)} value(s)**."
    )

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download CSV template",
            data=dataframe_to_csv_bytes(df_template_raw, header=False),
            file_name=f"{file_base}.csv",
            mime="text/csv",
            key=f"tmpl_csv_{row_pos}_{widget_suffix}",
            help="Downloads a values-only CSV file with no headers or row labels.",
        )
    with dl2:
        xlsx_bytes = dataframe_to_xlsx_bytes(df_template_raw, header=False)
        if xlsx_bytes is not None:
            st.download_button(
                "Download Excel template",
                data=xlsx_bytes,
                file_name=f"{file_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"tmpl_xlsx_{row_pos}_{widget_suffix}",
                help="Downloads a values-only Excel file with no headers or row labels.",
            )
        else:
            st.info(
                "Install `xlsxwriter` to download .xlsx files. "
                "The CSV template can still be opened in Excel."
            )

    uploaded = st.file_uploader(
        f"Upload values for {pname}",
        type=["csv", "xlsx", "xls"],
        key=f"upload_param_{row_pos}_{widget_suffix}",
        help=(
            "Upload numeric values only. Do not add headers, row labels, "
            "set names, or other text."
        ),
    )

    if uploaded is None:
        st.caption("Assignment order used by the model:")
        st.dataframe(df_preview, use_container_width=True, hide_index=True)
        return current_values

    df_uploaded, read_error = read_parameter_upload(uploaded)
    if read_error:
        st.error(read_error)
        st.caption("Assignment order used by the model:")
        st.dataframe(df_preview, use_container_width=True, hide_index=True)
        return current_values

    values, errors = validate_and_convert_parameter_df(df_uploaded, set_names, set_specs)
    if errors:
        for err in errors:
            st.error(err)
        st.write("Uploaded file preview:")
        st.dataframe(df_uploaded, use_container_width=True, hide_index=True)
        st.caption("Expected assignment order:")
        st.dataframe(df_preview, use_container_width=True, hide_index=True)
        return current_values

    st.success(
        f"Values loaded successfully: {total_elems(set_names, set_specs)} value(s) assigned to {sig(pname, set_names)}."
    )
    _set_param_values(row_pos, values)
    st.caption("Parameter preview:")
    st.dataframe(
        template_df_for_parameter(set_names, set_specs, values),
        use_container_width=True,
        hide_index=True,
    )
    return values

# ============================================================
# UTILITIES — EXPRESSIONS + DYNAMIC SUMMATION BOUNDS
# ============================================================
DOMAIN_LABELS = {"Binary": "Binary", "NonNegativeReals": "Nonnegative reals", "NonNegativeIntegers": "Nonnegative integers"}

def _term_sums(t: dict) -> list[dict]:
    """Return the new summation structure while remaining compatible with old models."""
    if "sums" in t:
        return t.get("sums", [])
    return [{"set": set_name, "lower": "1", "upper": f"N_{set_name}"} for set_name in t.get("sum_over", [])]

def _expr_names(expr: str) -> set[str]:
    try:
        tree = ast.parse((expr or "").strip(), mode="eval")
    except Exception:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

def _safe_bound_eval(expr: str, numeric_env: dict[str, int], set_specs: dict) -> int:
    """Safely evaluate integer bound expressions such as j+2, 2*j+1, or N_i-1."""
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("A summation bound cannot be empty.")

    names = dict(numeric_env)
    names.update({f"N_{set_name}": int(data["size"]) for set_name, data in set_specs.items()})
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"Unknown symbol `{node.id}` in bound `{expr}`.")
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = ev(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.FloorDiv): return a // b
            if isinstance(node.op, ast.Mod): return a % b
            if isinstance(node.op, ast.Pow): return a ** b
        raise ValueError(f"Unsupported expression in bound `{expr}`.")

    value = ev(tree)
    rounded = round(float(value))
    if abs(float(value) - rounded) > 1e-9:
        raise ValueError(f"Bound `{expr}` evaluates to {value}, but set positions must be integers.")
    return int(rounded)

def _validate_bound_expression(expr: str, set_names: list[str], current_sum_set: str | None = None) -> list[str]:
    errors = []
    try:
        tree = ast.parse((expr or "").strip(), mode="eval")
    except Exception:
        return [f"Invalid bound expression `{expr}`."]

    allowed_names = set(set_names) | {f"N_{set_name}" for set_name in set_names}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            errors.append(f"Functions are not allowed in bound `{expr}`.")
        elif isinstance(node, ast.Name) and node.id not in allowed_names:
            errors.append(f"Unknown symbol `{node.id}` in bound `{expr}`.")
        elif isinstance(node, (ast.Attribute, ast.Subscript, ast.List, ast.Dict, ast.Set, ast.Tuple)):
            errors.append(f"Unsupported syntax in bound `{expr}`.")
    if current_sum_set and current_sum_set in _expr_names(expr):
        errors.append(f"Bound `{expr}` cannot depend on its own summation set `{current_sum_set}`.")
    return list(dict.fromkeys(errors))

def _set_position(set_name: str, value: str, set_specs: dict) -> int:
    elements = set_specs[set_name]["elements"]
    try:
        return elements.index(value) + 1
    except ValueError as exc:
        raise ValueError(f"Value `{value}` does not belong to set `{set_name}`.") from exc

def _bound_env(env: dict, set_specs: dict) -> dict[str, int]:
    return {set_name: _set_position(set_name, value, set_specs) for set_name, value in env.items() if set_name in set_specs}

def _sum_values(sum_spec: dict, env: dict, set_specs: dict) -> list[str]:
    set_name = sum_spec["set"]
    numeric_env = _bound_env(env, set_specs)
    lower = _safe_bound_eval(sum_spec.get("lower", "1"), numeric_env, set_specs)
    upper = _safe_bound_eval(sum_spec.get("upper", f"N_{set_name}"), numeric_env, set_specs)
    elements = set_specs[set_name]["elements"]
    if lower > upper:
        return []
    # Intersect the requested range with the valid positions of the set.
    start = max(1, lower)
    end = min(len(elements), upper)
    if start > end:
        return []
    return elements[start - 1:end]

def _fac_latex(f: dict) -> str:
    if f["type"] == "constant":
        v = float(f["value"])
        return str(int(v)) if v == int(v) else f"{v:.2f}"
    n, sets_used = f["name"], f["sets"]
    return n if not sets_used else rf"{n}_{{{','.join(sets_used)}}}"

def _factor_ops(t: dict) -> list[str]:
    """Return operators between factors. Old models default to multiplication."""
    factors = t.get("factors", [])
    expected = max(0, len(factors) - 1)
    ops = list(t.get("factor_ops", []))
    if len(ops) < expected:
        ops.extend(["*"] * (expected - len(ops)))
    return ops[:expected]

def _term_connector(t: dict, position: int) -> str:
    """
    Connector between complete terms.
    Backward compatibility:
      - old `sign` becomes + or -
      - first term only supports unary +/-
    """
    if position == 0:
        return "-" if t.get("connector", t.get("sign", "+")) == "-" else "+"
    op = t.get("connector")
    if op in {"+", "-", "*", "/"}:
        return op
    return "-" if t.get("sign", "+") == "-" else "+"

def _factors_latex(t: dict) -> str:
    factors = t.get("factors", [])
    if not factors:
        return "0"

    body = _fac_latex(factors[0])
    for op, factor in zip(_factor_ops(t), factors[1:]):
        rhs = _fac_latex(factor)
        if op == "/":
            body = rf"\frac{{{body}}}{{{rhs}}}"
        else:
            body = rf"{body} \cdot {rhs}"
    return body

def term_body_latex(t: dict) -> str:
    """LaTeX body for one complete term, without the connector to other terms."""
    body = _factors_latex(t)
    for s in reversed(_term_sums(t)):
        set_name = s["set"]
        lower = s.get("lower", "1")
        upper = s.get("upper", f"N_{set_name}")
        body = rf"\sum_{{{set_name}={lower}}}^{{{upper}}}\left({body}\right)"
    return body

def term_latex(t: dict) -> str:
    """
    Backward-compatible standalone term preview.
    Uses connector/sign as a unary sign only when it is + or -.
    """
    body = term_body_latex(t)
    op = t.get("connector", t.get("sign", "+"))
    return f"- {body}" if op == "-" else f"+ {body}"

def expr_latex(terms: list[dict]) -> str:
    if not terms:
        return "0"

    out = ""
    for pos, t in enumerate(terms):
        body = term_body_latex(t)
        op = _term_connector(t, pos)

        if pos == 0:
            out = rf"- {body}" if op == "-" else body
            continue

        if op == "+":
            out = rf"{out} + {body}"
        elif op == "-":
            out = rf"{out} - {body}"
        elif op == "*":
            out = rf"\left({out}\right)\cdot\left({body}\right)"
        elif op == "/":
            out = rf"\frac{{\left({out}\right)}}{{\left({body}\right)}}"

    return out

def family_latex(fam: dict) -> str:
    sense_map = {"<=": r"\leq", ">=": r"\geq", "=": "="}
    lhs = expr_latex(fam.get("lhs_terms", []))
    rhs = expr_latex(fam.get("rhs_terms", []))
    s = sense_map.get(fam.get("sense", "<="), r"\leq")
    txt = f"{lhs} {s} {rhs}"
    if fam.get("forall"):
        txt += r"\quad \forall " + ", ".join(fam["forall"])
    return txt

def term_free_sets(t: dict) -> list[str]:
    used = []
    for f in t.get("factors", []):
        if f["type"] == "object":
            used.extend(f["sets"])

    sums = _term_sums(t)
    sum_sets = [s["set"] for s in sums]
    for s in sums:
        for expr in (s.get("lower", "1"), s.get("upper", f"N_{s['set']}")):
            used.extend(name for name in _expr_names(expr) if not name.startswith("N_"))

    seen, out = set(), []
    for x in used:
        if x not in seen:
            seen.add(x); out.append(x)
    return [x for x in out if x not in sum_sets]

# ============================================================
# UTILITIES — VALIDATION
# ============================================================
def validate_term_sums(t: dict, set_names: list[str], context: str) -> list[str]:
    errs = []
    sums = _term_sums(t)
    seen = set()
    for pos, s in enumerate(sums, 1):
        set_name = s.get("set")
        if set_name not in set_names:
            errs.append(f"{context}: summation {pos} uses an undefined set `{set_name}`.")
            continue
        if set_name in seen:
            errs.append(f"{context}: set `{set_name}` is used in more than one summation in the same term.")
        seen.add(set_name)
        errs.extend(f"{context}: {e}" for e in _validate_bound_expression(s.get("lower", "1"), set_names, set_name))
        errs.extend(f"{context}: {e}" for e in _validate_bound_expression(s.get("upper", f"N_{set_name}"), set_names, set_name))

        later_sets = {z.get("set") for z in sums[pos:] if z.get("set")}
        dependencies = (_expr_names(s.get("lower", "1")) | _expr_names(s.get("upper", f"N_{set_name}"))) & set(set_names)
        invalid_later_sets = sorted(dependencies & later_sets)
        if invalid_later_sets:
            errs.append(f"{context}: bounds for `{set_name}` cannot depend on inner/later summation sets {invalid_later_sets}.")
    return errs

def validate_obj(terms: list[dict], set_names: list[str] | None = None) -> list[str]:
    errs = []
    set_names = set_names or list(spec.get("sets", {}).keys())
    for i, t in enumerate(terms, 1):
        errs.extend(validate_term_sums(t, set_names, f"Objective term {i}"))
        free = term_free_sets(t)
        if free:
            errs.append(f"Objective term {i}: free sets without a matching summation → {', '.join(free)}")
    return errs

def validate_family(fam: dict, set_names: list[str] | None = None) -> list[str]:
    errs = []
    set_names = set_names or list(spec.get("sets", {}).keys())
    for i, t in enumerate(fam.get("lhs_terms", []), 1):
        errs.extend(validate_term_sums(t, set_names, f"Constraint {fam.get('name', '')} LHS term {i}"))
    for i, t in enumerate(fam.get("rhs_terms", []), 1):
        errs.extend(validate_term_sums(t, set_names, f"Constraint {fam.get('name', '')} RHS term {i}"))

    lhs_free = list({x for t in fam.get("lhs_terms", []) for x in term_free_sets(t)})
    rhs_free = list({x for t in fam.get("rhs_terms", []) for x in term_free_sets(t)})
    forall = fam.get("forall", [])
    lc, rc = not lhs_free, not rhs_free
    if not lc and not rc:
        if sorted(lhs_free) != sorted(rhs_free):
            errs.append(f"Free sets on LHS {lhs_free} ≠ free sets on RHS {rhs_free}")
        if sorted(lhs_free) != sorted(forall):
            errs.append(f"Free sets {lhs_free} ≠ forall sets {forall}")
    elif lc and not rc:
        if sorted(rhs_free) != sorted(forall):
            errs.append(f"Constant LHS: RHS sets {rhs_free} must match forall sets {forall}")
    elif not lc and rc:
        if sorted(lhs_free) != sorted(forall):
            errs.append(f"Constant RHS: LHS sets {lhs_free} must match forall sets {forall}")
    elif lc and rc and forall:
        errs.append(f"Both sides are constant, but forall sets were defined: {forall}")
    return errs

def _term_has_decision(t: dict) -> bool:
    return any(
        f.get("type") == "object" and f.get("kind") == "decision"
        for f in t.get("factors", [])
    )

def _term_is_parameter_only(t: dict) -> bool:
    """True when the complete term contains no decision symbol."""
    return not _term_has_decision(t)

def validate_linearity(spec: dict) -> list[str]:
    errs = []

    def chk_internal(terms, ctx):
        for i, t in enumerate(terms, 1):
            factors = t.get("factors", [])
            ops = _factor_ops(t)

            decision_count = sum(
                1 for f in factors
                if f.get("type") == "object" and f.get("kind") == "decision"
            )
            if decision_count > 1:
                errs.append(
                    f"{ctx} term {i}: {decision_count} decision symbols occur in the same factor chain "
                    "→ nonlinear expression."
                )

            for pos, (op, factor) in enumerate(zip(ops, factors[1:]), start=2):
                if op != "/":
                    continue

                if factor.get("type") == "object" and factor.get("kind") == "decision":
                    errs.append(
                        f"{ctx} term {i}: factor {pos} is a decision symbol in the denominator "
                        "→ nonlinear expression."
                    )

                if factor.get("type") == "constant" and abs(float(factor.get("value", 0.0))) < 1e-12:
                    errs.append(
                        f"{ctx} term {i}: factor {pos} is zero and cannot be used as a denominator."
                    )

                if factor.get("type") == "object" and factor.get("kind") == "parameter":
                    ps = spec.get("parameters", {}).get(factor.get("name"), {})
                    vals = ps.get("values", {})
                    if any(abs(float(v)) < 1e-12 for v in vals.values()):
                        errs.append(
                            f"{ctx} term {i}: denominator parameter `{factor.get('name')}` contains "
                            "at least one zero value."
                        )

    def chk_connectors(terms, ctx):
        """
        Term-level × or ÷ is allowed only when it preserves linearity:
        - multiplying a decision-containing accumulated expression by a parameter-only term is valid;
        - multiplying two decision-containing expressions is nonlinear;
        - division by a decision-containing term is nonlinear.
        """
        if not terms:
            return

        acc_has_decision = _term_has_decision(terms[0])

        for pos, t in enumerate(terms[1:], start=1):
            op = _term_connector(t, pos)
            rhs_has_decision = _term_has_decision(t)

            if op == "*":
                if acc_has_decision and rhs_has_decision:
                    errs.append(
                        f"{ctx} term {pos+1}: multiplying two expressions that contain decision "
                        "decision symbols is nonlinear."
                    )
                acc_has_decision = acc_has_decision or rhs_has_decision

            elif op == "/":
                if rhs_has_decision:
                    errs.append(
                        f"{ctx} term {pos+1}: division by an expression containing a decision "
                        "decision symbol is nonlinear."
                    )
                # If denominator is parameter-only, decision status of the accumulated expression is unchanged.

            else:
                acc_has_decision = acc_has_decision or rhs_has_decision

    def chk(terms, ctx):
        chk_internal(terms, ctx)
        chk_connectors(terms, ctx)

    obj = spec.get("objective")
    if obj:
        chk(obj.get("terms", []), "Objective")
    for r, fam in enumerate(spec.get("constraints", []), 1):
        name = fam.get("name", f"R{r}")
        chk(fam.get("lhs_terms", []), f"Constraint {name} LHS")
        chk(fam.get("rhs_terms", []), f"Constraint {name} RHS")
    return errs

# ============================================================
# UTILITIES — PYOMO
# ============================================================
_DOMAINS = {"Binary": pyo.Binary, "NonNegativeReals": pyo.NonNegativeReals, "NonNegativeIntegers": pyo.NonNegativeIntegers}

SOLVER_OPTIONS = {
    "HiGHS": "appsi_highs",
}


def solver_factory_from_label(label: str):
    solver_name = SOLVER_OPTIONS.get(label)
    if solver_name is None:
        raise ValueError(f"Unsupported solver: {label}")
    solver = pyo.SolverFactory(solver_name)
    try:
        available = solver.available(exception_flag=False)
    except TypeError:
        available = solver.available()
    if not available:
        raise RuntimeError(f"Solver `{solver_name}` is not available in this environment.")
    return solver_name, solver

def _get_val(model, f: dict, env: dict):
    if f["type"] == "constant":
        return float(f["value"])
    comp = getattr(model, f"{'par' if f['kind'] == 'parameter' else 'decision'}_{f['name']}")
    sets_used = f["sets"]
    if not sets_used:
        return comp
    key = tuple(env[i] for i in sets_used)
    return comp[key[0]] if len(key) == 1 else comp[key]

def _eval_factor_chain(model, t: dict, env: dict):
    factors = t.get("factors", [])
    if not factors:
        return 0

    val = _get_val(model, factors[0], env)
    for op, factor in zip(_factor_ops(t), factors[1:]):
        rhs = _get_val(model, factor, env)
        if op == "/":
            val = val / rhs
        else:
            val = val * rhs
    return val

def _eval_term_body(model, t: dict, env: dict):
    """Evaluate one complete term without applying its connector."""
    sums = _term_sums(t)
    set_specs = getattr(model, "_set_specs")

    def recurse(pos, local_env):
        if pos == len(sums):
            return _eval_factor_chain(model, t, local_env)

        sum_spec = sums[pos]
        set_name = sum_spec["set"]
        values = _sum_values(sum_spec, local_env, set_specs)
        return sum(recurse(pos + 1, {**local_env, set_name: v}) for v in values)

    return recurse(0, dict(env))

def _build_expr(model, terms: list[dict], env: dict):
    """
    Build the full expression using connectors between complete terms.
    Evaluation is left-associative for × and ÷ at the term level.
    """
    if not terms:
        return 0

    first = _eval_term_body(model, terms[0], env)
    acc = -first if _term_connector(terms[0], 0) == "-" else first

    for pos, t in enumerate(terms[1:], start=1):
        rhs = _eval_term_body(model, t, env)
        op = _term_connector(t, pos)

        if op == "+":
            acc = acc + rhs
        elif op == "-":
            acc = acc - rhs
        elif op == "*":
            acc = acc * rhs
        elif op == "/":
            acc = acc / rhs

    return acc

def build_pyomo_model(spec: dict):
    m = pyo.ConcreteModel()
    set_specs = spec["sets"]
    m._set_specs = set_specs

    for n, s in set_specs.items():
        setattr(m, f"set_{n}", pyo.Set(initialize=s["elements"], ordered=True))

    for pn, ps in spec["parameters"].items():
        sets_used, vals = ps["sets"], ps["values"]
        if not sets_used:
            setattr(m, f"par_{pn}", pyo.Param(initialize=float(vals["__scalar__"])))
        else:
            sets = [getattr(m, f"set_{i}") for i in sets_used]
            init = {}
            for c in combos(sets_used, set_specs):
                k = c[0] if len(c) == 1 else c
                init[k] = float(vals.get(str(c), 0.0))
            setattr(m, f"par_{pn}", pyo.Param(*sets, initialize=init))

    for decision_name, decision_record in spec["decisions"].items():
        sets_used = decision_record["sets"]
        dom = _DOMAINS[decision_record["domain"]]
        if not sets_used:
            setattr(m, f"decision_{decision_name}", pyo.Var(domain=dom))
        else:
            sets = [getattr(m, f"set_{i}") for i in sets_used]
            setattr(m, f"decision_{decision_name}", pyo.Var(*sets, domain=dom))

    obj = spec["objective"]
    obj_expr = _build_expr(m, obj["terms"], {})
    m.OBJ = pyo.Objective(expr=obj_expr, sense=pyo.minimize if obj["sense"] == "minimize" else pyo.maximize)

    for ci, fam in enumerate(spec.get("constraints", []), 1):
        name = fam.get("name", f"R{ci}")
        lhs_t, rhs_t = fam.get("lhs_terms", []), fam.get("rhs_terms", [])
        forall, sense = fam.get("forall", []), fam.get("sense", "<=")
        ops = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "=": lambda a, b: a == b}

        if not forall:
            con = pyo.Constraint(expr=ops[sense](_build_expr(m, lhs_t, {}), _build_expr(m, rhs_t, {})))
        else:
            sets = [getattr(m, f"set_{i}") for i in forall]
            def _rule(mdl, *args, _lhs=lhs_t, _rhs=rhs_t, _fa=forall, _s=sense):
                env = dict(zip(_fa, args))
                return ops[_s](_build_expr(mdl, _lhs, env), _build_expr(mdl, _rhs, env))
            con = pyo.Constraint(*sets, rule=_rule)
        setattr(m, f"con_{name}", con)

    return m

def decision_solution_df(model, decision_name: str, decision_spec: dict, set_specs: dict) -> pd.DataFrame:
    comp = getattr(model, f"decision_{decision_name}")
    sets_used = decision_spec["sets"]
    if not sets_used:
        return pd.DataFrame({"decision_symbol": [decision_name], "value": [pyo.value(comp)]})
    rows = []
    for c in combos(sets_used, set_specs):
        row = {set_name: c[i] for i, set_name in enumerate(sets_used)}
        row["value"] = pyo.value(comp[c[0]] if len(c) == 1 else comp[c])
        rows.append(row)
    return pd.DataFrame(rows)

def all_decisions_df(model, spec: dict) -> pd.DataFrame:
    dfs = []
    for decision_name, decision_record in spec["decisions"].items():
        df = decision_solution_df(model, decision_name, decision_record, spec["sets"])
        df.insert(0, "decision_name", decision_name)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def count_expanded(spec: dict, key: str) -> int:
    set_specs = spec.get("sets", {})
    total = 0
    items = spec.get(key, {})
    collection = items.values() if isinstance(items, dict) else items
    for item in collection:
        sets_used = item.get("sets", item.get("forall", []))
        if not sets_used:
            total += 1
        elif all(i in set_specs for i in sets_used):
            n = 1
            for i in sets_used:
                n *= set_specs[i]["size"]
            total += n
    return total

# ============================================================
# UI HELPERS
# ============================================================
def hero(title: str, text: str):
    st.markdown(f'<div class="top-hero"><h2>{title}</h2><p>{text}</p></div>', unsafe_allow_html=True)

def kpi_card(title: str, value: Any):
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

def section_box(subtitle: str, text: str = "", instructions: str = ""):
    safe_tip = (instructions or "").replace('"', '&quot;')
    tip = f'<span class="help-tip" title="{safe_tip}">?</span>' if instructions else ""
    st.markdown(
        f'<div class="section-box"><b style="font-size:1.1rem">{subtitle}</b>{tip}'
        + (f'<p style="color:#d7e6ff;margin-top:6px">{text}</p>' if text else "")
        + '</div>',
        unsafe_allow_html=True
    )

def _rand_controls(key_prefix: str) -> tuple[float, float, bool, int]:
    c1, c2, c3, c4 = st.columns(4)
    lo = c1.number_input("Minimum", value=0.0, key=f"{key_prefix}_lo", help="Smallest value allowed in the random generation range.")
    hi = c2.number_input("Maximum", value=10.0, key=f"{key_prefix}_hi", help="Largest value allowed in the random generation range.")
    integer = c3.checkbox("Integer values", value=False, key=f"{key_prefix}_int", help="Generate integer values instead of continuous values.")
    seed = int(c4.number_input("Seed", value=123, step=1, key=f"{key_prefix}_seed", help="Use the same seed to reproduce the same random values."))
    return lo, hi, integer, seed

def _objective_term_position_from_key(t_key: str) -> int | None:
    """
    Extract the objective-term position from keys such as:
      obj_t0
      obj_t1_f0
      obj_t2_f1
    Constraint keys return None.
    """
    if not t_key.startswith("obj_t"):
        return None
    tail = t_key[len("obj_t"):]
    token = tail.split("_", 1)[0]
    return int(token) if token.isdigit() else None


def _keep_objective_term_open_kwargs(t_key: str) -> dict:
    """
    Callback arguments added to widgets inside an objective term.
    When a widget changes, Streamlit reruns the script; this callback
    records which expander must be reopened.
    """
    term_position = _objective_term_position_from_key(t_key)
    if term_position is None:
        return {}
    return {
        "on_change": _open_objective_term,
        "args": (term_position,),
    }


def build_factor_ui(
    t_key: str,
    factor_position: int,
    old_factor: dict | None,
    catalog: list[dict],
    label_map: dict,
    default_type="object"
) -> dict | None:
    old_factor = old_factor or {}

    ftype = st.selectbox(
        f"Factor type {factor_position+1}",
        ["object", "constant"],
        index=0 if old_factor.get("type", default_type) == "object" else 1,
        format_func=lambda x: "Parameter / Decision Symbol" if x == "object" else "Constant",
        key=f"{t_key}_ftype_{factor_position}",
        help=(
            "Choose a parameter, decision symbol, or numeric constant. "
            "Multiplication and division are selected between consecutive factors."
        ),
        **_keep_objective_term_open_kwargs(t_key),
    )

    if ftype == "object":
        labels = [o["label"] for o in catalog]
        if not labels:
            st.error("No parameters or decision symbols are available.")
            return None

        default_lbl = labels[0]
        if old_factor.get("type") == "object" and old_factor.get("label") in labels:
            default_lbl = old_factor["label"]

        chosen = st.selectbox(
            f"Object {factor_position+1}",
            labels,
            index=labels.index(default_lbl),
            key=f"{t_key}_fobj_{factor_position}",
            help="Select the parameter or decision symbol used in this factor.",
            **_keep_objective_term_open_kwargs(t_key),
        )
        item = label_map[chosen]
        st.caption(f"Sets: {', '.join(item['sets']) or 'none'}")
        return {
            "type": "object",
            "kind": item["kind"],
            "name": item["name"],
            "sets": item["sets"],
            "label": item["label"],
        }

    dval = (
        float(old_factor.get("value", 0.0))
        if old_factor.get("type") == "constant"
        else 0.0
    )
    val = st.number_input(
        f"Constant {factor_position+1}",
        value=dval,
        key=f"{t_key}_fconst_{factor_position}",
        help="Enter the numeric value used in this factor.",
        **_keep_objective_term_open_kwargs(t_key),
    )
    return {"type": "constant", "value": float(val)}


def build_term_ui(
    t_key: str,
    term_position: int,
    old_term: dict | None,
    catalog: list[dict],
    label_map: dict,
    set_names: list[str],
    default_const_type="object"
) -> dict:
    old = old_term or {}
    old_sums = _term_sums(old)
    old_ops = _factor_ops(old)

    c1, c2, c3 = st.columns([1, 2, 2])

    if term_position == 0:
        connector_options = ["+", "-"]
        old_connector = old.get("connector", old.get("sign", "+"))
        if old_connector not in connector_options:
            old_connector = "+"
        connector = c1.selectbox(
            "Initial sign",
            connector_options,
            index=connector_options.index(old_connector),
            key=f"{t_key}_connector",
            help="The first term has no previous term, so only a positive or negative initial sign is meaningful.",
            **_keep_objective_term_open_kwargs(t_key),
        )
    else:
        connector_options = ["+", "-", "*", "/"]
        old_connector = old.get("connector", old.get("sign", "+"))
        if old_connector not in connector_options:
            old_connector = "+"
        connector = c1.selectbox(
            f"Connector {term_position+1}",
            connector_options,
            index=connector_options.index(old_connector),
            format_func=lambda x: {
                "+": "+  Add",
                "-": "−  Subtract",
                "*": "×  Multiply",
                "/": "÷  Divide",
            }[x],
            key=f"{t_key}_connector",
            help=(
                "Operation between the complete previous expression and this term. "
                "Multiplication/division are allowed only when the resulting model remains linear."
            ),
            **_keep_objective_term_open_kwargs(t_key),
        )
    n_factors = int(c2.number_input(
        f"Factors {term_position+1}",
        min_value=1,
        max_value=6,
        value=max(1, len(old.get("factors", [])) or 2),
        step=1,
        key=f"{t_key}_nfac",
        help="Number of factors connected by multiplication or division.",
        **_keep_objective_term_open_kwargs(t_key),
    ))
    n_sums = int(c3.number_input(
        f"Summations {term_position+1}",
        min_value=0,
        max_value=max(0, len(set_names)),
        value=min(len(old_sums), len(set_names)),
        step=1,
        key=f"{t_key}_nsums",
        help="Number of nested summations applied to the complete factor chain.",
        **_keep_objective_term_open_kwargs(t_key),
    ))

    st.caption(
        "The connector above operates between complete terms. "
        "The operators below operate between factors inside this term."
    )

    old_factors = old.get("factors", [])
    factors = []
    factor_ops = []

    st.markdown("##### Factors and operators")
    for fi in range(n_factors):
        with st.container(border=True):
            if fi > 0:
                default_op = old_ops[fi - 1] if fi - 1 < len(old_ops) else "*"
                op = st.selectbox(
                    f"Operator before factor {fi+1}",
                    ["*", "/"],
                    index=0 if default_op != "/" else 1,
                    format_func=lambda x: "×  Multiply" if x == "*" else "÷  Divide",
                    key=f"{t_key}_fop_{fi-1}",
                    help=(
                        "Choose how this factor is combined with the previous factor. "
                        "Division by a decision symbol is not allowed because the model would no longer be linear."
                    ),
                    **_keep_objective_term_open_kwargs(t_key),
                )
                factor_ops.append(op)

            f = build_factor_ui(
                f"{t_key}_f{fi}",
                fi,
                old_factors[fi] if fi < len(old_factors) else None,
                catalog,
                label_map,
                default_const_type,
            )
            if f:
                factors.append(f)

    sums = []
    if n_sums:
        st.markdown("##### Summation bounds")
        st.info(
            "**How sets work**\n\n"
            "- `i`, `j`, etc. identify positions inside the sets used by parameters and decision symbols.\n"
            "- To sum over the complete set `i`, use **Lower = `1`** and **Upper = `N_i`**.\n"
            "- `N_i` means the total size of set `i`.\n"
            "- Do **not** use `i` as the upper bound of the same `i` summation; that would be circular.\n"
            "- A bound may depend on another free or outer set, e.g. `i = j+2, ..., N_i`.\n"
            "- Nested summations are evaluated from top to bottom: Summation 1 is outer, Summation 2 is inner."
        )

    used_sum_sets = []
    for si in range(n_sums):
        old_sum = old_sums[si] if si < len(old_sums) else {}
        default_set = (
            old_sum.get("set")
            if old_sum.get("set") in set_names
            else (set_names[si] if si < len(set_names) else set_names[0])
        )

        with st.container(border=True):
            st.markdown(f"**Summation {si+1}**")
            a, b, c = st.columns([1.2, 1.8, 1.8])
            set_name = a.selectbox(
                f"Sum set {si+1}",
                set_names,
                index=set_names.index(default_set),
                key=f"{t_key}_sumset_{si}",
                help=(
                    "Set iterated by this summation. "
                    "Summation 1 is the outermost summation."
                ),
                **_keep_objective_term_open_kwargs(t_key),
            )

            default_lower = str(old_sum.get("lower", "1"))
            default_upper = str(old_sum.get("upper", f"N_{set_name}"))

            # If an older UI stored the same set symbol as the upper bound,
            # show the mathematically meaningful full-range default instead.
            if default_upper == set_name:
                default_upper = f"N_{set_name}"

            lower = b.text_input(
                f"Lower bound {si+1}",
                value=default_lower,
                key=f"{t_key}_sumlo_{si}",
                help=(
                    "Inclusive lower set position. Examples: `1`, `j+2`, `2*j+1`. "
                    "It may depend on a free or outer set."
                ),
                **_keep_objective_term_open_kwargs(t_key),
            ).strip()

            upper = c.text_input(
                f"Upper bound {si+1}",
                value=default_upper,
                key=f"{t_key}_sumhi_{si}",
                help=(
                    f"Inclusive upper position. Use `N_{set_name}` to traverse the complete `{set_name}` set."
                ),
                **_keep_objective_term_open_kwargs(t_key),
            ).strip()

            if set_name in used_sum_sets:
                st.error(f"Set `{set_name}` is already used by another summation in this term.")
            used_sum_sets.append(set_name)

            sums.append({
                "set": set_name,
                "lower": lower or "1",
                "upper": upper or f"N_{set_name}",
            })

    term = {
        "connector": connector,
        # Keep sign for backward compatibility with older saved specifications.
        "sign": connector if connector in {"+", "-"} else "+",
        "factors": factors,
        "factor_ops": factor_ops,
        "sums": sums,
    }

    st.markdown("##### Term preview")
    st.latex(term_latex(term))
    return term

def object_catalog(spec: dict) -> tuple[list[dict], dict]:
    items = []
    for pn, ps in spec["parameters"].items():
        lbl = sig(pn, ps["sets"])
        items.append({"kind": "parameter", "name": pn, "sets": ps["sets"], "label": lbl})
    for decision_name, decision_record in spec["decisions"].items():
        lbl = sig(decision_name, decision_record["sets"])
        items.append({"kind": "decision", "name": decision_name, "sets": decision_record["sets"], "label": lbl})
    return items, {o["label"]: o for o in items}

def _open_family(r: int):
    st.session_state["constraint_family_expander_open"] = r

def _open_parameter(p: int):
    st.session_state["parameter_expander_open"] = p

def _open_objective_term(t: int):
    st.session_state["objective_term_expander_open"] = t

# ============================================================
# SIDEBAR
# ============================================================
n_sets = len(spec["sets"])
n_decisions = count_expanded(spec, "decisions")
n_con = count_expanded(spec, "constraints")

st.sidebar.markdown("""
<div style="padding:.4rem 0 1rem;border-bottom:1px solid rgba(61,132,255,.18);margin-bottom:1rem">
    <div style="font-size:1.45rem;font-weight:800;color:#fff;margin-bottom:.35rem">Navigation</div>
    <div style="color:#b9c9e8;font-size:.92rem">Build, validate, and solve the model step by step.</div>
</div>""", unsafe_allow_html=True)

section = st.sidebar.radio("Go to:", ["Data Input", "Model Definition", "Model Outputs"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown('<div style="font-size:1.05rem;font-weight:800;color:#fff;margin-bottom:.8rem">Current status</div>', unsafe_allow_html=True)

def _sb_kpi(label, value):
    return f"""<div style="background:linear-gradient(135deg,rgba(8,22,55,.95),rgba(3,10,28,.98));
        border:1px solid rgba(61,132,255,.22);border-radius:14px;padding:12px 14px;margin-bottom:10px">
        <div style="font-size:.88rem;color:#cfe0ff;font-weight:700">{label}</div>
        <div style="font-size:1.8rem;color:#fff;font-weight:800">{value}</div></div>"""

c1, c2 = st.sidebar.columns(2)
c1.markdown(_sb_kpi("Sets", n_sets), unsafe_allow_html=True)
c2.markdown(_sb_kpi("Decision Symbols", n_decisions), unsafe_allow_html=True)
st.sidebar.markdown(_sb_kpi("Defined constraints", n_con), unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
st.title("MILP General")
st.caption("Application for building and solving single-objective mixed-integer linear programming models.")

# ============================================================
# SECTION 1: DATA INPUT
# ============================================================
if section == "Data Input":
    hero("1. Data Input", "Define the model sets, parameters, and decision symbols.")

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Sets", len(spec["sets"]))
    with c2: kpi_card("Parameters", len(spec["parameters"]))
    with c3: kpi_card("Decision Symbols", len(spec["decisions"]))

    st.markdown("<br>", unsafe_allow_html=True)
    tab_sets, tab_par, tab_decision = st.tabs(["Sets", "Parameters", "Decision Symbols"])

    # -- SETS --
    with tab_sets:
        section_box("Set Configuration", "Define the finite sets used by parameters, decision symbols, and constraints.", "Create each set with a symbolic name and a finite size. Elements are internally ordered from position 1 to N.")
        n = st.number_input("Number of sets", 1, 10, max(1, len(spec["sets"]) or 3), step=1, key="num_sets")
        existing_names = list(spec["sets"].keys())

        set_specs_new, errors = {}, []
        used = set()
        for r in range(int(n)):
            default_name = existing_names[r] if r < len(existing_names) else f"set_{r+1}"
            default_size = spec["sets"].get(default_name, {}).get("size", 3)
            col1, col2 = st.columns(2)
            name = col1.text_input(f"Name {r+1}", value=default_name, key=f"set_name_{r}").strip()
            size = int(col2.number_input(f"Size of {name or f'set {r+1}'}", 1, 1000, int(default_size), step=1, key=f"set_size_{r}"))
            if not valid_sym(name):
                errors.append(f"`{name}` is not a valid name.")
            elif name in used:
                errors.append(f"Set `{name}` is duplicated.")
            else:
                used.add(name)
                set_specs_new[name] = {"size": size, "elements": set_elements(size, name)}

        for e in errors: st.error(e)

        if not errors:
            spec["sets"] = set_specs_new
            # Remove parameters and decision symbols whose sets no longer exist
            valid = set(set_specs_new)
            spec["parameters"] = {k: v for k, v in spec["parameters"].items() if all(i in valid for i in v.get("sets", []))}
            spec["decisions"] = {k: v for k, v in spec["decisions"].items() if all(i in valid for i in v.get("sets", []))}

            if set_specs_new:
                st.write("**Preview:**")
                st.dataframe(pd.DataFrame([
                    {"Set": n, "Size": s["size"], "Elements": ", ".join(s["elements"])}
                    for n, s in set_specs_new.items()
                ]), use_container_width=True, hide_index=True)

    # -- PARAMETERS --
    with tab_par:
        section_box(
            "Parameter Configuration",
            "Define parameters manually, load them from Excel/CSV, or generate random values.",
            "Choose the parameter signature first. For large parameter arrays, Excel/CSV or random generation is recommended."
        )
        set_specs = spec["sets"]

        if not set_specs:
            st.info("Define valid sets first.")
        else:
            cur = spec["parameters"]
            n_p = int(st.number_input(
                "Number of parameters",
                0,
                30,
                max(1, len(cur)) if cur else 1,
                step=1,
                key="num_params"
            ))
            set_options = list(set_specs.keys())
            new_params = {}
            old_names = list(cur.keys())

            if n_p == 0:
                st.info("No parameters defined.")

            for p in range(n_p):
                old_name = old_names[p] if p < len(old_names) else f"param_{p+1}"
                old_record = cur.get(old_name, {})
                default_sets = old_record.get("sets", [])

                preview_name = st.session_state.get(f"pname_{p}", old_name)
                preview_sets = st.session_state.get(f"psets_{p}", default_sets)
                preview_sets = [set_name for set_name in preview_sets if set_name in set_specs]
                preview_ne = total_elems(preview_sets, set_specs)

                preview_modes = ["Manual", "Excel/CSV", "Random"] if preview_ne <= 12 else ["Excel/CSV", "Random"]
                preview_mode = st.session_state.get(f"pmode_{p}", old_record.get("mode", preview_modes[0]))
                if preview_mode == "Excel":
                    preview_mode = "Excel/CSV"
                if preview_mode not in preview_modes:
                    preview_mode = preview_modes[0]

                preview_label = (
                    f"Parameter {p+1}: {preview_name} — "
                    f"{sig(preview_name, preview_sets)} — "
                    f"{preview_ne} element(s) — {preview_mode}"
                )
                expanded = (
                    st.session_state.get("parameter_expander_open") == p or
                    (st.session_state.get("parameter_expander_open") is None and p == 0)
                )

                with st.expander(preview_label, expanded=expanded):
                    st.markdown(f"### Parameter {p+1}")
                    col1, col2 = st.columns([2, 3])
                    pname = col1.text_input(
                        f"Parameter name {p+1}",
                        value=old_name,
                        key=f"pname_{p}",
                        on_change=_open_parameter,
                        args=(p,)
                    ).strip()
                    parameter_sets = col2.multiselect(
                        f"Sets for {pname}",
                        set_options,
                        default=default_sets,
                        key=f"psets_{p}",
                        on_change=_open_parameter,
                        args=(p,)
                    )

                    if not valid_sym(pname):
                        st.error(f"`{pname}` is not valid.")
                        continue
                    if pname in new_params:
                        st.error(f"`{pname}` is duplicated.")
                        continue

                    ne = total_elems(parameter_sets, set_specs)
                    st.write(f"**Signature:** `{sig(pname, parameter_sets)}`")
                    st.write(f"**Total number of elements:** `{ne}`")

                    modes = ["Manual", "Excel/CSV", "Random"] if ne <= 12 else ["Excel/CSV", "Random"]
                    old_mode = old_record.get("mode", modes[0])
                    if old_mode == "Excel":
                        old_mode = "Excel/CSV"
                    if old_mode not in modes:
                        old_mode = modes[0]

                    mode = st.radio(
                        f"Input mode for {pname}",
                        modes,
                        index=modes.index(old_mode),
                        horizontal=True,
                        key=f"pmode_{p}",
                        on_change=_open_parameter,
                        args=(p,)
                    )

                    old_vals = old_record.get("values", {})
                    current_values = _initial_param_values(p, pname, parameter_sets, set_specs, old_vals)
                    record = {"sets": parameter_sets, "mode": mode, "values": dict(current_values)}

                    if mode == "Manual":
                        if not parameter_sets:
                            value = st.number_input(
                                f"Value of {pname}",
                                value=scalar_get(current_values),
                                key=f"pscalar_{p}",
                                on_change=_open_parameter,
                                args=(p,)
                            )
                            record["values"] = _set_param_values(p, scalar_set(value))

                        elif len(parameter_sets) == 1:
                            labels = set_specs[parameter_sets[0]]["elements"]
                            df0 = vals_to_df1d(labels, current_values)
                            edited = st.data_editor(
                                df0,
                                use_container_width=True,
                                num_rows="fixed",
                                hide_index=True,
                                disabled=["label"],
                                key=f"pman1d_{p}_{_param_signature_key(pname, parameter_sets)}",
                                on_change=_open_parameter,
                                args=(p,)
                            )
                            values = {str((str(row["label"]),)): float(row["value"]) for _, row in edited.iterrows()}
                            record["values"] = _set_param_values(p, values)

                        else:
                            clist = combos(parameter_sets, set_specs)
                            df0 = vals_to_df(parameter_sets, clist, current_values)
                            edited = st.data_editor(
                                df0,
                                use_container_width=True,
                                num_rows="fixed",
                                hide_index=True,
                                disabled=list(parameter_sets),
                                key=f"pmannd_{p}_{_param_signature_key(pname, parameter_sets)}",
                                on_change=_open_parameter,
                                args=(p,)
                            )
                            record["values"] = _set_param_values(p, df_to_vals(edited, parameter_sets))

                    elif mode == "Excel/CSV":
                        record["values"] = parameter_template_controls(p, pname, parameter_sets, set_specs, current_values)

                    else:  # Random
                        if not parameter_sets:
                            lo, hi, intg, seed = _rand_controls(f"ps_{p}")
                            if lo > hi:
                                st.error("Minimum > maximum.")
                                continue
                            if st.button(f"Generate {pname}", key=f"pgen_{p}", on_click=_open_parameter, args=(p,)):
                                current_values = _set_param_values(p, rand_scalar(lo, hi, intg, seed))
                            st.write(f"Value: **{scalar_get(current_values):.4f}**")
                            record["values"] = dict(current_values)

                        else:
                            lo, hi, intg, seed = _rand_controls(f"prand_{p}")
                            if lo > hi:
                                st.error("Minimum > maximum.")
                                continue
                            clist = combos(parameter_sets, set_specs)
                            if st.button(f"Generate values for {pname}", key=f"pgen_{p}", on_click=_open_parameter, args=(p,)):
                                current_values = _set_param_values(p, rand_vals(clist, lo, hi, intg, seed))
                            record["values"] = dict(current_values)
                            st.dataframe(
                                template_df_for_parameter(parameter_sets, set_specs, record["values"]),
                                use_container_width=True,
                                hide_index=True
                            )

                    st.markdown("#### Parameter preview")
                    st.dataframe(
                        template_df_for_parameter(parameter_sets, set_specs, record["values"]),
                        use_container_width=True,
                        hide_index=True
                    )

                    new_params[pname] = record

            spec["parameters"] = new_params

            if new_params:
                st.markdown("---")
                st.markdown("### Saved parameters")
                for i, (name, param) in enumerate(new_params.items()):
                    title = (
                        f"Parameter {i+1}: {sig(name, param['sets'])} — "
                        f"{total_elems(param['sets'], set_specs)} element(s) — {param['mode']}"
                    )
                    with st.expander(title, expanded=False):
                        st.dataframe(
                            template_df_for_parameter(param["sets"], set_specs, param["values"]),
                            use_container_width=True,
                            hide_index=True
                        )

    # -- DECISION SYMBOLS --
    with tab_decision:
        section_box("Decision Symbol Configuration", "Define decision symbols, their sets, and their domains.", "Assign each decision symbol the sets that determine its dimensions, then choose its mathematical domain.")
        set_specs = spec["sets"]

        if not set_specs:
            st.info("Define valid sets first.")
        else:
            cur = spec["decisions"]
            n_decision_symbols = int(st.number_input("Number of decision symbols", 0, 30, max(1, len(cur)) if cur else 1, step=1, key="num_decisions"))
            set_options = list(set_specs.keys())
            dom_opts = ["Binary", "NonNegativeReals", "NonNegativeIntegers"]
            new_decisions = {}

            for decision_position in range(n_decision_symbols):
                st.markdown(f"#### Decision Symbol {decision_position+1}")
                old_names = list(cur.keys())
                old_name = old_names[decision_position] if decision_position < len(old_names) else f"x_{decision_position+1}"
                col1, col2, col3 = st.columns([2, 3, 2])
                decision_name = col1.text_input(f"Name {v+1}", value=old_name, key=f"decision_name_{decision_position}").strip()
                decision_sets = col2.multiselect(f"Sets for {decision_name}", set_options, default=cur.get(old_name, {}).get("sets", []), key=f"dsets_{decision_position}")
                old_dom = cur.get(old_name, {}).get("domain", "NonNegativeReals")
                decision_domain = col3.selectbox(f"Domain of {decision_name}", dom_opts, index=dom_opts.index(old_dom if old_dom in dom_opts else "NonNegativeReals"), key=f"decision_domain_{decision_position}")

                if not valid_sym(decision_name): st.error(f"`{decision_name}` is not valid."); continue
                if decision_name in new_decisions: st.error(f"`{decision_name}` is duplicated."); continue
                new_decisions[decision_name] = {"sets": decision_sets, "domain": decision_domain}

            spec["decisions"] = new_decisions
            if new_decisions:
                st.write("**Summary:**")
                st.dataframe(pd.DataFrame([
                    {"Decision Symbol": sig(name, record["sets"]), "Domain": DOMAIN_LABELS.get(record["domain"], record["domain"]), "Components": total_elems(record["sets"], set_specs)}
                    for name, record in new_decisions.items()
                ]), use_container_width=True, hide_index=True)

# ============================================================
# SECTION 2: MODEL DEFINITION
# ============================================================
elif section == "Model Definition":
    hero("2. Model Definition", "Build the objective function, constraint families, and mathematical representation.")
    set_specs = spec["sets"]

    if not set_specs:
        st.warning("Define at least one set first.")
    elif not spec["decisions"]:
        st.warning("Define at least one decision symbol first.")
    else:
        catalog, label_map = object_catalog(spec)
        set_names = list(set_specs.keys())
        tab_obj, tab_rest, tab_math = st.tabs(["Objective Function", "Constraints", "Mathematical Model"])

        # -- OBJECTIVE FUNCTION --
        with tab_obj:
            section_box(
                "Objective Function",
                "Combine parameters, decision symbols, constants, and summations to define the single objective.",
                "Every free set reference in the objective must be eliminated by a summation. Use N_i for the full upper bound of set i; dynamic bounds may depend on another free or outer set, such as j+2."
            )
            cur_obj = spec.get("objective") or {}
            sense_opts = ["minimize", "maximize"]
            sense = st.radio("Objective sense:", sense_opts, index=sense_opts.index(cur_obj.get("sense", "minimize")), horizontal=True, key="obj_sense", help="Choose whether the objective function is minimized or maximized.")
            old_terms = cur_obj.get("terms", [])
            n_terms = int(st.number_input("Objective terms", 1, 20, max(1, len(old_terms) or 1), step=1, key="n_obj_terms", help="Number of expression terms in the objective function. Terms can be connected by +, −, ×, or ÷. Each term is shown in a collapsible panel."))
            st.info(
                "**Expression structure**\n\n"
                "- Create separate **Objective terms** and connect them with **+**, **−**, **×**, or **÷**.\n"
                "- Inside each term, factors can also be connected using **× Multiply** or **÷ Divide**.\n"
                "- Apply summations to the complete factor chain.\n"
                "- Example: `- 8 × x[i,j]` with summations over `i` and `j` represents "
                r"$-8\sum_i\sum_j x_{ij}$."
            )

            obj_terms = []
            for t in range(n_terms):
                old_term = old_terms[t] if t < len(old_terms) else None
                if old_term:
                    preview_connector = _term_connector(old_term, t)
                    preview_symbol = {"*": "×", "/": "÷"}.get(preview_connector, preview_connector)
                    old_preview = f"{preview_symbol} {term_body_latex(old_term)}" if t > 0 else term_latex(old_term)
                else:
                    old_preview = "new term"
                term_title = f"Objective term {t+1} — {old_preview}"

                expanded = (
                    st.session_state.get("objective_term_expander_open") == t
                    or (
                        st.session_state.get("objective_term_expander_open") is None
                        and t == 0
                    )
                )

                with st.expander(term_title, expanded=expanded):
                    term = build_term_ui(
                        f"obj_t{t}",
                        t,
                        old_term,
                        catalog,
                        label_map,
                        set_names,
                    )
                    obj_terms.append(term)

            errs = validate_obj(obj_terms, set_names)
            for e in errs: st.error(e)
            if not errs: st.success("Objective function is structurally consistent.")
            spec["objective"] = {"sense": sense, "terms": obj_terms}

        # -- CONSTRAINTS --
        with tab_rest:
            section_box(
                "Constraint Families",
                "Define set-based constraint families by building the left-hand side, operator, and right-hand side.",
                "Use For all for the free sets of the family. For a complete sum over i use 1 to N_i. A dynamic bound may depend on another free or outer set, for example i=j+2,...,N_i."
            )
            old_fams = spec.get("constraints", [])
            n_fams = int(st.number_input("Constraint families", 0, 30, len(old_fams), step=1, key="n_fams", help="Number of algebraic constraint families in the model."))
            new_fams = []

            if n_fams == 0:
                st.info("No constraints defined.")

            for r in range(n_fams):
                old_fam = old_fams[r] if r < len(old_fams) else None
                default_name = (old_fam or {}).get("name", f"R{r+1}")

                # Build preview for expander title
                preview = {
                    "name": st.session_state.get(f"cfname_{r}", default_name),
                    "forall": st.session_state.get(f"cfforall_{r}", (old_fam or {}).get("forall", [])),
                    "sense": st.session_state.get(f"cfsense_{r}", (old_fam or {}).get("sense", "<=")),
                    "lhs_terms": (old_fam or {}).get("lhs_terms", []),
                    "rhs_terms": (old_fam or {}).get("rhs_terms", []),
                }
                fam_label = f"Family {r+1}: {preview['name']} — {family_latex(preview)}"
                expanded = (st.session_state.get("constraint_family_expander_open") == r or
                            (st.session_state.get("constraint_family_expander_open") is None and r == 0))

                with st.expander(fam_label, expanded=expanded):
                    st.markdown(f"### Family {r+1}")
                    cf1, cf2, cf3 = st.columns(3)
                    fname = cf1.text_input(f"Family name {r+1}", value=default_name, key=f"cfname_{r}", on_change=_open_family, args=(r,), help="Use a short symbolic name such as Capacity or Balance.").strip()
                    forall = cf2.multiselect(f"For all sets in {fname}", set_names, default=(old_fam or {}).get("forall", []), key=f"cfforall_{r}", on_change=_open_family, args=(r,), help="Select the free sets that define this constraint family.")
                    sense_f = cf3.selectbox(f"Operator for {fname}", ["<=", ">=", "="], index=["<=", ">=", "="].index((old_fam or {}).get("sense", "<=")), key=f"cfsense_{r}", on_change=_open_family, args=(r,), help="Choose the relational operator between the left-hand and right-hand sides.")

                    if not valid_sym(fname): st.error(f"`{fname}` is not valid."); continue

                    colL, colR = st.columns(2)
                    old_lhs = (old_fam or {}).get("lhs_terms", [])
                    old_rhs = (old_fam or {}).get("rhs_terms", [])

                    with colL:
                        st.markdown(f"#### LHS of {fname}")
                        n_lhs = int(st.number_input(
                            f"LHS terms for {fname}", 0, 10, len(old_lhs),
                            step=1, key=f"nlhs_{r}",
                            on_change=_open_family, args=(r,),
                            help="Each LHS term is grouped in its own box. The complete constraint family can be collapsed from its header."
                        ))
                        lhs_terms = []
                        for t in range(n_lhs):
                            with st.container(border=True):
                                st.markdown(f"**LHS term {t+1}**")
                                lhs_terms.append(
                                    build_term_ui(
                                        f"lhs_{r}_{t}", t,
                                        old_lhs[t] if t < len(old_lhs) else None,
                                        catalog, label_map, set_names
                                    )
                                )

                    with colR:
                        st.markdown(f"#### RHS of {fname}")
                        n_rhs = int(st.number_input(
                            f"RHS terms for {fname}", 0, 10, len(old_rhs),
                            step=1, key=f"nrhs_{r}",
                            on_change=_open_family, args=(r,),
                            help="Each RHS term is grouped in its own box. The complete constraint family can be collapsed from its header."
                        ))
                        rhs_terms = []
                        for t in range(n_rhs):
                            with st.container(border=True):
                                st.markdown(f"**RHS term {t+1}**")
                                rhs_terms.append(
                                    build_term_ui(
                                        f"rhs_{r}_{t}", t,
                                        old_rhs[t] if t < len(old_rhs) else None,
                                        catalog, label_map, set_names,
                                        default_const_type="constant"
                                    )
                                )

                    family_record = {"name": fname, "forall": forall, "sense": sense_f, "lhs_terms": lhs_terms, "rhs_terms": rhs_terms}
                    st.markdown(f"### Preview — {fname}")
                    st.latex(family_latex(family_record))
                    fam_errs = validate_family(family_record, set_names)
                    for e in fam_errs: st.error(e)
                    if not fam_errs: st.success("Constraint family is structurally consistent.")
                    new_fams.append(family_record)

            spec["constraints"] = new_fams

            if new_fams:
                st.markdown("---")
                st.markdown("### Saved constraint families")
                for i, fam in enumerate(new_fams):
                    with st.expander(f"Family {i+1}: {fam.get('name')} — {family_latex(fam)}", expanded=False):
                        st.latex(family_latex(fam))

        # -- MATHEMATICAL MODEL --
        with tab_math:
            section_box(
                "Mathematical Model",
                "Review the complete algebraic model before solving it.",
                "Check the objective, constraint families, quantifiers, and dynamic summation bounds exactly as they will be interpreted by the solver."
            )
            st.markdown("### Structured model")
            obj = spec.get("objective")
            if obj:
                symbol = r"\min" if obj["sense"] == "minimize" else r"\max"
                st.latex(rf"{symbol}\ Z = {expr_latex(obj['terms'])}")
            else:
                st.info("No objective function defined.")
            st.markdown("**Subject to:**")
            if not spec["constraints"]:
                st.info("No constraints defined.")
            else:
                for fam in spec["constraints"]:
                    st.latex(family_latex(fam))

# ============================================================
# SECTION 3: MODEL OUTPUTS
# ============================================================
elif section == "Model Outputs":
    hero("3. Model Outputs", "Solve the model and inspect the optimal objective value and decision symbols.")

    # Validate
    errs = []
    if not spec["objective"]:
        errs.append("No objective function defined.")
    else:
        errs.extend(validate_obj(spec["objective"].get("terms", []), list(spec["sets"].keys())))
    for fam in spec.get("constraints", []):
        errs.extend(validate_family(fam, list(spec["sets"].keys())))
    errs.extend(validate_linearity(spec))
    if not spec["decisions"]: errs.append("No decision symbols defined.")
    if not spec["sets"]: errs.append("No sets defined.")

    for e in errs: st.error(e)
    if errs: st.stop()
    st.success("Model specification is valid.")

    tab_solve, tab_decision_values = st.tabs(["Solve", "Decision Values"])

    with tab_solve:
        section_box(
            "Solve Model",
            "Validate the model, select an available solver, and compute the solution.",
            "The application currently supports linear models. Products containing more than one decision symbol are rejected as nonlinear."
        )
        st.subheader("Solve model")
        solver_label = st.selectbox(
            "Solver",
            list(SOLVER_OPTIONS.keys()),
            index=0,
            help="HiGHS is used to solve continuous, integer, and binary linear models."
        )

        if st.button("Solve model", type="primary"):
            try:
                model = build_pyomo_model(spec)
                solver_name, solver = solver_factory_from_label(solver_label)
                result = solver.solve(model)

                try:
                    objective_value = pyo.value(model.OBJ)
                except Exception:
                    objective_value = None

                spec["results"] = {
                    "solver_name": solver_name,
                    "termination_condition": str(result.solver.termination_condition),
                    "status": str(result.solver.status),
                    "objective_value": objective_value,
                }
                st.session_state["solved_model_object"] = model
                st.success("Model solved successfully.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        results = spec.get("results")
        model = st.session_state.get("solved_model_object")
        if not results or not model:
            st.info("The model has not been solved yet.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: kpi_card("Solver", results.get("solver_name", ""))
            with c2: kpi_card("Status", results["status"])
            with c3: kpi_card("Termination", results["termination_condition"])
            obj_val = results.get("objective_value")
            kpi_card("Optimal value", "Not available" if obj_val is None else f"{obj_val:,.6f}")

    with tab_decision_values:
        section_box(
            "Decision Values",
            "Inspect the value of each decision symbol and export the results.",
            "Select one decision symbol for its full solution table or use Nonzero Decisions to focus only on active decisions."
        )
        results = spec.get("results")
        model = st.session_state.get("solved_model_object")
        if not results or not model:
            st.info("Solve the model first.")
        else:
            subtab_decision, subtab_nz = st.tabs(["Select Decision", "Nonzero Decisions"])

            with subtab_decision:
                st.subheader("Solution by decision")
                decision_names = list(spec["decisions"].keys())
                sel = st.selectbox("Decision Symbol", decision_names)
                df = decision_solution_df(model, sel, spec["decisions"][sel], spec["sets"])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("Download CSV", data=df.to_csv(index=False).encode(), file_name=f"{sel}_solution.csv", mime="text/csv")

            with subtab_nz:
                st.subheader("Nonzero decisions")
                full_df = all_decisions_df(model, spec)
                nz_df = full_df[full_df["value"].abs() > 1e-9].reset_index(drop=True) if not full_df.empty else full_df
                if nz_df.empty:
                    st.info("There are no nonzero decisions.")
                else:
                    st.dataframe(nz_df, use_container_width=True, hide_index=True)
                st.download_button("Download CSV", data=nz_df.to_csv(index=False).encode(), file_name="nonzero_decisions.csv", mime="text/csv")
