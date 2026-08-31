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
st.set_page_config(page_title="Algebraic Model Builder", layout="wide")

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
    "indices": {}, "parameters": {}, "variables": {},
    "objective": None, "constraints": [], "results": None,
}

def _init():
    if "model_spec" not in st.session_state:
        st.session_state["model_spec"] = _EMPTY_SPEC.copy()
    if "constraint_family_expander_abierto" not in st.session_state:
        st.session_state["constraint_family_expander_abierto"] = None
    if "parameter_expander_abierto" not in st.session_state:
        st.session_state["parameter_expander_abierto"] = None

_init()
spec = st.session_state["model_spec"]

# ============================================================
# UTILITIES — SYMBOLS & INDICES
# ============================================================
def valid_sym(name: str) -> bool:
    name = (name or "").strip()
    return bool(name) and (name[0].isalpha() or name[0] == "_") and all(c.isalnum() or c == "_" for c in name[1:])

def idx_elements(size: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, size + 1)]

def combos(idx_names: list[str], idx_specs: dict) -> list[tuple]:
    if not idx_names:
        return [tuple()]
    return list(itertools.product(*[idx_specs[n]["elements"] for n in idx_names]))

def total_elems(idx_names: list[str], idx_specs: dict) -> int:
    n = 1
    for k in idx_names:
        n *= idx_specs[k]["size"]
    return n if idx_names else 1

def sig(name: str, idxs: list[str]) -> str:
    return f"{name}[{', '.join(idxs)}]" if idxs else name

# ============================================================
# UTILITIES — VALUES SERIALIZATION
# ============================================================
def scalar_get(vals: dict, default=0.0) -> float:
    return float(vals.get("__scalar__", default))

def scalar_set(v: float) -> dict:
    return {"__scalar__": float(v)}

def df_to_vals(df: pd.DataFrame, idx_names: list[str]) -> dict:
    out = {}
    if len(idx_names) == 1:
        for _, row in df.iterrows():
            out[str((row["label"],))] = float(row["value"])
    else:
        for _, row in df.iterrows():
            key = tuple(str(row[i]) for i in idx_names)
            out[str(key)] = float(row["value"])
    return out

def vals_to_df(idx_names: list[str], combo_list: list[tuple], vals: dict) -> pd.DataFrame:
    rows = []
    for c in combo_list:
        row = {n: c[i] for i, n in enumerate(idx_names)}
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


def _param_signature_key(pname: str, idx_names: list[str]) -> str:
    return f"{pname}__{'_'.join(idx_names) if idx_names else 'scalar'}"


def _empty_values_for_parameter(idx_names: list[str], idx_specs: dict) -> dict:
    if not idx_names:
        return {"__scalar__": 0.0}
    return {str(c): 0.0 for c in combos(idx_names, idx_specs)}


def _values_match_structure(values: dict, idx_names: list[str], idx_specs: dict) -> bool:
    if not isinstance(values, dict):
        return False
    if not idx_names:
        return "__scalar__" in values
    expected = {str(c) for c in combos(idx_names, idx_specs)}
    return expected.issubset(set(values.keys()))


def _initial_param_values(row_pos: int, pname: str, idx_names: list[str], idx_specs: dict, old_vals: dict) -> dict:
    """Return persistent values compatible with the current parameter structure."""
    store_key = _param_store_key(row_pos)

    # 1) Priority: live values from the editor in session_state.
    stored = st.session_state.get(store_key)
    if _values_match_structure(stored, idx_names, idx_specs):
        return dict(stored)

    # 2) Then: values saved in the previous spec.
    if _values_match_structure(old_vals, idx_names, idx_specs):
        st.session_state[store_key] = dict(old_vals)
        return dict(old_vals)

    # 3) If dimensionality changed, initialize zeros with the new structure.
    fresh = _empty_values_for_parameter(idx_names, idx_specs)
    st.session_state[store_key] = dict(fresh)
    return fresh


def _set_param_values(row_pos: int, values: dict) -> dict:
    st.session_state[_param_store_key(row_pos)] = dict(values)
    return dict(values)


def template_df_for_parameter(idx_names: list[str], idx_specs: dict, current_values: dict) -> pd.DataFrame:
    if not idx_names:
        return pd.DataFrame([{"value": scalar_get(current_values, 0.0)}])
    if len(idx_names) == 1:
        return vals_to_df1d(idx_specs[idx_names[0]]["elements"], current_values)
    return vals_to_df(idx_names, combos(idx_names, idx_specs), current_values)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_xlsx_bytes(df: pd.DataFrame) -> bytes | None:
    """Create XLSX only when xlsxwriter is installed; otherwise return None without breaking the app."""
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="parameter")
        return buffer.getvalue()
    except Exception:
        return None


def read_parameter_upload(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file), None
        if name.endswith((".xlsx", ".xls")):
            try:
                return pd.read_excel(uploaded_file), None
            except ImportError:
                return None, "To load .xlsx files, install `openpyxl` or use the .csv template."
            except ModuleNotFoundError:
                return None, "To load .xlsx files, install `openpyxl` or use the .csv template."
        return None, "Unsupported format. Use .csv or .xlsx."
    except Exception as exc:
        return None, f"The file could not be read: {exc}"


def validate_and_convert_parameter_df(df: pd.DataFrame, idx_names: list[str], idx_specs: dict) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    if df is None or df.empty:
        return None, ["The file is empty."]

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "value" not in df.columns:
        errors.append("A column named exactly `value` is required.")
        return None, errors

    try:
        df["value"] = pd.to_numeric(df["value"], errors="raise")
    except Exception:
        errors.append("The `value` column must contain numeric values only.")
        return None, errors

    if not idx_names:
        if len(df) < 1:
            errors.append("A scalar parameter requires at least one row containing the `value` column.")
            return None, errors
        return {"__scalar__": float(df.iloc[0]["value"])}, []

    if len(idx_names) == 1:
        idx = idx_names[0]
        index_col = "label" if "label" in df.columns else idx if idx in df.columns else None
        if index_col is None:
            errors.append(f"A `label` column or an index column is required: `{idx}`.")
            return None, errors

        work = df[[index_col, "value"]].copy()
        work[index_col] = work[index_col].astype(str)

        if work[index_col].duplicated().any():
            repeated = work.loc[work[index_col].duplicated(), index_col].unique().tolist()
            errors.append(f"Duplicated labels: {repeated}.")

        expected = set(idx_specs[idx]["elements"])
        observed = set(work[index_col].tolist())
        missing = sorted(expected - observed)
        observed_extra = sorted(observed - expected)
        if missing:
            errors.append(f"Missing labels for index `{idx}`: {missing}.")
        if observed_extra:
            errors.append(f"Labels not belonging to index `{idx}`: {observed_extra}.")
        if errors:
            return None, errors

        values = {}
        for _, row in work.iterrows():
            values[str((str(row[index_col]),))] = float(row["value"])
        return values, []

    required_cols = idx_names + ["value"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}.")
        return None, errors

    work = df[required_cols].copy()
    for idx in idx_names:
        work[idx] = work[idx].astype(str)

    if work.duplicated(subset=idx_names).any():
        repeated_rows = work.loc[work.duplicated(subset=idx_names), idx_names].drop_duplicates().to_dict("records")
        errors.append(f"Duplicated index combinations: {repeated_rows}.")

    expected = set(combos(idx_names, idx_specs))
    observed = set(tuple(row[idx] for idx in idx_names) for _, row in work.iterrows())
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"Missing index combinations: {missing}.")
    if extra:
        errors.append(f"Combinations not belonging to the defined indices: {extra}.")
    if errors:
        return None, errors

    values = {}
    for _, row in work.iterrows():
        key = tuple(str(row[idx]) for idx in idx_names)
        values[str(key)] = float(row["value"])
    return values, []


def parameter_template_controls(row_pos: int, pname: str, idx_names: list[str], idx_specs: dict, current_values: dict):
    df_template = template_df_for_parameter(idx_names, idx_specs, current_values)
    file_base = f"template_{pname}"
    widget_suffix = _param_signature_key(pname, idx_names)

    st.caption("Download the template, fill in only the `value` column, and upload it again. Do not rename the index columns.")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download CSV template",
            data=dataframe_to_csv_bytes(df_template),
            file_name=f"{file_base}.csv",
            mime="text/csv",
            key=f"tmpl_csv_{row_pos}_{widget_suffix}",
        )
    with dl2:
        xlsx_bytes = dataframe_to_xlsx_bytes(df_template)
        if xlsx_bytes is not None:
            st.download_button(
                "Download Excel template",
                data=xlsx_bytes,
                file_name=f"{file_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"tmpl_xlsx_{row_pos}_{widget_suffix}",
            )
        else:
            st.info("Install `xlsxwriter` to download .xlsx files. The CSV template can still be opened in Excel.")

    uploaded = st.file_uploader(
        f"Upload values for {pname}",
        type=["csv", "xlsx", "xls"],
        key=f"upload_param_{row_pos}_{widget_suffix}",
    )

    if uploaded is None:
        st.dataframe(df_template, use_container_width=True, hide_index=True)
        return current_values

    df_uploaded, read_error = read_parameter_upload(uploaded)
    if read_error:
        st.error(read_error)
        st.dataframe(df_template, use_container_width=True, hide_index=True)
        return current_values

    values, errors = validate_and_convert_parameter_df(df_uploaded, idx_names, idx_specs)
    if errors:
        for err in errors:
            st.error(err)
        st.write("Uploaded file preview:")
        st.dataframe(df_uploaded, use_container_width=True, hide_index=True)
        return current_values

    st.success("Values loaded successfully.")
    _set_param_values(row_pos, values)
    st.dataframe(template_df_for_parameter(idx_names, idx_specs, values), use_container_width=True, hide_index=True)
    return values

# ============================================================
# UTILITIES — EXPRESSIONS + DYNAMIC SUMMATION BOUNDS
# ============================================================
DOMAIN_LABELS = {"Binary": "Binary", "NonNegativeReals": "Nonnegative reals", "NonNegativeIntegers": "Nonnegative integers"}

def _term_sums(t: dict) -> list[dict]:
    """Return the new summation structure while remaining compatible with old models."""
    if "sums" in t:
        return t.get("sums", [])
    return [{"index": idx, "lower": "1", "upper": f"N_{idx}"} for idx in t.get("sum_over", [])]

def _expr_names(expr: str) -> set[str]:
    try:
        tree = ast.parse((expr or "").strip(), mode="eval")
    except Exception:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

def _safe_bound_eval(expr: str, numeric_env: dict[str, int], idx_specs: dict) -> int:
    """Safely evaluate integer bound expressions such as j+2, 2*j+1, or N_i-1."""
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("A summation bound cannot be empty.")

    names = dict(numeric_env)
    names.update({f"N_{idx}": int(data["size"]) for idx, data in idx_specs.items()})
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
        raise ValueError(f"Bound `{expr}` evaluates to {value}, but index positions must be integers.")
    return int(rounded)

def _validate_bound_expression(expr: str, idx_names: list[str], current_sum_idx: str | None = None) -> list[str]:
    errors = []
    try:
        tree = ast.parse((expr or "").strip(), mode="eval")
    except Exception:
        return [f"Invalid bound expression `{expr}`."]

    allowed_names = set(idx_names) | {f"N_{idx}" for idx in idx_names}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            errors.append(f"Functions are not allowed in bound `{expr}`.")
        elif isinstance(node, ast.Name) and node.id not in allowed_names:
            errors.append(f"Unknown symbol `{node.id}` in bound `{expr}`.")
        elif isinstance(node, (ast.Attribute, ast.Subscript, ast.List, ast.Dict, ast.Set, ast.Tuple)):
            errors.append(f"Unsupported syntax in bound `{expr}`.")
    if current_sum_idx and current_sum_idx in _expr_names(expr):
        errors.append(f"Bound `{expr}` cannot depend on its own summation index `{current_sum_idx}`.")
    return list(dict.fromkeys(errors))

def _index_position(idx: str, value: str, idx_specs: dict) -> int:
    elements = idx_specs[idx]["elements"]
    try:
        return elements.index(value) + 1
    except ValueError as exc:
        raise ValueError(f"Value `{value}` does not belong to index `{idx}`.") from exc

def _bound_env(env: dict, idx_specs: dict) -> dict[str, int]:
    return {idx: _index_position(idx, value, idx_specs) for idx, value in env.items() if idx in idx_specs}

def _sum_values(sum_spec: dict, env: dict, idx_specs: dict) -> list[str]:
    idx = sum_spec["index"]
    numeric_env = _bound_env(env, idx_specs)
    lower = _safe_bound_eval(sum_spec.get("lower", "1"), numeric_env, idx_specs)
    upper = _safe_bound_eval(sum_spec.get("upper", f"N_{idx}"), numeric_env, idx_specs)
    elements = idx_specs[idx]["elements"]
    if lower > upper:
        return []
    # Intersect the requested range with the valid positions of the index set.
    start = max(1, lower)
    end = min(len(elements), upper)
    if start > end:
        return []
    return elements[start - 1:end]

def _fac_latex(f: dict) -> str:
    if f["type"] == "constant":
        v = float(f["value"])
        return str(int(v)) if v == int(v) else f"{v:.2f}"
    n, idxs = f["name"], f["indices"]
    return n if not idxs else rf"{n}_{{{','.join(idxs)}}}"

def term_latex(t: dict) -> str:
    factors = t.get("factors", [])
    body = r" \cdot ".join(_fac_latex(f) for f in factors) if factors else "0"
    for s in reversed(_term_sums(t)):
        idx = s["index"]
        lower = s.get("lower", "1")
        upper = s.get("upper", f"N_{idx}")
        body = rf"\sum_{{{idx}={lower}}}^{{{upper}}}\left({body}\right)"
    return f"- {body}" if t.get("sign") == "-" else f"+ {body}"

def expr_latex(terms: list[dict]) -> str:
    if not terms:
        return "0"
    out = " ".join(term_latex(t) for t in terms).strip()
    return out[2:] if out.startswith("+ ") else out

def family_latex(fam: dict) -> str:
    sense_map = {"<=": r"\leq", ">=": r"\geq", "=": "="}
    lhs = expr_latex(fam.get("lhs_terms", []))
    rhs = expr_latex(fam.get("rhs_terms", []))
    s = sense_map.get(fam.get("sense", "<="), r"\leq")
    txt = f"{lhs} {s} {rhs}"
    if fam.get("forall"):
        txt += r"\quad \forall " + ", ".join(fam["forall"])
    return txt

def term_free_idxs(t: dict) -> list[str]:
    used = []
    for f in t.get("factors", []):
        if f["type"] == "object":
            used.extend(f["indices"])

    sums = _term_sums(t)
    sum_indices = [s["index"] for s in sums]
    for s in sums:
        for expr in (s.get("lower", "1"), s.get("upper", f"N_{s['index']}")):
            used.extend(name for name in _expr_names(expr) if not name.startswith("N_"))

    seen, out = set(), []
    for x in used:
        if x not in seen:
            seen.add(x); out.append(x)
    return [x for x in out if x not in sum_indices]

# ============================================================
# UTILITIES — VALIDATION
# ============================================================
def validate_term_sums(t: dict, idx_names: list[str], context: str) -> list[str]:
    errs = []
    sums = _term_sums(t)
    seen = set()
    for pos, s in enumerate(sums, 1):
        idx = s.get("index")
        if idx not in idx_names:
            errs.append(f"{context}: summation {pos} uses an undefined index `{idx}`.")
            continue
        if idx in seen:
            errs.append(f"{context}: index `{idx}` is used in more than one summation in the same term.")
        seen.add(idx)
        errs.extend(f"{context}: {e}" for e in _validate_bound_expression(s.get("lower", "1"), idx_names, idx))
        errs.extend(f"{context}: {e}" for e in _validate_bound_expression(s.get("upper", f"N_{idx}"), idx_names, idx))

        later_indices = {z.get("index") for z in sums[pos:] if z.get("index")}
        dependencies = (_expr_names(s.get("lower", "1")) | _expr_names(s.get("upper", f"N_{idx}"))) & set(idx_names)
        invalid_later = sorted(dependencies & later_indices)
        if invalid_later:
            errs.append(f"{context}: bounds for `{idx}` cannot depend on inner/later summation indices {invalid_later}.")
    return errs

def validate_obj(terms: list[dict], idx_names: list[str] | None = None) -> list[str]:
    errs = []
    idx_names = idx_names or list(spec.get("indices", {}).keys())
    for i, t in enumerate(terms, 1):
        errs.extend(validate_term_sums(t, idx_names, f"Objective term {i}"))
        free = term_free_idxs(t)
        if free:
            errs.append(f"Objective term {i}: free indices without a matching summation → {', '.join(free)}")
    return errs

def validate_family(fam: dict, idx_names: list[str] | None = None) -> list[str]:
    errs = []
    idx_names = idx_names or list(spec.get("indices", {}).keys())
    for i, t in enumerate(fam.get("lhs_terms", []), 1):
        errs.extend(validate_term_sums(t, idx_names, f"Constraint {fam.get('name', '')} LHS term {i}"))
    for i, t in enumerate(fam.get("rhs_terms", []), 1):
        errs.extend(validate_term_sums(t, idx_names, f"Constraint {fam.get('name', '')} RHS term {i}"))

    lhs_free = list({x for t in fam.get("lhs_terms", []) for x in term_free_idxs(t)})
    rhs_free = list({x for t in fam.get("rhs_terms", []) for x in term_free_idxs(t)})
    forall = fam.get("forall", [])
    lc, rc = not lhs_free, not rhs_free
    if not lc and not rc:
        if sorted(lhs_free) != sorted(rhs_free):
            errs.append(f"Free indices on LHS {lhs_free} ≠ free indices on RHS {rhs_free}")
        if sorted(lhs_free) != sorted(forall):
            errs.append(f"Free indices {lhs_free} ≠ forall indices {forall}")
    elif lc and not rc:
        if sorted(rhs_free) != sorted(forall):
            errs.append(f"Constant LHS: RHS indices {rhs_free} must match forall indices {forall}")
    elif not lc and rc:
        if sorted(lhs_free) != sorted(forall):
            errs.append(f"Constant RHS: LHS indices {lhs_free} must match forall indices {forall}")
    elif lc and rc and forall:
        errs.append(f"Both sides are constant, but forall indices were defined: {forall}")
    return errs

def validate_linearity(spec: dict) -> list[str]:
    errs = []
    def chk(terms, ctx):
        for i, t in enumerate(terms, 1):
            nv = sum(1 for f in t.get("factors", []) if f["type"] == "object" and f.get("kind") == "variable")
            if nv > 1:
                errs.append(f"{ctx} term {i}: {nv} variables are multiplied together → nonlinear expression")
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
    "HiGHS (appsi_highs)": "appsi_highs",
    "GLPK": "glpk",
    "CBC": "cbc",
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
    comp = getattr(model, f"{'par' if f['kind'] == 'parameter' else 'var'}_{f['name']}")
    idxs = f["indices"]
    if not idxs:
        return comp
    key = tuple(env[i] for i in idxs)
    return comp[key[0]] if len(key) == 1 else comp[key]

def _eval_term(model, t: dict, env: dict):
    sums = _term_sums(t)
    idx_specs = getattr(model, "_idx_specs")

    def recurse(pos, local_env):
        if pos == len(sums):
            val = 1
            for f in t.get("factors", []):
                val = val * _get_val(model, f, local_env)
            return (-val) if t.get("sign") == "-" else val

        sum_spec = sums[pos]
        idx = sum_spec["index"]
        values = _sum_values(sum_spec, local_env, idx_specs)
        return sum(recurse(pos + 1, {**local_env, idx: v}) for v in values)

    return recurse(0, dict(env))

def _build_expr(model, terms: list[dict], env: dict):
    return sum(_eval_term(model, t, env) for t in terms) if terms else 0

def build_pyomo_model(spec: dict):
    m = pyo.ConcreteModel()
    idx_specs = spec["indices"]
    m._idx_specs = idx_specs

    for n, s in idx_specs.items():
        setattr(m, f"set_{n}", pyo.Set(initialize=s["elements"], ordered=True))

    for pn, ps in spec["parameters"].items():
        idxs, vals = ps["indices"], ps["values"]
        if not idxs:
            setattr(m, f"par_{pn}", pyo.Param(initialize=float(vals["__scalar__"])))
        else:
            sets = [getattr(m, f"set_{i}") for i in idxs]
            init = {}
            for c in combos(idxs, idx_specs):
                k = c[0] if len(c) == 1 else c
                init[k] = float(vals.get(str(c), 0.0))
            setattr(m, f"par_{pn}", pyo.Param(*sets, initialize=init))

    for vn, vs in spec["variables"].items():
        idxs = vs["indices"]
        dom = _DOMAINS[vs["domain"]]
        if not idxs:
            setattr(m, f"var_{vn}", pyo.Var(domain=dom))
        else:
            sets = [getattr(m, f"set_{i}") for i in idxs]
            setattr(m, f"var_{vn}", pyo.Var(*sets, domain=dom))

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

def var_solution_df(model, vname: str, vspec: dict, idx_specs: dict) -> pd.DataFrame:
    comp = getattr(model, f"var_{vname}")
    idxs = vspec["indices"]
    if not idxs:
        return pd.DataFrame({"variable": [vname], "value": [pyo.value(comp)]})
    rows = []
    for c in combos(idxs, idx_specs):
        row = {idx: c[i] for i, idx in enumerate(idxs)}
        row["value"] = pyo.value(comp[c[0]] if len(c) == 1 else comp[c])
        rows.append(row)
    return pd.DataFrame(rows)

def all_vars_df(model, spec: dict) -> pd.DataFrame:
    dfs = []
    for vn, vs in spec["variables"].items():
        df = var_solution_df(model, vn, vs, spec["indices"])
        df.insert(0, "variable_name", vn)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def count_expanded(spec: dict, key: str) -> int:
    idx_specs = spec.get("indices", {})
    total = 0
    items = spec.get(key, {})
    collection = items.values() if isinstance(items, dict) else items
    for item in collection:
        idxs = item.get("indices", item.get("forall", []))
        if not idxs:
            total += 1
        elif all(i in idx_specs for i in idxs):
            n = 1
            for i in idxs:
                n *= idx_specs[i]["size"]
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

def build_factor_ui(t_key: str, f_idx: int, old_factor: dict | None, catalog: list[dict], label_map: dict, default_type="object") -> dict | None:
    cfa, cfb, cfc = st.columns([1.5, 2.5, 2])
    ftype = cfa.selectbox(
        f"Factor type {f_idx+1}", ["object", "constant"],
        index=0 if (old_factor or {}).get("type", default_type) == "object" else 1,
        format_func=lambda x: "Parameter / Variable" if x == "object" else "Constant",
        key=f"{t_key}_ftype_{f_idx}",
        help="Choose whether this factor is a model object or a numeric constant."
    )
    if ftype == "object":
        labels = [o["label"] for o in catalog]
        if not labels:
            st.error("No parameters or variables are available.")
            return None
        default_lbl = labels[0]
        if old_factor and old_factor.get("type") == "object" and old_factor.get("label") in labels:
            default_lbl = old_factor["label"]
        chosen = cfb.selectbox(
            f"Object {f_idx+1}", labels, index=labels.index(default_lbl), key=f"{t_key}_fobj_{f_idx}",
            help="Select the parameter or decision variable used in this product."
        )
        item = label_map[chosen]
        cfc.write(f"Indices: {', '.join(item['indices']) or 'none'}")
        return {"type": "object", "kind": item["kind"], "name": item["name"], "indices": item["indices"], "label": item["label"]}
    else:
        dval = float((old_factor or {}).get("value", 0.0)) if (old_factor or {}).get("type") == "constant" else 0.0
        val = cfb.number_input(
            f"Constant {f_idx+1}", value=dval, key=f"{t_key}_fconst_{f_idx}",
            help="Enter the numeric coefficient or constant used in this product."
        )
        return {"type": "constant", "value": float(val)}

def build_term_ui(t_key: str, t_idx: int, old_term: dict | None, catalog: list[dict], label_map: dict, idx_names: list[str], default_const_type="object") -> dict:
    old = old_term or {}
    old_sums = _term_sums(old)

    c1, c2, c3 = st.columns([1, 2, 2])
    sign = c1.selectbox(
        f"Sign {t_idx+1}", ["+", "-"], index=0 if old.get("sign", "+") == "+" else 1, key=f"{t_key}_sign",
        help="Select whether this term is added to or subtracted from the expression."
    )
    n_factors = int(c2.number_input(
        f"Factors {t_idx+1}", min_value=1, max_value=4, value=max(1, len(old.get("factors", [])) or 2), step=1, key=f"{t_key}_nfac",
        help="Number of factors multiplied inside this term."
    ))
    n_sums = int(c3.number_input(
        f"Summations {t_idx+1}", min_value=0, max_value=max(0, len(idx_names)), value=min(len(old_sums), len(idx_names)), step=1, key=f"{t_key}_nsums",
        help="Number of nested summations for this term. Each summation can have its own lower and upper bound."
    ))

    old_factors = old.get("factors", [])
    factors = []
    for fi in range(n_factors):
        f = build_factor_ui(f"{t_key}_f{fi}", fi, old_factors[fi] if fi < len(old_factors) else None, catalog, label_map, default_const_type)
        if f:
            factors.append(f)

    sums = []
    if n_sums:
        st.markdown("##### Summation bounds")
        st.caption("Bounds use index positions. Examples: `1`, `j+2`, `2*j+1`, `N_i-1`. `N_i` means the size of index `i`.")

    used_sum_indices = []
    for si in range(n_sums):
        old_sum = old_sums[si] if si < len(old_sums) else {}
        default_idx = old_sum.get("index") if old_sum.get("index") in idx_names else (idx_names[si] if si < len(idx_names) else idx_names[0])
        a, b, c = st.columns([1.2, 1.8, 1.8])
        idx = a.selectbox(
            f"Sum index {si+1}", idx_names, index=idx_names.index(default_idx), key=f"{t_key}_sumidx_{si}",
            help="Index iterated by this summation. Nested summations are evaluated from top to bottom."
        )
        lower = b.text_input(
            f"Lower bound {si+1}", value=str(old_sum.get("lower", "1")), key=f"{t_key}_sumlo_{si}",
            help="Inclusive lower position. It may depend on a free or outer index, e.g. `j+2`."
        ).strip()
        upper = c.text_input(
            f"Upper bound {si+1}", value=str(old_sum.get("upper", f"N_{idx}")), key=f"{t_key}_sumhi_{si}",
            help=f"Inclusive upper position. Use `N_{idx}` for the full size of index `{idx}`."
        ).strip()
        if idx in used_sum_indices:
            st.error(f"Index `{idx}` is already used by another summation in this term.")
        used_sum_indices.append(idx)
        sums.append({"index": idx, "lower": lower or "1", "upper": upper or f"N_{idx}"})

    term = {"sign": sign, "factors": factors, "sums": sums}
    st.latex(term_latex(term))
    return term

def object_catalog(spec: dict) -> tuple[list[dict], dict]:
    items = []
    for pn, ps in spec["parameters"].items():
        lbl = sig(pn, ps["indices"])
        items.append({"kind": "parameter", "name": pn, "indices": ps["indices"], "label": lbl})
    for vn, vs in spec["variables"].items():
        lbl = sig(vn, vs["indices"])
        items.append({"kind": "variable", "name": vn, "indices": vs["indices"], "label": lbl})
    return items, {o["label"]: o for o in items}

def _open_family(r: int):
    st.session_state["constraint_family_expander_abierto"] = r

def _open_parameter(p: int):
    st.session_state["parameter_expander_abierto"] = p

# ============================================================
# SIDEBAR
# ============================================================
n_idx = len(spec["indices"])
n_var = count_expanded(spec, "variables")
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
c1.markdown(_sb_kpi("Indices", n_idx), unsafe_allow_html=True)
c2.markdown(_sb_kpi("Variables", n_var), unsafe_allow_html=True)
st.sidebar.markdown(_sb_kpi("Defined constraints", n_con), unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
st.title("Linear Model Solver")
st.caption("Application for building and solving single-objective linear models.")

# ============================================================
# SECTION 1: DATA INPUT
# ============================================================
if section == "Data Input":
    hero("1. Data Input", "Define the model indices, parameters, and decision variables.")

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Indices", len(spec["indices"]))
    with c2: kpi_card("Parameters", len(spec["parameters"]))
    with c3: kpi_card("Variables", len(spec["variables"]))

    st.markdown("<br>", unsafe_allow_html=True)
    tab_ind, tab_par, tab_var = st.tabs(["Indices", "Parameters", "Variables"])

    # -- INDICES --
    with tab_ind:
        section_box("Index Configuration", "Define the base index sets used by parameters, variables, and constraints.", "Create each index with a symbolic name and a finite size. Elements are internally ordered from position 1 to N.")
        n = st.number_input("Number of indices", 1, 10, max(1, len(spec["indices"]) or 3), step=1, key="num_indices")
        existing_names = list(spec["indices"].keys())

        idx_specs_new, errors = {}, []
        used = set()
        for r in range(int(n)):
            default_name = existing_names[r] if r < len(existing_names) else f"idx_{r+1}"
            default_size = spec["indices"].get(default_name, {}).get("size", 3)
            col1, col2 = st.columns(2)
            name = col1.text_input(f"Name {r+1}", value=default_name, key=f"idx_name_{r}").strip()
            size = int(col2.number_input(f"Size of {name or f'index {r+1}'}", 1, 1000, int(default_size), step=1, key=f"idx_size_{r}"))
            if not valid_sym(name):
                errors.append(f"`{name}` is not a valid name.")
            elif name in used:
                errors.append(f"Index `{name}` is duplicated.")
            else:
                used.add(name)
                idx_specs_new[name] = {"size": size, "elements": idx_elements(size, name)}

        for e in errors: st.error(e)

        if not errors:
            spec["indices"] = idx_specs_new
            # clean orphan params/vars
            valid = set(idx_specs_new)
            spec["parameters"] = {k: v for k, v in spec["parameters"].items() if all(i in valid for i in v.get("indices", []))}
            spec["variables"] = {k: v for k, v in spec["variables"].items() if all(i in valid for i in v.get("indices", []))}

            if idx_specs_new:
                st.write("**Preview:**")
                st.dataframe(pd.DataFrame([
                    {"Index": n, "Size": s["size"], "Elements": ", ".join(s["elements"])}
                    for n, s in idx_specs_new.items()
                ]), use_container_width=True, hide_index=True)

    # -- PARAMETERS --
    with tab_par:
        section_box(
            "Parameter Configuration",
            "Define parameters manually, load them from Excel/CSV, or generate random values.",
            "Choose the parameter signature first. For large parameter arrays, Excel/CSV or random generation is recommended."
        )
        idx_specs = spec["indices"]

        if not idx_specs:
            st.info("Define valid indices first.")
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
            idx_opts = list(idx_specs.keys())
            new_params = {}
            old_names = list(cur.keys())

            if n_p == 0:
                st.info("No parameters defined.")

            for p in range(n_p):
                old_name = old_names[p] if p < len(old_names) else f"param_{p+1}"
                old_record = cur.get(old_name, {})
                default_indices = old_record.get("indices", [])

                preview_name = st.session_state.get(f"pname_{p}", old_name)
                preview_indices = st.session_state.get(f"pidxs_{p}", default_indices)
                preview_indices = [idx for idx in preview_indices if idx in idx_specs]
                preview_ne = total_elems(preview_indices, idx_specs)

                preview_modes = ["Manual", "Excel/CSV", "Random"] if preview_ne <= 12 else ["Excel/CSV", "Random"]
                preview_mode = st.session_state.get(f"pmode_{p}", old_record.get("mode", preview_modes[0]))
                if preview_mode == "Excel":
                    preview_mode = "Excel/CSV"
                if preview_mode == "Aleatorio":
                    preview_mode = "Random"
                if preview_mode not in preview_modes:
                    preview_mode = preview_modes[0]

                preview_label = (
                    f"Parameter {p+1}: {preview_name} — "
                    f"{sig(preview_name, preview_indices)} — "
                    f"{preview_ne} element(s) — {preview_mode}"
                )
                expanded = (
                    st.session_state.get("parameter_expander_abierto") == p or
                    (st.session_state.get("parameter_expander_abierto") is None and p == 0)
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
                    p_idxs = col2.multiselect(
                        f"Indices of {pname}",
                        idx_opts,
                        default=default_indices,
                        key=f"pidxs_{p}",
                        on_change=_open_parameter,
                        args=(p,)
                    )

                    if not valid_sym(pname):
                        st.error(f"`{pname}` is not valid.")
                        continue
                    if pname in new_params:
                        st.error(f"`{pname}` is duplicated.")
                        continue

                    ne = total_elems(p_idxs, idx_specs)
                    st.write(f"**Firma:** `{sig(pname, p_idxs)}`")
                    st.write(f"**Total number of elements:** `{ne}`")

                    modes = ["Manual", "Excel/CSV", "Random"] if ne <= 12 else ["Excel/CSV", "Random"]
                    old_mode = old_record.get("mode", modes[0])
                    if old_mode == "Excel":
                        old_mode = "Excel/CSV"
                    if old_mode == "Aleatorio":
                        old_mode = "Random"
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
                    current_values = _initial_param_values(p, pname, p_idxs, idx_specs, old_vals)
                    record = {"indices": p_idxs, "mode": mode, "values": dict(current_values)}

                    if mode == "Manual":
                        if not p_idxs:
                            value = st.number_input(
                                f"Value of {pname}",
                                value=scalar_get(current_values),
                                key=f"pscalar_{p}",
                                on_change=_open_parameter,
                                args=(p,)
                            )
                            record["values"] = _set_param_values(p, scalar_set(value))

                        elif len(p_idxs) == 1:
                            labels = idx_specs[p_idxs[0]]["elements"]
                            df0 = vals_to_df1d(labels, current_values)
                            edited = st.data_editor(
                                df0,
                                use_container_width=True,
                                num_rows="fixed",
                                hide_index=True,
                                disabled=["label"],
                                key=f"pman1d_{p}_{_param_signature_key(pname, p_idxs)}",
                                on_change=_open_parameter,
                                args=(p,)
                            )
                            values = {str((str(row["label"]),)): float(row["value"]) for _, row in edited.iterrows()}
                            record["values"] = _set_param_values(p, values)

                        else:
                            clist = combos(p_idxs, idx_specs)
                            df0 = vals_to_df(p_idxs, clist, current_values)
                            edited = st.data_editor(
                                df0,
                                use_container_width=True,
                                num_rows="fixed",
                                hide_index=True,
                                disabled=list(p_idxs),
                                key=f"pmannd_{p}_{_param_signature_key(pname, p_idxs)}",
                                on_change=_open_parameter,
                                args=(p,)
                            )
                            record["values"] = _set_param_values(p, df_to_vals(edited, p_idxs))

                    elif mode == "Excel/CSV":
                        record["values"] = parameter_template_controls(p, pname, p_idxs, idx_specs, current_values)

                    else:  # Random
                        if not p_idxs:
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
                            clist = combos(p_idxs, idx_specs)
                            if st.button(f"Generate values for {pname}", key=f"pgen_{p}", on_click=_open_parameter, args=(p,)):
                                current_values = _set_param_values(p, rand_vals(clist, lo, hi, intg, seed))
                            record["values"] = dict(current_values)
                            st.dataframe(
                                template_df_for_parameter(p_idxs, idx_specs, record["values"]),
                                use_container_width=True,
                                hide_index=True
                            )

                    st.markdown("#### Parameter preview")
                    st.dataframe(
                        template_df_for_parameter(p_idxs, idx_specs, record["values"]),
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
                        f"Parameter {i+1}: {sig(name, param['indices'])} — "
                        f"{total_elems(param['indices'], idx_specs)} element(s) — {param['mode']}"
                    )
                    with st.expander(title, expanded=False):
                        st.dataframe(
                            template_df_for_parameter(param["indices"], idx_specs, param["values"]),
                            use_container_width=True,
                            hide_index=True
                        )

    # -- VARIABLES --
    with tab_var:
        section_box("Variable Configuration", "Define decision variables, their indices, and their domains.", "Assign each variable the indices that determine its dimensions, then choose its mathematical domain.")
        idx_specs = spec["indices"]

        if not idx_specs:
            st.info("Define valid indices first.")
        else:
            cur = spec["variables"]
            n_v = int(st.number_input("Number of variables", 0, 30, max(1, len(cur)) if cur else 1, step=1, key="num_vars"))
            idx_opts = list(idx_specs.keys())
            dom_opts = ["Binary", "NonNegativeReals", "NonNegativeIntegers"]
            new_vars = {}

            for v in range(n_v):
                st.markdown(f"#### Variable {v+1}")
                old_names = list(cur.keys())
                old_name = old_names[v] if v < len(old_names) else f"x_{v+1}"
                col1, col2, col3 = st.columns([2, 3, 2])
                vname = col1.text_input(f"Name {v+1}", value=old_name, key=f"vname_{v}").strip()
                v_idxs = col2.multiselect(f"Indices of {vname}", idx_opts, default=cur.get(old_name, {}).get("indices", []), key=f"vidxs_{v}")
                old_dom = cur.get(old_name, {}).get("domain", "NonNegativeReals")
                v_dom = col3.selectbox(f"Domain of {vname}", dom_opts, index=dom_opts.index(old_dom if old_dom in dom_opts else "NonNegativeReals"), key=f"vdom_{v}")

                if not valid_sym(vname): st.error(f"`{vname}` is not valid."); continue
                if vname in new_vars: st.error(f"`{vname}` is duplicated."); continue
                new_vars[vname] = {"indices": v_idxs, "domain": v_dom}

            spec["variables"] = new_vars
            if new_vars:
                st.write("**Summary:**")
                st.dataframe(pd.DataFrame([
                    {"Variable": sig(n, v["indices"]), "Domain": DOMAIN_LABELS.get(v["domain"], v["domain"]), "Components": total_elems(v["indices"], idx_specs)}
                    for n, v in new_vars.items()
                ]), use_container_width=True, hide_index=True)

# ============================================================
# SECTION 2: MODEL DEFINITION
# ============================================================
elif section == "Model Definition":
    hero("2. Model Definition", "Build the objective function, constraint families, and mathematical representation.")
    idx_specs = spec["indices"]

    if not idx_specs:
        st.warning("Define at least one index first.")
    elif not spec["variables"]:
        st.warning("Define at least one variable first.")
    else:
        catalog, label_map = object_catalog(spec)
        idx_names = list(idx_specs.keys())
        tab_obj, tab_rest, tab_math = st.tabs(["Objective Function", "Constraints", "Mathematical Model"])

        # -- OBJECTIVE FUNCTION --
        with tab_obj:
            section_box(
                "Objective Function",
                "Combine parameters, variables, constants, and summations to define the single objective.",
                "Every free index in the objective must be eliminated by a summation. Dynamic summation bounds may use expressions such as j+2 or 2*j+1."
            )
            cur_obj = spec.get("objective") or {}
            sense_opts = ["minimize", "maximize"]
            sense = st.radio("Objective sense:", sense_opts, index=sense_opts.index(cur_obj.get("sense", "minimize")), horizontal=True, key="obj_sense", help="Choose whether the objective function is minimized or maximized.")
            old_terms = cur_obj.get("terms", [])
            n_terms = int(st.number_input("Objective terms", 1, 20, max(1, len(old_terms) or 1), step=1, key="n_obj_terms", help="Number of additive/subtractive terms in the objective function."))
            obj_terms = []
            for t in range(n_terms):
                st.markdown(f"#### Objective term {t+1}")
                term = build_term_ui(f"obj_t{t}", t, old_terms[t] if t < len(old_terms) else None, catalog, label_map, idx_names)
                obj_terms.append(term)

            errs = validate_obj(obj_terms, idx_names)
            for e in errs: st.error(e)
            if not errs: st.success("Objective function is structurally consistent.")
            spec["objective"] = {"sense": sense, "terms": obj_terms}

        # -- CONSTRAINTS --
        with tab_rest:
            section_box(
                "Constraint Families",
                "Define indexed constraint families by building the left-hand side, operator, and right-hand side.",
                "Use For all for the free indices of the family. Inside each term, summation bounds can depend on those free indices, for example sum from i=j+2 to N_i."
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
                expanded = (st.session_state.get("constraint_family_expander_abierto") == r or
                            (st.session_state.get("constraint_family_expander_abierto") is None and r == 0))

                with st.expander(fam_label, expanded=expanded):
                    st.markdown(f"### Family {r+1}")
                    cf1, cf2, cf3 = st.columns(3)
                    fname = cf1.text_input(f"Family name {r+1}", value=default_name, key=f"cfname_{r}", on_change=_open_family, args=(r,), help="Use a short symbolic name such as Capacity or Balance.").strip()
                    forall = cf2.multiselect(f"For all indices in {fname}", idx_names, default=(old_fam or {}).get("forall", []), key=f"cfforall_{r}", on_change=_open_family, args=(r,), help="Select the free indices that define this constraint family.")
                    sense_f = cf3.selectbox(f"Operator for {fname}", ["<=", ">=", "="], index=["<=", ">=", "="].index((old_fam or {}).get("sense", "<=")), key=f"cfsense_{r}", on_change=_open_family, args=(r,), help="Choose the relational operator between the left-hand and right-hand sides.")

                    if not valid_sym(fname): st.error(f"`{fname}` is not valid."); continue

                    colL, colR = st.columns(2)
                    old_lhs = (old_fam or {}).get("lhs_terms", [])
                    old_rhs = (old_fam or {}).get("rhs_terms", [])

                    with colL:
                        st.markdown(f"#### LHS of {fname}")
                        n_lhs = int(st.number_input(f"LHS terms for {fname}", 0, 10, len(old_lhs), step=1, key=f"nlhs_{r}", on_change=_open_family, args=(r,)))
                        lhs_terms = [build_term_ui(f"lhs_{r}_{t}", t, old_lhs[t] if t < len(old_lhs) else None, catalog, label_map, idx_names) for t in range(n_lhs)]

                    with colR:
                        st.markdown(f"#### RHS of {fname}")
                        n_rhs = int(st.number_input(f"RHS terms for {fname}", 0, 10, len(old_rhs), step=1, key=f"nrhs_{r}", on_change=_open_family, args=(r,)))
                        rhs_terms = [build_term_ui(f"rhs_{r}_{t}", t, old_rhs[t] if t < len(old_rhs) else None, catalog, label_map, idx_names, default_const_type="constant") for t in range(n_rhs)]

                    family_record = {"name": fname, "forall": forall, "sense": sense_f, "lhs_terms": lhs_terms, "rhs_terms": rhs_terms}
                    st.markdown(f"### Preview — {fname}")
                    st.latex(family_latex(family_record))
                    fam_errs = validate_family(family_record, idx_names)
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
    hero("3. Model Outputs", "Solve the model and inspect the optimal objective value and decision variables.")

    # Validate
    errs = []
    if not spec["objective"]:
        errs.append("No objective function defined.")
    else:
        errs.extend(validate_obj(spec["objective"].get("terms", []), list(spec["indices"].keys())))
    for fam in spec.get("constraints", []):
        errs.extend(validate_family(fam, list(spec["indices"].keys())))
    errs.extend(validate_linearity(spec))
    if not spec["variables"]: errs.append("No variables defined.")
    if not spec["indices"]: errs.append("No indices defined.")

    for e in errs: st.error(e)
    if errs: st.stop()
    st.success("Model specification is valid.")

    tab_solve, tab_vars = st.tabs(["Solve", "Solution Variables"])

    with tab_solve:
        section_box(
            "Solve Model",
            "Validate the model, select an available solver, and compute the solution.",
            "The application currently supports linear models. Products containing more than one decision variable are rejected as nonlinear."
        )
        st.subheader("Solve model")
        solver_label = st.selectbox(
            "Solver",
            list(SOLVER_OPTIONS.keys()),
            index=0,
            help="HiGHS is the recommended option for continuous, integer, and binary linear models."
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

    with tab_vars:
        section_box(
            "Solution Variables",
            "Inspect the value of each decision variable and export the results.",
            "Select one variable for its full solution table or use Nonzero Variables to focus only on active decisions."
        )
        results = spec.get("results")
        model = st.session_state.get("solved_model_object")
        if not results or not model:
            st.info("Solve the model first.")
        else:
            subtab_var, subtab_nz = st.tabs(["Select Variable", "Nonzero Variables"])

            with subtab_var:
                st.subheader("Solution by variable")
                vnames = list(spec["variables"].keys())
                sel = st.selectbox("Variable", vnames)
                df = var_solution_df(model, sel, spec["variables"][sel], spec["indices"])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("Download CSV", data=df.to_csv(index=False).encode(), file_name=f"{sel}_solution.csv", mime="text/csv")

            with subtab_nz:
                st.subheader("Nonzero variables")
                full_df = all_vars_df(model, spec)
                nz_df = full_df[full_df["value"].abs() > 1e-9].reset_index(drop=True) if not full_df.empty else full_df
                if nz_df.empty:
                    st.info("There are no nonzero variables.")
                else:
                    st.dataframe(nz_df, use_container_width=True, hide_index=True)
                st.download_button("Download CSV", data=nz_df.to_csv(index=False).encode(), file_name="nonzero_variables.csv", mime="text/csv")
