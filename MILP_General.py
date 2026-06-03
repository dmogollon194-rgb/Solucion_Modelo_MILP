import streamlit as st
import pandas as pd
import numpy as np
import itertools
from typing import Any
import pyomo.environ as pyo

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Constructor de Modelos Algebraicos", layout="wide")

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
# UTILITIES — EXPRESSIONS
# ============================================================
DOMAIN_LABELS = {"Binary": "Binarias", "NonNegativeReals": "Reales ≥ 0", "NonNegativeIntegers": "Enteras ≥ 0"}

def _fac_latex(f: dict) -> str:
    if f["type"] == "constant":
        v = float(f["value"])
        return str(int(v)) if v == int(v) else f"{v:.2f}"
    n, idxs = f["name"], f["indices"]
    return n if not idxs else rf"{n}_{{{','.join(idxs)}}}"

def term_latex(t: dict) -> str:
    factors = t.get("factors", [])
    body = r" \cdot ".join(_fac_latex(f) for f in factors) if factors else "0"
    for idx in t.get("sum_over", []):
        body = rf"\sum_{{{idx}}}\left({body}\right)"
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
    seen, out = set(), []
    for x in used:
        if x not in seen:
            seen.add(x); out.append(x)
    return [x for x in out if x not in t.get("sum_over", [])]

# ============================================================
# UTILITIES — VALIDATION
# ============================================================
def validate_obj(terms: list[dict]) -> list[str]:
    errs = []
    for i, t in enumerate(terms, 1):
        free = term_free_idxs(t)
        if free:
            errs.append(f"Término FO {i}: índices libres sin sumatoria → {', '.join(free)}")
    return errs

def validate_family(fam: dict) -> list[str]:
    errs = []
    lhs_free = list({x for t in fam.get("lhs_terms", []) for x in term_free_idxs(t)})
    rhs_free = list({x for t in fam.get("rhs_terms", []) for x in term_free_idxs(t)})
    forall = fam.get("forall", [])
    lc, rc = not lhs_free, not rhs_free
    if not lc and not rc:
        if sorted(lhs_free) != sorted(rhs_free):
            errs.append(f"Índices libres LHS {lhs_free} ≠ RHS {rhs_free}")
        if sorted(lhs_free) != sorted(forall):
            errs.append(f"Índices libres {lhs_free} ≠ forall {forall}")
    elif lc and not rc:
        if sorted(rhs_free) != sorted(forall):
            errs.append(f"LHS constante: índices RHS {rhs_free} deben coincidir con forall {forall}")
    elif not lc and rc:
        if sorted(lhs_free) != sorted(forall):
            errs.append(f"RHS constante: índices LHS {lhs_free} deben coincidir con forall {forall}")
    elif lc and rc and forall:
        errs.append(f"Ambos lados constantes pero hay forall {forall}")
    return errs

def validate_linearity(spec: dict) -> list[str]:
    errs = []
    def chk(terms, ctx):
        for i, t in enumerate(terms, 1):
            nv = sum(1 for f in t.get("factors", []) if f["type"] == "object" and f.get("kind") == "variable")
            if nv > 1:
                errs.append(f"{ctx} término {i}: {nv} variables en un producto → no lineal")
    obj = spec.get("objective")
    if obj:
        chk(obj.get("terms", []), "FO")
    for r, fam in enumerate(spec.get("constraints", []), 1):
        name = fam.get("name", f"R{r}")
        chk(fam.get("lhs_terms", []), f"Restricción {name} LHS")
        chk(fam.get("rhs_terms", []), f"Restricción {name} RHS")
    return errs

# ============================================================
# UTILITIES — PYOMO
# ============================================================
_DOMAINS = {"Binary": pyo.Binary, "NonNegativeReals": pyo.NonNegativeReals, "NonNegativeIntegers": pyo.NonNegativeIntegers}

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
    def recurse(pos, local_env):
        if pos == len(t.get("sum_over", [])):
            val = 1
            for f in t.get("factors", []):
                val = val * _get_val(model, f, local_env)
            return (-val) if t.get("sign") == "-" else val
        idx = t["sum_over"][pos]
        return sum(recurse(pos + 1, {**local_env, idx: v}) for v in getattr(model, f"set_{idx}"))
    return recurse(0, dict(env))

def _build_expr(model, terms: list[dict], env: dict):
    return sum(_eval_term(model, t, env) for t in terms) if terms else 0

def build_pyomo_model(spec: dict):
    m = pyo.ConcreteModel()
    idx_specs = spec["indices"]

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

def section_box(subtitle: str, text: str = ""):
    st.markdown(f'<div class="section-box"><b style="font-size:1.1rem">{subtitle}</b>'
                + (f'<p style="color:#d7e6ff;margin-top:6px">{text}</p>' if text else "") + '</div>', unsafe_allow_html=True)

def _rand_controls(key_prefix: str) -> tuple[float, float, bool, int]:
    c1, c2, c3, c4 = st.columns(4)
    lo = c1.number_input("Mínimo", value=0.0, key=f"{key_prefix}_lo")
    hi = c2.number_input("Máximo", value=10.0, key=f"{key_prefix}_hi")
    integer = c3.checkbox("Entero", value=False, key=f"{key_prefix}_int")
    seed = int(c4.number_input("Semilla", value=123, step=1, key=f"{key_prefix}_seed"))
    return lo, hi, integer, seed

def build_factor_ui(t_key: str, f_idx: int, old_factor: dict | None, catalog: list[dict], label_map: dict, default_type="object") -> dict | None:
    cfa, cfb, cfc = st.columns([1.5, 2.5, 2])
    ftype = cfa.selectbox(
        f"Tipo factor {f_idx+1}", ["object", "constant"],
        index=0 if (old_factor or {}).get("type", default_type) == "object" else 1,
        format_func=lambda x: "Parámetro/Variable" if x == "object" else "Constante",
        key=f"{t_key}_ftype_{f_idx}"
    )
    if ftype == "object":
        labels = [o["label"] for o in catalog]
        if not labels:
            st.error("No hay parámetros ni variables disponibles.")
            return None
        default_lbl = labels[0]
        if old_factor and old_factor.get("type") == "object" and old_factor.get("label") in labels:
            default_lbl = old_factor["label"]
        chosen = cfb.selectbox(f"Objeto {f_idx+1}", labels, index=labels.index(default_lbl), key=f"{t_key}_fobj_{f_idx}")
        item = label_map[chosen]
        cfc.write(f"Índices: {', '.join(item['indices']) or 'ninguno'}")
        return {"type": "object", "kind": item["kind"], "name": item["name"], "indices": item["indices"], "label": item["label"]}
    else:
        dval = float((old_factor or {}).get("value", 0.0)) if (old_factor or {}).get("type") == "constant" else 0.0
        val = cfb.number_input(f"Constante {f_idx+1}", value=dval, key=f"{t_key}_fconst_{f_idx}")
        return {"type": "constant", "value": float(val)}

def build_term_ui(t_key: str, t_idx: int, old_term: dict | None, catalog: list[dict], label_map: dict, idx_names: list[str], default_const_type="object") -> dict:
    c1, c2, c3 = st.columns([1, 2, 2])
    old = old_term or {}
    sign = c1.selectbox(f"Signo {t_idx+1}", ["+", "-"], index=0 if old.get("sign", "+") == "+" else 1, key=f"{t_key}_sign")
    n_factors = int(c2.number_input(f"Factores {t_idx+1}", min_value=1, max_value=4, value=max(1, len(old.get("factors", [])) or 2), step=1, key=f"{t_key}_nfac"))
    sum_over = c3.multiselect(f"Sumar sobre {t_idx+1}", idx_names, default=old.get("sum_over", []), key=f"{t_key}_sumover")

    old_factors = old.get("factors", [])
    factors = []
    for fi in range(n_factors):
        f = build_factor_ui(f"{t_key}_f{fi}", fi, old_factors[fi] if fi < len(old_factors) else None, catalog, label_map, default_const_type)
        if f:
            factors.append(f)

    term = {"sign": sign, "factors": factors, "sum_over": sum_over}
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

# ============================================================
# SIDEBAR
# ============================================================
n_idx = len(spec["indices"])
n_var = count_expanded(spec, "variables")
n_con = count_expanded(spec, "constraints")

st.sidebar.markdown("""
<div style="padding:.4rem 0 1rem;border-bottom:1px solid rgba(61,132,255,.18);margin-bottom:1rem">
    <div style="font-size:1.45rem;font-weight:800;color:#fff;margin-bottom:.35rem">Navegación</div>
    <div style="color:#b9c9e8;font-size:.92rem">Explora cada etapa de construcción y solución.</div>
</div>""", unsafe_allow_html=True)

section = st.sidebar.radio("Ir a:", ["Ingreso de información", "Definición del modelo", "Salidas del modelo"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown('<div style="font-size:1.05rem;font-weight:800;color:#fff;margin-bottom:.8rem">Estado actual</div>', unsafe_allow_html=True)

def _sb_kpi(label, value):
    return f"""<div style="background:linear-gradient(135deg,rgba(8,22,55,.95),rgba(3,10,28,.98));
        border:1px solid rgba(61,132,255,.22);border-radius:14px;padding:12px 14px;margin-bottom:10px">
        <div style="font-size:.88rem;color:#cfe0ff;font-weight:700">{label}</div>
        <div style="font-size:1.8rem;color:#fff;font-weight:800">{value}</div></div>"""

c1, c2 = st.sidebar.columns(2)
c1.markdown(_sb_kpi("Índices", n_idx), unsafe_allow_html=True)
c2.markdown(_sb_kpi("Variables", n_var), unsafe_allow_html=True)
st.sidebar.markdown(_sb_kpi("Restricciones definidas", n_con), unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
st.title("Solucionador de Modelos Lineales")
st.caption("Aplicación para solucionar modelos lineales con un solo objetivo.")

# ============================================================
# SECTION 1: INGRESO DE INFORMACIÓN
# ============================================================
if section == "Ingreso de información":
    hero("1. Ingreso de información", "Define los índices, parámetros y variables del modelo.")

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Índices", len(spec["indices"]))
    with c2: kpi_card("Parámetros", len(spec["parameters"]))
    with c3: kpi_card("Variables", len(spec["variables"]))

    st.markdown("<br>", unsafe_allow_html=True)
    tab_ind, tab_par, tab_var = st.tabs(["Índices", "Parámetros", "Variables"])

    # -- ÍNDICES --
    with tab_ind:
        section_box("Configuración de índices", "Define los conjuntos base del modelo.")
        n = st.number_input("Número de índices", 1, 10, max(1, len(spec["indices"]) or 3), step=1, key="num_indices")
        existing_names = list(spec["indices"].keys())

        idx_specs_new, errors = {}, []
        used = set()
        for r in range(int(n)):
            default_name = existing_names[r] if r < len(existing_names) else f"idx_{r+1}"
            default_size = spec["indices"].get(default_name, {}).get("size", 3)
            col1, col2 = st.columns(2)
            name = col1.text_input(f"Nombre {r+1}", value=default_name, key=f"idx_name_{r}").strip()
            size = int(col2.number_input(f"Tamaño de {name or f'idx {r+1}'}", 1, 1000, int(default_size), step=1, key=f"idx_size_{r}"))
            if not valid_sym(name):
                errors.append(f"`{name}` no es nombre válido.")
            elif name in used:
                errors.append(f"Índice `{name}` repetido.")
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
                st.write("**Vista previa:**")
                st.dataframe(pd.DataFrame([
                    {"Índice": n, "Tamaño": s["size"], "Elementos": ", ".join(s["elements"])}
                    for n, s in idx_specs_new.items()
                ]), use_container_width=True, hide_index=True)

    # -- PARÁMETROS --
    with tab_par:
        section_box("Configuración de parámetros", "Escalares, vectores, matrices o tensores.")
        idx_specs = spec["indices"]

        if not idx_specs:
            st.info("Primero define índices válidos.")
        else:
            cur = spec["parameters"]
            n_p = int(st.number_input("Número de parámetros", 0, 30, max(1, len(cur)) if cur else 1, step=1, key="num_params"))
            idx_opts = list(idx_specs.keys())
            new_params = {}

            for p in range(n_p):
                st.markdown(f"#### Parámetro {p+1}")
                old_names = list(cur.keys())
                old_name = old_names[p] if p < len(old_names) else f"param_{p+1}"
                col1, col2 = st.columns([2, 3])
                pname = col1.text_input(f"Nombre {p+1}", value=old_name, key=f"pname_{p}").strip()
                p_idxs = col2.multiselect(f"Índices de {pname}", idx_opts, default=cur.get(old_name, {}).get("indices", []), key=f"pidxs_{p}")

                if not valid_sym(pname): st.error(f"`{pname}` no válido."); continue
                if pname in new_params: st.error(f"`{pname}` repetido."); continue

                ne = total_elems(p_idxs, idx_specs)
                st.write(f"**Firma:** `{sig(pname, p_idxs)}` — **Elementos:** `{ne}`")
                modes = ["Manual", "Aleatorio"] if ne <= 12 else ["Aleatorio"]
                old_mode = cur.get(old_name, {}).get("mode", modes[0])
                mode = st.radio(f"Modo {pname}", modes, index=modes.index(old_mode if old_mode in modes else modes[0]), horizontal=True, key=f"pmode_{p}")
                old_vals = cur.get(old_name, {}).get("values", {})
                record = {"indices": p_idxs, "mode": mode, "values": {}}

                if not p_idxs:
                    if mode == "Manual":
                        record["values"] = scalar_set(st.number_input(f"Valor {pname}", value=scalar_get(old_vals), key=f"pscalar_{p}"))
                    else:
                        lo, hi, intg, seed = _rand_controls(f"ps_{p}")
                        if lo > hi: st.error("Mínimo > máximo."); continue
                        if st.button(f"Generar {pname}", key=f"pgen_{p}"):
                            st.session_state[f"_pvals_{pname}"] = rand_scalar(lo, hi, intg, seed)
                        gv = st.session_state.get(f"_pvals_{pname}", old_vals or rand_scalar(lo, hi, intg, seed))
                        record["values"] = gv
                        st.write(f"Valor: **{scalar_get(gv):.4f}**")
                elif len(p_idxs) == 1:
                    labels = idx_specs[p_idxs[0]]["elements"]
                    clist = combos(p_idxs, idx_specs)
                    if mode == "Manual":
                        df0 = vals_to_df1d(labels, old_vals)
                        edited = st.data_editor(df0, use_container_width=True, num_rows="fixed", hide_index=True, disabled=["label"], key=f"pman1d_{p}")
                        record["values"] = df_to_vals(edited.rename(columns={"label": p_idxs[0]}), p_idxs) if "label" not in edited else {str((row["label"],)): float(row["value"]) for _, row in edited.iterrows()}
                    else:
                        lo, hi, intg, seed = _rand_controls(f"p1d_{p}")
                        if lo > hi: st.error("Mínimo > máximo."); continue
                        if st.button(f"Generar {pname}", key=f"pgen1d_{p}"):
                            st.session_state[f"_pvals_{pname}"] = rand_vals(clist, lo, hi, intg, seed)
                        gv = st.session_state.get(f"_pvals_{pname}", old_vals or rand_vals(clist, lo, hi, intg, seed))
                        record["values"] = gv
                        st.dataframe(vals_to_df1d(labels, gv), use_container_width=True, hide_index=True)
                else:
                    clist = combos(p_idxs, idx_specs)
                    if mode == "Manual":
                        df0 = vals_to_df(p_idxs, clist, old_vals)
                        edited = st.data_editor(df0, use_container_width=True, num_rows="fixed", hide_index=True, disabled=list(p_idxs), key=f"pmannd_{p}")
                        record["values"] = df_to_vals(edited, p_idxs)
                    else:
                        lo, hi, intg, seed = _rand_controls(f"pnd_{p}")
                        if lo > hi: st.error("Mínimo > máximo."); continue
                        if st.button(f"Generar {pname}", key=f"pgennd_{p}"):
                            st.session_state[f"_pvals_{pname}"] = rand_vals(clist, lo, hi, intg, seed)
                        gv = st.session_state.get(f"_pvals_{pname}", old_vals or rand_vals(clist, lo, hi, intg, seed))
                        record["values"] = gv
                        st.dataframe(vals_to_df(p_idxs, clist, gv), use_container_width=True, hide_index=True)

                new_params[pname] = record

            spec["parameters"] = new_params
            if new_params:
                st.write("**Resumen:**")
                st.dataframe(pd.DataFrame([
                    {"Parámetro": sig(n, v["indices"]), "Modo": v["mode"], "Elementos": total_elems(v["indices"], idx_specs)}
                    for n, v in new_params.items()
                ]), use_container_width=True, hide_index=True)

    # -- VARIABLES --
    with tab_var:
        section_box("Configuración de variables", "Variables de decisión y su dominio.")
        idx_specs = spec["indices"]

        if not idx_specs:
            st.info("Primero define índices válidos.")
        else:
            cur = spec["variables"]
            n_v = int(st.number_input("Número de variables", 0, 30, max(1, len(cur)) if cur else 1, step=1, key="num_vars"))
            idx_opts = list(idx_specs.keys())
            dom_opts = ["Binary", "NonNegativeReals", "NonNegativeIntegers"]
            new_vars = {}

            for v in range(n_v):
                st.markdown(f"#### Variable {v+1}")
                old_names = list(cur.keys())
                old_name = old_names[v] if v < len(old_names) else f"x_{v+1}"
                col1, col2, col3 = st.columns([2, 3, 2])
                vname = col1.text_input(f"Nombre {v+1}", value=old_name, key=f"vname_{v}").strip()
                v_idxs = col2.multiselect(f"Índices de {vname}", idx_opts, default=cur.get(old_name, {}).get("indices", []), key=f"vidxs_{v}")
                old_dom = cur.get(old_name, {}).get("domain", "NonNegativeReals")
                v_dom = col3.selectbox(f"Dominio {vname}", dom_opts, index=dom_opts.index(old_dom if old_dom in dom_opts else "NonNegativeReals"), key=f"vdom_{v}")

                if not valid_sym(vname): st.error(f"`{vname}` no válido."); continue
                if vname in new_vars: st.error(f"`{vname}` repetido."); continue
                new_vars[vname] = {"indices": v_idxs, "domain": v_dom}

            spec["variables"] = new_vars
            if new_vars:
                st.write("**Resumen:**")
                st.dataframe(pd.DataFrame([
                    {"Variable": sig(n, v["indices"]), "Dominio": DOMAIN_LABELS.get(v["domain"], v["domain"]), "Componentes": total_elems(v["indices"], idx_specs)}
                    for n, v in new_vars.items()
                ]), use_container_width=True, hide_index=True)

# ============================================================
# SECTION 2: DEFINICIÓN DEL MODELO
# ============================================================
elif section == "Definición del modelo":
    hero("2. Definición del modelo", "Función objetivo, restricciones y modelo matemático.")
    idx_specs = spec["indices"]

    if not idx_specs:
        st.warning("Primero define al menos un índice.")
    elif not spec["variables"]:
        st.warning("Primero define al menos una variable.")
    else:
        catalog, label_map = object_catalog(spec)
        idx_names = list(idx_specs.keys())
        tab_obj, tab_rest, tab_math = st.tabs(["Función Objetivo", "Restricciones", "Modelo matemático"])

        # -- FUNCIÓN OBJETIVO --
        with tab_obj:
            cur_obj = spec.get("objective") or {}
            sense_opts = ["minimize", "maximize"]
            sense = st.radio("Objetivo:", sense_opts, index=sense_opts.index(cur_obj.get("sense", "minimize")), horizontal=True, key="obj_sense")
            old_terms = cur_obj.get("terms", [])
            n_terms = int(st.number_input("Términos en la FO", 1, 20, max(1, len(old_terms) or 1), step=1, key="n_obj_terms"))
            obj_terms = []
            for t in range(n_terms):
                st.markdown(f"#### Término FO {t+1}")
                term = build_term_ui(f"obj_t{t}", t, old_terms[t] if t < len(old_terms) else None, catalog, label_map, idx_names)
                obj_terms.append(term)

            errs = validate_obj(obj_terms)
            for e in errs: st.error(e)
            if not errs: st.success("Función objetivo estructuralmente consistente.")
            spec["objective"] = {"sense": sense, "terms": obj_terms}

        # -- RESTRICCIONES --
        with tab_rest:
            old_fams = spec.get("constraints", [])
            n_fams = int(st.number_input("Familias de restricciones", 0, 30, len(old_fams), step=1, key="n_fams"))
            new_fams = []

            if n_fams == 0:
                st.info("Sin restricciones definidas.")

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
                fam_label = f"Familia {r+1}: {preview['name']} — {family_latex(preview)}"
                expanded = (st.session_state.get("constraint_family_expander_abierto") == r or
                            (st.session_state.get("constraint_family_expander_abierto") is None and r == 0))

                with st.expander(fam_label, expanded=expanded):
                    st.markdown(f"### Familia {r+1}")
                    cf1, cf2, cf3 = st.columns(3)
                    fname = cf1.text_input(f"Nombre familia {r+1}", value=default_name, key=f"cfname_{r}", on_change=_open_family, args=(r,)).strip()
                    forall = cf2.multiselect(f"Para todo en {fname}", idx_names, default=(old_fam or {}).get("forall", []), key=f"cfforall_{r}", on_change=_open_family, args=(r,))
                    sense_f = cf3.selectbox(f"Operador {fname}", ["<=", ">=", "="], index=["<=", ">=", "="].index((old_fam or {}).get("sense", "<=")), key=f"cfsense_{r}", on_change=_open_family, args=(r,))

                    if not valid_sym(fname): st.error(f"`{fname}` no válido."); continue

                    colL, colR = st.columns(2)
                    old_lhs = (old_fam or {}).get("lhs_terms", [])
                    old_rhs = (old_fam or {}).get("rhs_terms", [])

                    with colL:
                        st.markdown(f"#### LHS de {fname}")
                        n_lhs = int(st.number_input(f"Términos LHS {fname}", 0, 10, len(old_lhs), step=1, key=f"nlhs_{r}", on_change=_open_family, args=(r,)))
                        lhs_terms = [build_term_ui(f"lhs_{r}_{t}", t, old_lhs[t] if t < len(old_lhs) else None, catalog, label_map, idx_names) for t in range(n_lhs)]

                    with colR:
                        st.markdown(f"#### RHS de {fname}")
                        n_rhs = int(st.number_input(f"Términos RHS {fname}", 0, 10, len(old_rhs), step=1, key=f"nrhs_{r}", on_change=_open_family, args=(r,)))
                        rhs_terms = [build_term_ui(f"rhs_{r}_{t}", t, old_rhs[t] if t < len(old_rhs) else None, catalog, label_map, idx_names, default_const_type="constant") for t in range(n_rhs)]

                    family_record = {"name": fname, "forall": forall, "sense": sense_f, "lhs_terms": lhs_terms, "rhs_terms": rhs_terms}
                    st.markdown(f"### Vista previa — {fname}")
                    st.latex(family_latex(family_record))
                    fam_errs = validate_family(family_record)
                    for e in fam_errs: st.error(e)
                    if not fam_errs: st.success("Familia estructuralmente consistente.")
                    new_fams.append(family_record)

            spec["constraints"] = new_fams

            if new_fams:
                st.markdown("---")
                st.markdown("### Familias guardadas")
                for i, fam in enumerate(new_fams):
                    with st.expander(f"Familia {i+1}: {fam.get('name')} — {family_latex(fam)}", expanded=False):
                        st.latex(family_latex(fam))

        # -- MODELO MATEMÁTICO --
        with tab_math:
            st.markdown("### Modelo estructurado")
            obj = spec.get("objective")
            if obj:
                symbol = r"\min" if obj["sense"] == "minimize" else r"\max"
                st.latex(rf"{symbol}\ Z = {expr_latex(obj['terms'])}")
            else:
                st.info("Sin función objetivo definida.")
            st.markdown("**Sujeto a:**")
            if not spec["constraints"]:
                st.info("Sin restricciones definidas.")
            else:
                for fam in spec["constraints"]:
                    st.latex(family_latex(fam))

# ============================================================
# SECTION 3: SALIDAS DEL MODELO
# ============================================================
elif section == "Salidas del modelo":
    hero("3. Resultados", "Solución óptima y configuración de las variables.")

    # Validate
    errs = []
    if not spec["objective"]:
        errs.append("Sin función objetivo.")
    else:
        errs.extend(validate_obj(spec["objective"].get("terms", [])))
    for fam in spec.get("constraints", []):
        errs.extend(validate_family(fam))
    errs.extend(validate_linearity(spec))
    if not spec["variables"]: errs.append("Sin variables definidas.")
    if not spec["indices"]: errs.append("Sin índices definidos.")

    for e in errs: st.error(e)
    if errs: st.stop()
    st.success("Especificación válida.")

    tab_solve, tab_vars = st.tabs(["Resolver", "Variables solución"])

    with tab_solve:
        st.subheader("Resolver modelo")
        if st.button("Resolver modelo", type="primary"):
            try:
                model = build_pyomo_model(spec)
                solver = pyo.SolverFactory("appsi_highs")
                result = solver.solve(model)
                spec["results"] = {
                    "solver_name": "appsi_highs",
                    "termination_condition": str(result.solver.termination_condition),
                    "status": str(result.solver.status),
                    "objective_value": pyo.value(model.OBJ),
                }
                st.session_state["solved_model_object"] = model
                st.success("Modelo resuelto correctamente.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        results = spec.get("results")
        model = st.session_state.get("solved_model_object")
        if not results or not model:
            st.info("Aún no has resuelto el modelo.")
        else:
            c1, c2 = st.columns(2)
            with c1: kpi_card("Status", results["status"])
            with c2: kpi_card("Termination", results["termination_condition"])
            kpi_card("Valor óptimo", f"{results['objective_value']:,.6f}")

    with tab_vars:
        results = spec.get("results")
        model = st.session_state.get("solved_model_object")
        if not results or not model:
            st.info("Primero resuelve el modelo.")
        else:
            subtab_var, subtab_nz = st.tabs(["Seleccionar variable", "Variables no nulas"])

            with subtab_var:
                st.subheader("Solución por variable")
                vnames = list(spec["variables"].keys())
                sel = st.selectbox("Variable", vnames)
                df = var_solution_df(model, sel, spec["variables"][sel], spec["indices"])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("Descargar CSV", data=df.to_csv(index=False).encode(), file_name=f"{sel}_solucion.csv", mime="text/csv")

            with subtab_nz:
                st.subheader("Variables no nulas")
                full_df = all_vars_df(model, spec)
                nz_df = full_df[full_df["value"].abs() > 1e-9].reset_index(drop=True) if not full_df.empty else full_df
                if nz_df.empty:
                    st.info("No hay variables no nulas.")
                else:
                    st.dataframe(nz_df, use_container_width=True, hide_index=True)
                st.download_button("Descargar CSV", data=nz_df.to_csv(index=False).encode(), file_name="variables_no_nulas.csv", mime="text/csv")
