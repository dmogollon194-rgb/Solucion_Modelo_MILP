"""
Constructor de modelos algebraicos
Versión pulida: código más limpio, sin redundancias, mejor UX.
"""

import itertools
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(page_title="Constructor de modelos algebraicos", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stMetric > div { background: #f8f9fa; border-radius: 8px; padding: .5rem; }
    div[data-testid="stSidebarNav"] { font-size: .9rem; }
    h2, h3 { margin-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔢 Constructor de modelos algebraicos")
st.caption("Índices · Parámetros · Variables · Función objetivo · Restricciones · Resolución")


# ─────────────────────────────────────────────
# ESTADO INICIAL
# ─────────────────────────────────────────────

def _init() -> None:
    if "model_spec" not in st.session_state:
        st.session_state["model_spec"] = {
            "indices": {},
            "parameters": {},
            "variables": {},
            "objective": None,
            "constraints": [],
            "results": None,
        }

_init()


# ─────────────────────────────────────────────
# UTILIDADES GENERALES
# ─────────────────────────────────────────────

def is_valid_symbol(name: str) -> bool:
    name = (name or "").strip()
    if not name or not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name[1:])


def build_elements(size: int, prefix: str) -> List[str]:
    return [f"{prefix}{t}" for t in range(1, size + 1)]


def cartesian(index_names: List[str], index_specs: Dict) -> List[Tuple]:
    if not index_names:
        return [tuple()]
    return list(itertools.product(*[index_specs[i]["elements"] for i in index_names]))


def total_elements(index_names: List[str], index_specs: Dict) -> int:
    n = 1
    for i in index_names:
        n *= int(index_specs[i]["size"])
    return n


def pretty_sig(name: str, indices: List[str]) -> str:
    return name if not indices else f"{name}[{', '.join(indices)}]"


# ── Values dict helpers ──

def scalar_set(val: float) -> Dict:   return {"__scalar__": float(val)}
def scalar_get(d: Dict, default=0.0): return float(d.get("__scalar__", default))

def df_to_values(df: pd.DataFrame, index_names: List[str]) -> Dict:
    if not index_names:
        return scalar_set(float(df["value"].iloc[0]))
    return {str(tuple(str(row[i]) for i in index_names)): float(row["value"]) for _, row in df.iterrows()}

def values_to_df(index_names: List[str], combos: List[Tuple], values: Dict) -> pd.DataFrame:
    if not index_names:
        return pd.DataFrame({"value": [scalar_get(values)]})
    rows = [{**dict(zip(index_names, c)), "value": float(values.get(str(c), 0.0))} for c in combos]
    return pd.DataFrame(rows)

def values_1d_to_df(labels: List[str], values: Dict) -> pd.DataFrame:
    return pd.DataFrame([{"label": l, "value": float(values.get(str((l,)), 0.0))} for l in labels])

def df_1d_to_values(df: pd.DataFrame, labels: List[str]) -> Dict:
    return {str((l,)): float(df.loc[df["label"] == l, "value"].iloc[0]) for l in labels}

def random_values(combos: List[Tuple], low: float, high: float, integer: bool, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    fn  = (lambda: int(rng.integers(int(low), int(high) + 1))) if integer else (lambda: float(rng.uniform(low, high)))
    return {str(c): float(fn()) for c in combos}

def random_scalar(low: float, high: float, integer: bool, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    val = int(rng.integers(int(low), int(high) + 1)) if integer else float(rng.uniform(low, high))
    return scalar_set(val)


# ─────────────────────────────────────────────
# UTILIDADES LATEX / TEXTO
# ─────────────────────────────────────────────

def _num(v: float) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:.2f}"

def factor_to_latex(fac: Dict) -> str:
    if fac["type"] == "constant":
        return _num(fac["value"])
    name, idxs = fac["name"], fac["indices"]
    return name if not idxs else rf"{name}_{{{','.join(idxs)}}}"

def term_to_latex(term: Dict) -> str:
    factors = term.get("factors", [])
    body    = r" \cdot ".join(factor_to_latex(f) for f in factors) if factors else "0"
    sums    = term.get("sum_over", [])
    if sums:
        body = " ".join(rf"\sum_{{{i}}}" for i in sums) + rf"\left({body}\right)"
    return f"- {body}" if term.get("sign") == "-" else f"+ {body}"

def expr_to_latex(terms: List[Dict]) -> str:
    if not terms:
        return "0"
    raw = " ".join(term_to_latex(t) for t in terms).strip()
    return raw[2:] if raw.startswith("+ ") else raw

def constraint_to_latex(fam: Dict) -> str:
    lhs   = expr_to_latex(fam.get("lhs_terms", []))
    rhs   = expr_to_latex(fam.get("rhs_terms", []))
    sense = {"<=": r"\leq", ">=": r"\geq", "=": "="}[fam.get("sense", "<=")]
    fa    = fam.get("forall", [])
    tail  = (r"\quad \forall " + ", ".join(fa)) if fa else ""
    return f"{lhs} {sense} {rhs}{tail}"


# ─────────────────────────────────────────────
# LÓGICA DE ÍNDICES EN TÉRMINOS
# ─────────────────────────────────────────────

def _ordered_unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def term_used_indices(term: Dict) -> List[str]:
    return _ordered_unique(i for f in term["factors"] if f["type"] == "object" for i in f["indices"])

def term_free_indices(term: Dict) -> List[str]:
    return [i for i in term_used_indices(term) if i not in term.get("sum_over", [])]


# ─────────────────────────────────────────────
# VALIDACIONES
# ─────────────────────────────────────────────

def validate_objective(terms: List[Dict]) -> List[str]:
    errors = []
    for i, t in enumerate(terms, 1):
        free = term_free_indices(t)
        if free:
            errors.append(f"Término FO {i} tiene índices libres: {', '.join(free)}. Deben estar cubiertos por sumatorias.")
    return errors

def validate_constraint(fam: Dict) -> List[str]:
    errors = []
    lhs_free = _ordered_unique(i for t in fam.get("lhs_terms", []) for i in term_free_indices(t))
    rhs_free = _ordered_unique(i for t in fam.get("rhs_terms", []) for i in term_free_indices(t))
    forall   = fam.get("forall", [])

    if lhs_free and rhs_free:
        if lhs_free != rhs_free:
            errors.append(f"Índices libres LHS {lhs_free} ≠ RHS {rhs_free}.")
        if lhs_free != forall:
            errors.append(f"Índices libres {lhs_free} ≠ para-todo {forall}.")
    elif not lhs_free and rhs_free:
        if rhs_free != forall:
            errors.append(f"LHS constante; índices libres RHS {rhs_free} deben coincidir con para-todo {forall}.")
    elif lhs_free and not rhs_free:
        if lhs_free != forall:
            errors.append(f"RHS constante; índices libres LHS {lhs_free} deben coincidir con para-todo {forall}.")
    else:
        if forall:
            errors.append(f"Ambos lados constantes; no debería haber para-todo {forall}.")
    return errors

def validate_linearity(spec: Dict) -> List[str]:
    errors = []
    def _check(terms, ctx):
        for i, t in enumerate(terms, 1):
            n = sum(1 for f in t.get("factors", []) if f["type"] == "object" and f.get("kind") == "variable")
            if n > 1:
                errors.append(f"{ctx} término {i}: {n} factores variables → no lineal.")
    obj = spec.get("objective")
    if obj:
        _check(obj.get("terms", []), "FO")
    for r, fam in enumerate(spec.get("constraints", []), 1):
        name = fam.get("name", f"R{r}")
        _check(fam.get("lhs_terms", []), f"R {name} LHS")
        _check(fam.get("rhs_terms", []), f"R {name} RHS")
    return errors


# ─────────────────────────────────────────────
# PYOMO
# ─────────────────────────────────────────────

_DOMAINS = {
    "Binary":              pyo.Binary,
    "NonNegativeReals":    pyo.NonNegativeReals,
    "NonNegativeIntegers": pyo.NonNegativeIntegers,
}
_DOMAIN_LABELS = {
    "Binary": "Binaria",
    "NonNegativeReals": "Real ≥ 0",
    "NonNegativeIntegers": "Entera ≥ 0",
}

def _get_comp(model, fac: Dict, env: Dict):
    if fac["type"] == "constant":
        return float(fac["value"])
    name, idxs, kind = fac["name"], fac["indices"], fac["kind"]
    comp = getattr(model, f"{'par' if kind == 'parameter' else 'var'}_{name}")
    if not idxs:
        return comp
    key = tuple(env[i] for i in idxs)
    return comp[key[0]] if len(key) == 1 else comp[key]

def _eval_term(model, term: Dict, env: Dict):
    val = 1
    for f in term.get("factors", []):
        val = val * _get_comp(model, f, env)
    if not term.get("factors"):
        val = 0
    def _sum(pos, local_env):
        sum_over = term.get("sum_over", [])
        if pos == len(sum_over):
            sign = -1 if term.get("sign") == "-" else 1
            return sign * val if pos == 0 else sign * _base(local_env)
        idx = sum_over[pos]
        return sum(_sum(pos + 1, {**local_env, idx: v}) for v in getattr(model, f"set_{idx}"))
    # rebuild with proper env
    def _base(local_env):
        v = 1
        for f in term.get("factors", []):
            v = v * _get_comp(model, f, local_env)
        return v if term.get("factors") else 0
    def recurse(pos, local_env):
        so = term.get("sum_over", [])
        if pos == len(so):
            raw = _base(local_env)
            return -raw if term.get("sign") == "-" else raw
        idx = so[pos]
        return sum(recurse(pos + 1, {**local_env, idx: v}) for v in getattr(model, f"set_{idx}"))
    return recurse(0, dict(env))

def _build_expr(model, terms: List[Dict], env: Dict):
    return sum(_eval_term(model, t, env) for t in terms) if terms else 0

def build_pyomo_model(spec: Dict):
    m = pyo.ConcreteModel()
    idx_specs = spec["indices"]

    # Sets
    for name, s in idx_specs.items():
        setattr(m, f"set_{name}", pyo.Set(initialize=s["elements"], ordered=True))

    # Parameters
    for pname, ps in spec["parameters"].items():
        idxs, vals = ps["indices"], ps["values"]
        if not idxs:
            comp = pyo.Param(initialize=float(vals["__scalar__"]))
        else:
            combos = cartesian(idxs, idx_specs)
            init   = {}
            for c in combos:
                k = c[0] if len(c) == 1 else c
                init[k] = float(vals.get(str(c), 0.0))
            comp = pyo.Param(*[getattr(m, f"set_{i}") for i in idxs], initialize=init)
        setattr(m, f"par_{pname}", comp)

    # Variables
    for vname, vs in spec["variables"].items():
        idxs   = vs["indices"]
        domain = _DOMAINS[vs["domain"]]
        comp   = pyo.Var(domain=domain) if not idxs else pyo.Var(*[getattr(m, f"set_{i}") for i in idxs], domain=domain)
        setattr(m, f"var_{vname}", comp)

    # Objective
    obj = spec["objective"]
    obj_expr  = _build_expr(m, obj["terms"], {})
    obj_sense = pyo.minimize if obj["sense"] == "minimize" else pyo.maximize
    m.OBJ = pyo.Objective(expr=obj_expr, sense=obj_sense)

    # Constraints
    for c_idx, fam in enumerate(spec.get("constraints", []), 1):
        fname  = fam.get("name", f"R{c_idx}")
        forall = fam.get("forall", [])
        sense  = fam.get("sense", "<=")

        def _expr(m2, lhs_t, rhs_t, env, sense2):
            lhs = _build_expr(m2, lhs_t, env)
            rhs = _build_expr(m2, rhs_t, env)
            return (lhs <= rhs) if sense2 == "<=" else (lhs >= rhs) if sense2 == ">=" else (lhs == rhs)

        if not forall:
            setattr(m, f"con_{fname}",
                    pyo.Constraint(expr=_expr(m, fam["lhs_terms"], fam["rhs_terms"], {}, sense)))
        else:
            sets = [getattr(m, f"set_{i}") for i in forall]
            lhs_t, rhs_t, fa, s = fam["lhs_terms"], fam["rhs_terms"], forall, sense
            def _rule(m2, *args, _lhs=lhs_t, _rhs=rhs_t, _fa=fa, _s=s):
                env = dict(zip(_fa, args))
                return _expr(m2, _lhs, _rhs, env, _s)
            setattr(m, f"con_{fname}", pyo.Constraint(*sets, rule=_rule))

    return m

def var_to_df(model, vname: str, vspec: Dict, idx_specs: Dict) -> pd.DataFrame:
    comp = getattr(model, f"var_{vname}")
    idxs = vspec["indices"]
    if not idxs:
        return pd.DataFrame({"variable": [vname], "value": [pyo.value(comp)]})
    rows = []
    for c in cartesian(idxs, idx_specs):
        val = pyo.value(comp[c[0]] if len(c) == 1 else comp[c])
        rows.append({**dict(zip(idxs, c)), "value": val})
    return pd.DataFrame(rows)

def all_vars_df(model, spec: Dict) -> pd.DataFrame:
    frames = []
    for vname, vspec in spec["variables"].items():
        df = var_to_df(model, vname, vspec, spec["indices"])
        df.insert(0, "variable", vname)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def nonzero_df(model, spec: Dict, tol=1e-9) -> pd.DataFrame:
    df = all_vars_df(model, spec)
    return df[df["value"].abs() > tol].reset_index(drop=True) if not df.empty else df


# ─────────────────────────────────────────────
# CATÁLOGO DE OBJETOS (parámetros + variables)
# ─────────────────────────────────────────────

def object_catalog(spec: Dict) -> List[Dict]:
    items = []
    for name, s in spec["parameters"].items():
        lbl = pretty_sig(name, s["indices"])
        items.append({"kind": "parameter", "name": name, "indices": s["indices"], "label": lbl})
    for name, s in spec["variables"].items():
        lbl = pretty_sig(name, s["indices"])
        items.append({"kind": "variable",  "name": name, "indices": s["indices"], "label": lbl})
    return items


# ─────────────────────────────────────────────
# WIDGET REUTILIZABLE: UN FACTOR
# ─────────────────────────────────────────────

def factor_widget(key_prefix: str, catalog: List[Dict], old_factor: Dict | None) -> Dict:
    """Renderiza los controles de un factor y devuelve el dict del factor."""
    label_to_item = {o["label"]: o for o in catalog}
    cat_labels    = [o["label"] for o in catalog]

    old_type = (old_factor or {}).get("type", "object")
    c1, c2, c3 = st.columns([1.5, 2.5, 2])

    with c1:
        ftype = st.selectbox(
            "Tipo",
            ["object", "constant"],
            index=0 if old_type == "object" else 1,
            format_func=lambda x: "Param/Var" if x == "object" else "Constante",
            key=f"{key_prefix}_type",
        )

    if ftype == "object":
        if not cat_labels:
            st.error("Sin objetos disponibles.")
            return {"type": "object", "kind": "?", "name": "?", "indices": [], "label": "?"}
        default_lbl = cat_labels[0]
        if old_factor and old_factor.get("type") == "object":
            lbl = old_factor.get("label", default_lbl)
            if lbl in cat_labels:
                default_lbl = lbl
        with c2:
            chosen = st.selectbox("Objeto", cat_labels, index=cat_labels.index(default_lbl), key=f"{key_prefix}_obj")
        item = label_to_item[chosen]
        with c3:
            st.caption(f"Índices: {', '.join(item['indices']) or 'ninguno'}")
        return {"type": "object", "kind": item["kind"], "name": item["name"],
                "indices": item["indices"], "label": item["label"]}
    else:
        default_val = float((old_factor or {}).get("value", 1.0))
        with c2:
            val = st.number_input("Valor", value=default_val, key=f"{key_prefix}_const")
        return {"type": "constant", "value": float(val)}


# ─────────────────────────────────────────────
# WIDGET REUTILIZABLE: UN TÉRMINO
# ─────────────────────────────────────────────

def term_widget(key_prefix: str, catalog: List[Dict], index_names: List[str],
                old_term: Dict | None, show_preview=True) -> Dict:
    """Renderiza los controles de un término y devuelve el dict del término."""
    old = old_term or {}
    c1, c2, c3 = st.columns([1, 2, 2])

    with c1:
        sign = st.selectbox("Signo", ["+", "-"],
                            index=0 if old.get("sign", "+") == "+" else 1,
                            key=f"{key_prefix}_sign")
    with c2:
        n_factors = st.number_input("Factores", min_value=1, max_value=4,
                                    value=max(1, len(old.get("factors", []))),
                                    step=1, key=f"{key_prefix}_nfac")
    with c3:
        sum_over = st.multiselect("Sumar sobre", index_names,
                                  default=[i for i in old.get("sum_over", []) if i in index_names],
                                  key=f"{key_prefix}_sum")

    factors = []
    for fi in range(int(n_factors)):
        old_fac = old.get("factors", [])[fi] if fi < len(old.get("factors", [])) else None
        factors.append(factor_widget(f"{key_prefix}_f{fi}", catalog, old_fac))

    term = {"sign": sign, "factors": factors, "sum_over": sum_over}

    if show_preview:
        st.latex(term_to_latex(term))
        free = term_free_indices(term)
        if free:
            st.warning(f"Índices libres: {', '.join(free)}")
        else:
            st.success("Sin índices libres ✓")

    return term


# ─────────────────────────────────────────────
# WIDGET REUTILIZABLE: PARÁMETROS ALEATORIOS
# ─────────────────────────────────────────────

def random_controls(key_prefix: str) -> Tuple[float, float, bool, int]:
    c1, c2, c3, c4 = st.columns(4)
    with c1: low  = st.number_input("Mínimo",  value=0.0,  key=f"{key_prefix}_low")
    with c2: high = st.number_input("Máximo",  value=10.0, key=f"{key_prefix}_high")
    with c3: intm = st.checkbox("Entero",      value=False, key=f"{key_prefix}_int")
    with c4: seed = st.number_input("Semilla", value=123,  step=1, key=f"{key_prefix}_seed")
    return low, high, intm, int(seed)


# ─────────────────────────────────────────────
# BARRA LATERAL
# ─────────────────────────────────────────────

st.sidebar.title("Navegación")
section = st.sidebar.radio("Ir a:", [
    "📥 Ingreso de información",
    "📐 Definición del modelo",
    "📊 Salidas del modelo",
], index=0)

spec = st.session_state["model_spec"]
st.sidebar.markdown("---")
st.sidebar.markdown("**Estado actual**")
cols = st.sidebar.columns(3)
cols[0].metric("Índices",     len(spec["indices"]))
cols[1].metric("Parámetros",  len(spec["parameters"]))
cols[2].metric("Variables",   len(spec["variables"]))
if spec["objective"]:
    st.sidebar.markdown(f"✅ Función objetivo definida")
if spec["constraints"]:
    st.sidebar.markdown(f"✅ {len(spec['constraints'])} familia(s) de restricciones")


# ═══════════════════════════════════════════════════════
# SECCIÓN 1: INGRESO DE INFORMACIÓN
# ═══════════════════════════════════════════════════════

if section == "📥 Ingreso de información":
    st.header("Ingreso de información")

    # ── ÍNDICES ──────────────────────────────────────────
    st.subheader("1. Índices")
    st.caption("Define los conjuntos de índices del modelo, p. ej. `i`, `j`, `t`.")

    num_idx = st.number_input("Número de índices", min_value=1, max_value=10,
                               value=max(1, len(spec["indices"]) or 3), step=1, key="num_indices")

    idx_rows, used_names, idx_errors = [], set(), []
    existing_idx_names = list(spec["indices"].keys())

    for r in range(int(num_idx)):
        default_name = existing_idx_names[r] if r < len(existing_idx_names) else f"idx_{r+1}"
        default_size = spec["indices"].get(default_name, {}).get("size", 3)
        c1, c2 = st.columns([2, 2])
        with c1:
            name = st.text_input(f"Nombre índice {r+1}", value=default_name, key=f"idx_name_{r}").strip()
        with c2:
            size = st.number_input(f"Tamaño de {name or f'índice {r+1}'}", min_value=1, max_value=1000,
                                    value=int(default_size), step=1, key=f"idx_size_{r}")
        if not is_valid_symbol(name):
            idx_errors.append(f"`{name}` no es un nombre válido.")
        elif name in used_names:
            idx_errors.append(f"Índice `{name}` duplicado.")
        else:
            used_names.add(name)
            idx_rows.append({"name": name, "size": int(size), "elements": build_elements(int(size), name)})

    for e in idx_errors:
        st.error(e)
    valid_idx = not idx_errors

    if valid_idx:
        index_specs = {r["name"]: {"size": r["size"], "elements": r["elements"]} for r in idx_rows}
        spec["indices"] = index_specs
        # Limpiar parámetros/variables que referencien índices eliminados
        valid_set = set(index_specs)
        spec["parameters"] = {k: v for k, v in spec["parameters"].items()
                               if all(i in valid_set for i in v.get("indices", []))}
        spec["variables"]  = {k: v for k, v in spec["variables"].items()
                               if all(i in valid_set for i in v.get("indices", []))}
        # Preview
        preview = [{"Índice": n, "Tamaño": s["size"], "Elementos": ", ".join(s["elements"])}
                   for n, s in index_specs.items()]
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
    else:
        index_specs = spec.get("indices", {})

    # ── PARÁMETROS ───────────────────────────────────────
    st.markdown("---")
    st.subheader("2. Parámetros")

    if not valid_idx or not index_specs:
        st.info("Primero define índices válidos.")
    else:
        idx_options = list(index_specs.keys())
        cur_params  = spec["parameters"]
        num_params  = st.number_input("Número de parámetros", min_value=0, max_value=30,
                                       value=max(1, len(cur_params)), step=1, key="num_params")
        new_params  = {}
        cur_names   = list(cur_params.keys())

        for p in range(int(num_params)):
            with st.expander(f"Parámetro {p+1}", expanded=p == 0):
                old_name = cur_names[p] if p < len(cur_names) else f"param_{p+1}"
                old_spec = cur_params.get(old_name, {})
                c1, c2   = st.columns([2, 3])
                with c1:
                    pname = st.text_input("Nombre", value=old_name, key=f"p_name_{p}").strip()
                with c2:
                    p_idx = st.multiselect("Índices", idx_options,
                                           default=[i for i in old_spec.get("indices", []) if i in idx_options],
                                           key=f"p_idx_{p}")

                if not is_valid_symbol(pname):
                    st.error(f"`{pname}` no es válido.")
                    continue
                if pname in new_params:
                    st.error(f"Parámetro `{pname}` duplicado.")
                    continue

                n_elems = total_elements(p_idx, index_specs)
                st.caption(f"Firma: `{pretty_sig(pname, p_idx)}` — {n_elems} elemento(s)")

                mode_opts = ["Manual", "Aleatorio"] if n_elems <= 12 else ["Aleatorio"]
                old_mode  = old_spec.get("mode", mode_opts[0])
                mode      = st.radio("Modo", mode_opts,
                                     index=mode_opts.index(old_mode) if old_mode in mode_opts else 0,
                                     horizontal=True, key=f"p_mode_{p}")
                existing  = old_spec.get("values", {})
                record    = {"indices": p_idx, "mode": mode, "values": {}}

                if not p_idx:
                    # Escalar
                    if mode == "Manual":
                        v = st.number_input("Valor", value=scalar_get(existing), key=f"p_scalar_{p}")
                        record["values"] = scalar_set(v)
                    else:
                        low, high, intm, seed = random_controls(f"p_{p}")
                        if low > high: st.error("Mínimo > máximo."); continue
                        if st.button(f"Generar {pname}", key=f"p_gen_{p}"):
                            st.session_state[f"pv_{pname}"] = random_scalar(low, high, intm, seed)
                        gen = st.session_state.get(f"pv_{pname}", existing or random_scalar(low, high, intm, seed))
                        record["values"] = gen
                        st.info(f"Valor: **{scalar_get(gen):.4f}**")

                elif len(p_idx) == 1:
                    labels = index_specs[p_idx[0]]["elements"]
                    combos = cartesian(p_idx, index_specs)
                    if mode == "Manual":
                        df0    = values_1d_to_df(labels, existing)
                        edited = st.data_editor(df0, use_container_width=True, num_rows="fixed",
                                                hide_index=True, disabled=["label"], key=f"p_ed1d_{p}")
                        record["values"] = df_1d_to_values(edited, labels)
                    else:
                        low, high, intm, seed = random_controls(f"p_{p}")
                        if low > high: st.error("Mínimo > máximo."); continue
                        if st.button(f"Generar {pname}", key=f"p_gen_{p}"):
                            st.session_state[f"pv_{pname}"] = random_values(combos, low, high, intm, seed)
                        gen = st.session_state.get(f"pv_{pname}", existing or random_values(combos, low, high, intm, seed))
                        record["values"] = gen
                        st.dataframe(values_1d_to_df(labels, gen), use_container_width=True, hide_index=True)

                else:
                    combos = cartesian(p_idx, index_specs)
                    if mode == "Manual":
                        df0    = values_to_df(p_idx, combos, existing)
                        edited = st.data_editor(df0, use_container_width=True, num_rows="fixed",
                                                hide_index=True, disabled=list(p_idx), key=f"p_ednd_{p}")
                        record["values"] = df_to_values(edited, p_idx)
                    else:
                        low, high, intm, seed = random_controls(f"p_{p}")
                        if low > high: st.error("Mínimo > máximo."); continue
                        if st.button(f"Generar {pname}", key=f"p_gen_{p}"):
                            st.session_state[f"pv_{pname}"] = random_values(combos, low, high, intm, seed)
                        gen = st.session_state.get(f"pv_{pname}", existing or random_values(combos, low, high, intm, seed))
                        record["values"] = gen
                        st.dataframe(values_to_df(p_idx, combos, gen), use_container_width=True, hide_index=True)

                new_params[pname] = record

        spec["parameters"] = new_params

        if new_params:
            summary = [{"Parámetro": pretty_sig(n, s["indices"]),
                        "Modo": s["mode"],
                        "Elementos": total_elements(s["indices"], index_specs)}
                       for n, s in new_params.items()]
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    # ── VARIABLES ────────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Variables")

    if not valid_idx or not index_specs:
        st.info("Primero define índices válidos.")
    else:
        idx_options = list(index_specs.keys())
        cur_vars    = spec["variables"]
        num_vars    = st.number_input("Número de variables", min_value=0, max_value=30,
                                       value=max(1, len(cur_vars)), step=1, key="num_vars")
        new_vars    = {}
        cur_vnames  = list(cur_vars.keys())

        for v in range(int(num_vars)):
            with st.expander(f"Variable {v+1}", expanded=v == 0):
                old_name = cur_vnames[v] if v < len(cur_vnames) else f"x_{v+1}"
                old_vs   = cur_vars.get(old_name, {})
                c1, c2, c3 = st.columns([2, 3, 2])
                with c1:
                    vname = st.text_input("Nombre", value=old_name, key=f"v_name_{v}").strip()
                with c2:
                    v_idx = st.multiselect("Índices", idx_options,
                                           default=[i for i in old_vs.get("indices", []) if i in idx_options],
                                           key=f"v_idx_{v}")
                domain_opts = list(_DOMAIN_LABELS.keys())
                old_domain  = old_vs.get("domain", "NonNegativeReals")
                with c3:
                    domain = st.selectbox("Dominio", domain_opts,
                                          index=domain_opts.index(old_domain) if old_domain in domain_opts else 1,
                                          format_func=lambda x: _DOMAIN_LABELS[x],
                                          key=f"v_domain_{v}")

                if not is_valid_symbol(vname):
                    st.error(f"`{vname}` no es válido.")
                    continue
                if vname in new_vars:
                    st.error(f"Variable `{vname}` duplicada.")
                    continue

                st.caption(f"Firma: `{pretty_sig(vname, v_idx)}` — "
                           f"{total_elements(v_idx, index_specs)} componente(s) — {_DOMAIN_LABELS[domain]}")
                new_vars[vname] = {"indices": v_idx, "domain": domain}

        spec["variables"] = new_vars


# ═══════════════════════════════════════════════════════
# SECCIÓN 2: DEFINICIÓN DEL MODELO
# ═══════════════════════════════════════════════════════

elif section == "📐 Definición del modelo":
    st.header("Definición del modelo")

    if not spec["indices"]:
        st.warning("Primero define al menos un índice."); st.stop()
    if not spec["variables"]:
        st.warning("Primero define al menos una variable."); st.stop()

    catalog     = object_catalog(spec)
    index_names = list(spec["indices"].keys())

    # ── TIPO DE PROBLEMA ────────────────────────────────
    st.subheader("1. Sentido de la optimización")
    cur_obj   = spec.get("objective") or {}
    sense_opt = ["minimize", "maximize"]
    sense     = st.radio("Sentido", sense_opt,
                         index=sense_opt.index(cur_obj.get("sense", "minimize")),
                         format_func=lambda x: "Minimizar" if x == "minimize" else "Maximizar",
                         horizontal=True, key="obj_sense")

    # ── FUNCIÓN OBJETIVO ────────────────────────────────
    st.markdown("---")
    st.subheader("2. Función objetivo")
    old_obj_terms = cur_obj.get("terms", [])
    n_obj_terms   = st.number_input("Número de términos", min_value=1, max_value=20,
                                     value=max(1, len(old_obj_terms)), step=1, key="n_obj_terms")
    obj_terms = []
    for t in range(int(n_obj_terms)):
        with st.expander(f"Término FO {t+1}", expanded=True):
            old_t = old_obj_terms[t] if t < len(old_obj_terms) else None
            obj_terms.append(term_widget(f"obj_{t}", catalog, index_names, old_t))

    spec["objective"] = {"sense": sense, "terms": obj_terms}

    sense_sym = r"\min" if sense == "minimize" else r"\max"
    st.markdown("**Vista previa de la función objetivo:**")
    st.latex(rf"{sense_sym}\ Z = {expr_to_latex(obj_terms)}")

    obj_errs = validate_objective(obj_terms)
    for e in obj_errs: st.error(e)
    if not obj_errs: st.success("Función objetivo estructuralmente consistente ✓")

    # ── RESTRICCIONES ───────────────────────────────────
    st.markdown("---")
    st.subheader("3. Familias de restricciones")
    old_cons  = spec.get("constraints", [])
    n_families = st.number_input("Número de familias", min_value=0, max_value=30,
                                  value=max(0, len(old_cons)), step=1, key="n_families")
    new_cons = []

    for r in range(int(n_families)):
        old_fam = old_cons[r] if r < len(old_cons) else None
        old_f   = old_fam or {}
        with st.expander(f"Familia {r+1}: {old_f.get('name', f'R{r+1}')}", expanded=False):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                fname = st.text_input("Nombre", value=old_f.get("name", f"R{r+1}"), key=f"fname_{r}")
            with c2:
                forall = st.multiselect("Para todo", index_names,
                                        default=[i for i in old_f.get("forall", []) if i in index_names],
                                        key=f"forall_{r}")
            with c3:
                sense_f = st.selectbox("Sentido", ["<=", ">=", "="],
                                        index=["<=", ">=", "="].index(old_f.get("sense", "<=")),
                                        key=f"sense_{r}")

            # LHS
            st.markdown("**Lado izquierdo (LHS)**")
            old_lhs   = old_f.get("lhs_terms", [])
            n_lhs     = st.number_input("Términos LHS", min_value=1, max_value=20,
                                         value=max(1, len(old_lhs)), step=1, key=f"nlhs_{r}")
            lhs_terms = []
            for t in range(int(n_lhs)):
                with st.container():
                    st.markdown(f"*Término LHS {t+1}*")
                    old_t = old_lhs[t] if t < len(old_lhs) else None
                    lhs_terms.append(term_widget(f"lhs_{r}_{t}", catalog, index_names, old_t))

            # RHS
            st.markdown("**Lado derecho (RHS)**")
            old_rhs   = old_f.get("rhs_terms", [])
            n_rhs     = st.number_input("Términos RHS", min_value=1, max_value=20,
                                         value=max(1, len(old_rhs)), step=1, key=f"nrhs_{r}")
            rhs_terms = []
            for t in range(int(n_rhs)):
                with st.container():
                    st.markdown(f"*Término RHS {t+1}*")
                    old_t = old_rhs[t] if t < len(old_rhs) else None
                    rhs_terms.append(term_widget(f"rhs_{r}_{t}", catalog, index_names, old_t))

            family = {"name": fname, "forall": forall, "sense": sense_f,
                      "lhs_terms": lhs_terms, "rhs_terms": rhs_terms}
            fam_errs = validate_constraint(family)
            st.markdown("**Vista previa:**")
            st.latex(constraint_to_latex(family))
            for e in fam_errs: st.error(e)
            if not fam_errs: st.success(f"Familia `{fname}` consistente ✓")
            new_cons.append(family)

    spec["constraints"] = new_cons

    # ── RESUMEN ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("4. Resumen del modelo")
    st.latex(rf"{sense_sym}\ Z = {expr_to_latex(obj_terms)}")
    if new_cons:
        st.write("**Sujeto a:**")
        for fam in new_cons:
            st.latex(constraint_to_latex(fam))
    else:
        st.info("Sin restricciones definidas.")


# ═══════════════════════════════════════════════════════
# SECCIÓN 3: SALIDAS DEL MODELO
# ═══════════════════════════════════════════════════════

elif section == "📊 Salidas del modelo":
    st.header("Salidas del modelo")

    # Validación
    st.subheader("1. Validación")
    errors = []
    if not spec["indices"]:   errors.append("Sin índices definidos.")
    if not spec["variables"]: errors.append("Sin variables definidas.")
    if not spec["objective"]: errors.append("Sin función objetivo.")
    else:
        errors.extend(validate_objective(spec["objective"].get("terms", [])))
    for fam in spec.get("constraints", []):
        errors.extend(validate_constraint(fam))
    errors.extend(validate_linearity(spec))

    if errors:
        for e in errors: st.error(e)
        st.stop()
    st.success("Especificación válida ✓")

    # Resumen
    st.markdown("---")
    st.subheader("2. Modelo")
    obj  = spec["objective"]
    sym  = r"\min" if obj["sense"] == "minimize" else r"\max"
    st.latex(rf"{sym}\ Z = {expr_to_latex(obj['terms'])}")
    if spec["constraints"]:
        st.write("Sujeto a:")
        for fam in spec["constraints"]:
            st.latex(constraint_to_latex(fam))

    # Resolver
    st.markdown("---")
    st.subheader("3. Resolver")
    solver_name = st.selectbox("Solver", ["appsi_highs", "glpk", "cbc"], index=0)

    if st.button("▶ Resolver modelo", type="primary"):
        with st.spinner("Construyendo y resolviendo…"):
            try:
                model  = build_pyomo_model(spec)
                solver = pyo.SolverFactory(solver_name)
                result = solver.solve(model)

                spec["results"] = {
                    "solver_name":          solver_name,
                    "termination_condition": str(result.solver.termination_condition),
                    "status":               str(result.solver.status),
                    "objective_value":      pyo.value(model.OBJ),
                }
                st.session_state["solved_model"] = model
                st.success("¡Modelo resuelto!")
            except Exception as ex:
                st.error(f"Error: {ex}"); st.stop()

    # Resultados
    results = spec.get("results")
    model   = st.session_state.get("solved_model")

    if not results or not model:
        st.info("Aún no has resuelto el modelo.")
    else:
        st.markdown("---")
        st.subheader("4. Resultados")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Solver",       results["solver_name"])
        c2.metric("Status",       results["status"])
        c3.metric("Terminación",  results["termination_condition"])
        c4.metric("Valor óptimo", f"{results['objective_value']:.6f}")

        st.markdown("---")
        st.subheader("5. Solución por variable")
        vnames = list(spec["variables"].keys())
        sel    = st.selectbox("Variable", vnames)
        df_v   = var_to_df(model, sel, spec["variables"][sel], spec["indices"])
        st.dataframe(df_v, use_container_width=True, hide_index=True)
        st.download_button(f"⬇ Descargar {sel} (CSV)",
                           df_v.to_csv(index=False).encode(),
                           f"sol_{sel}.csv", "text/csv")

        st.markdown("---")
        st.subheader("6. Variables no nulas")
        df_nz = nonzero_df(model, spec)
        st.dataframe(df_nz, use_container_width=True, hide_index=True)
        st.download_button("⬇ Descargar variables no nulas (CSV)",
                           df_nz.to_csv(index=False).encode(),
                           "sol_no_nulas.csv", "text/csv")
