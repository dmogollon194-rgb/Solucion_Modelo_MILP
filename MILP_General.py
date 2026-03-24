import streamlit as st
import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Tuple, Any
import pyomo.environ as pyo

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Constructor de Modelos Algebraicos",
    page_icon="📐",
    layout="wide",
)

# ============================================================
# ESTILOS VISUALES
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #07101f 0%, #050b16 100%);
        color: #f3f7ff;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #f3f7ff;
    }

    .top-hero {
        background: linear-gradient(135deg, rgba(8,22,55,0.95), rgba(3,10,28,0.98));
        border: 1px solid rgba(61, 132, 255, 0.22);
        border-radius: 22px;
        padding: 22px 26px;
        margin-bottom: 16px;
        box-shadow: 0 0 0 1px rgba(61, 132, 255, 0.06), 0 10px 35px rgba(0, 0, 0, 0.35);
    }

    .top-hero h2 {
        margin: 0 0 8px 0;
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
    }

    .top-hero p {
        margin: 0;
        font-size: 1.0rem;
        color: #d7e6ff;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(8,22,55,0.95), rgba(3,10,28,0.98));
        border: 1px solid rgba(61, 132, 255, 0.22);
        border-radius: 18px;
        padding: 16px 18px;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 0 0 1px rgba(61, 132, 255, 0.05), 0 10px 28px rgba(0, 0, 0, 0.28);
        margin-bottom: 10px;
    }

    .metric-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.05;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(8,22,55,0.96), rgba(3,10,28,0.99));
        border: 1px solid rgba(61, 132, 255, 0.22);
        border-radius: 18px;
        padding: 16px 18px;
        min-height: 122px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 0 0 1px rgba(61, 132, 255, 0.05), 0 10px 28px rgba(0, 0, 0, 0.28);
        margin-bottom: 12px;
    }

    .result-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .result-value {
        font-size: 2.35rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.05;
        word-break: break-word;
    }

    .result-wide {
        background: linear-gradient(135deg, rgba(8,22,55,0.96), rgba(3,10,28,0.99));
        border: 1px solid rgba(61, 132, 255, 0.22);
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        min-height: 120px;
        box-shadow: 0 0 0 1px rgba(61, 132, 255, 0.05), 0 10px 28px rgba(0, 0, 0, 0.28);
        margin-top: 4px;
        margin-bottom: 12px;
    }

    .result-wide .result-title {
        margin-bottom: 12px;
    }

    .result-wide .result-value {
        font-size: 2.05rem;
    }

    .section-box {
        background: rgba(5, 12, 28, 0.78);
        border: 1px solid rgba(61, 132, 255, 0.16);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        margin-bottom: 14px;
    }

    .section-subtitle {
        font-size: 1.18rem;
        font-weight: 800;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .section-text {
        color: #d7e6ff;
        font-size: 0.98rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08101f 0%, #050b16 100%);
        border-right: 1px solid rgba(61, 132, 255, 0.18);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(61, 132, 255, 0.14);
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0c2b69, #0a1f49);
        color: white;
        border: 1px solid rgba(100, 162, 255, 0.45);
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1rem;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #0c2b69, #0a1f49);
        color: white;
        border: 1px solid rgba(100, 162, 255, 0.45);
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1rem;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div, .stTextArea textarea {
        background-color: rgba(7, 16, 35, 0.92) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(8, 22, 55, 0.72);
        border-radius: 12px 12px 0 0;
        padding: 10px 18px;
        color: #dfeaff;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #12306f, #0b1f4a);
        color: white !important;
    }

    hr {
        border-color: rgba(61, 132, 255, 0.18);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ESTADO INICIAL
# ============================================================

def init_session_state() -> None:
    defaults = {
        "model_spec": {
            "indices": {},
            "parameters": {},
            "variables": {},
            "objective": None,
            "constraints": [],
            "results": None,
        }
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# ============================================================
# UTILIDADES GENERALES
# ============================================================

def is_valid_symbol(name: str) -> bool:
    if not isinstance(name, str):
        return False
    name = name.strip()
    if len(name) == 0:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    for ch in name[1:]:
        if not (ch.isalnum() or ch == "_"):
            return False
    return True


def sanitize_symbol(name: str) -> str:
    return name.strip()


def build_index_elements(size: int, label_prefix: str) -> List[str]:
    return [f"{label_prefix}{t}" for t in range(1, size + 1)]


def total_elements_for_dims(index_names: List[str], index_specs: Dict[str, Dict[str, Any]]) -> int:
    if len(index_names) == 0:
        return 1
    total = 1
    for idx_name in index_names:
        total *= int(index_specs[idx_name]["size"])
    return total


def cartesian_labels(index_names: List[str], index_specs: Dict[str, Dict[str, Any]]) -> List[Tuple[str, ...]]:
    if len(index_names) == 0:
        return [tuple()]
    arrays = [index_specs[idx]["elements"] for idx in index_names]
    return list(itertools.product(*arrays))


def pretty_param_signature(name: str, index_names: List[str]) -> str:
    if len(index_names) == 0:
        return name
    return f"{name}[{', '.join(index_names)}]"


def pretty_var_signature(name: str, index_names: List[str]) -> str:
    if len(index_names) == 0:
        return name
    return f"{name}[{', '.join(index_names)}]"


def values_dict_from_scalar(value: float) -> Dict[str, float]:
    return {"__scalar__": float(value)}


def scalar_from_values_dict(values: Dict[str, float], default: float = 0.0) -> float:
    return float(values.get("__scalar__", default))


def values_dict_from_dataframe_1d(df: pd.DataFrame, labels: List[str], value_col: str = "value") -> Dict[str, float]:
    out = {}
    for lbl in labels:
        val = df.loc[df["label"] == lbl, value_col].iloc[0]
        out[str((lbl,))] = float(val)
    return out


def dataframe_1d_from_values_dict(labels: List[str], values_dict: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for lbl in labels:
        key = str((lbl,))
        rows.append({"label": lbl, "value": float(values_dict.get(key, 0.0))})
    return pd.DataFrame(rows)


def values_dict_from_dataframe_nd(df: pd.DataFrame, index_names: List[str], value_col: str = "value") -> Dict[str, float]:
    out = {}
    for _, row in df.iterrows():
        key_tuple = tuple(str(row[idx]) for idx in index_names)
        out[str(key_tuple)] = float(row[value_col])
    return out


def dataframe_nd_from_values_dict(
    index_names: List[str],
    combos: List[Tuple[str, ...]],
    values_dict: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for combo in combos:
        row = {}
        for pos, idx_name in enumerate(index_names):
            row[idx_name] = combo[pos]
        row["value"] = float(values_dict.get(str(combo), 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def random_values_dict(
    combos: List[Tuple[str, ...]],
    low: float,
    high: float,
    integer_mode: bool,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for combo in combos:
        if integer_mode:
            val = int(rng.integers(low=int(low), high=int(high) + 1))
            out[str(combo)] = float(val)
        else:
            val = float(rng.uniform(low, high))
            out[str(combo)] = float(val)
    return out


def random_scalar(low: float, high: float, integer_mode: bool, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if integer_mode:
        val = int(rng.integers(low=int(low), high=int(high) + 1))
        return {"__scalar__": float(val)}
    return {"__scalar__": float(rng.uniform(low, high))}


def infer_domain_text(domain_code: str) -> str:
    mapper = {
        "Binary": "Binarias",
        "NonNegativeReals": "Reales ≥ 0",
        "NonNegativeIntegers": "Enteras ≥ 0",
    }
    return mapper.get(domain_code, domain_code)


def ordered_unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def reset_parameters_if_invalid(index_specs: Dict[str, Dict[str, Any]]) -> None:
    valid_index_names = set(index_specs.keys())
    old_params = st.session_state["model_spec"]["parameters"]
    cleaned = {}
    for pname, pspec in old_params.items():
        idxs = pspec.get("indices", [])
        if all(idx in valid_index_names for idx in idxs):
            cleaned[pname] = pspec
    st.session_state["model_spec"]["parameters"] = cleaned


def reset_variables_if_invalid(index_specs: Dict[str, Dict[str, Any]]) -> None:
    valid_index_names = set(index_specs.keys())
    old_vars = st.session_state["model_spec"]["variables"]
    cleaned = {}
    for vname, vspec in old_vars.items():
        idxs = vspec.get("indices", [])
        if all(idx in valid_index_names for idx in idxs):
            cleaned[vname] = vspec
    st.session_state["model_spec"]["variables"] = cleaned


# ============================================================
# UTILIDADES DE DEFINICIÓN DEL MODELO
# ============================================================

def object_catalog(model_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []

    for pname, pspec in model_spec["parameters"].items():
        items.append({
            "kind": "parameter",
            "name": pname,
            "indices": pspec["indices"],
            "label": pretty_param_signature(pname, pspec["indices"])
        })

    for vname, vspec in model_spec["variables"].items():
        items.append({
            "kind": "variable",
            "name": vname,
            "indices": vspec["indices"],
            "label": pretty_var_signature(vname, vspec["indices"])
        })

    return items


def factor_to_text(factor: Dict[str, Any]) -> str:
    if factor["type"] == "object":
        return factor["label"]
    elif factor["type"] == "constant":
        return str(factor["value"])
    return "<?>"


def term_used_indices(term: Dict[str, Any]) -> List[str]:
    idxs = []
    for fac in term["factors"]:
        if fac["type"] == "object":
            idxs.extend(fac["indices"])
    return ordered_unique(idxs)


def term_free_indices(term: Dict[str, Any]) -> List[str]:
    used = term_used_indices(term)
    summed = term.get("sum_over", [])
    return [idx for idx in used if idx not in summed]


def build_term_text(term: Dict[str, Any]) -> str:
    sign = term.get("sign", "+")
    factors_txt = " * ".join(factor_to_text(f) for f in term["factors"]) if term["factors"] else "0"
    txt = factors_txt

    sum_over = term.get("sum_over", [])
    if len(sum_over) > 0:
        sums = " ".join([f"Σ_{{{idx}}}" for idx in sum_over])
        txt = f"{sums} ({txt})"

    if sign == "-":
        txt = f"- {txt}"
    else:
        txt = f"+ {txt}"

    return txt


def build_expression_text(terms: List[Dict[str, Any]]) -> str:
    if len(terms) == 0:
        return "0"
    parts = [build_term_text(t) for t in terms]
    expr = " ".join(parts).strip()
    if expr.startswith("+ "):
        expr = expr[2:]
    return expr


def validate_objective_terms(terms: List[Dict[str, Any]]) -> List[str]:
    errors = []
    for pos, term in enumerate(terms, start=1):
        free_idxs = term_free_indices(term)
        if len(free_idxs) > 0:
            errors.append(
                f"En el término {pos} de la función objetivo quedaron índices libres: {', '.join(free_idxs)}. "
                f"Todos los índices usados deben estar cubiertos por sumatorias."
            )
    return errors


def validate_constraint_family(family: Dict[str, Any]) -> List[str]:
    errors = []

    lhs_terms = family.get("lhs_terms", [])
    rhs_terms = family.get("rhs_terms", [])
    forall = family.get("forall", [])

    lhs_free = ordered_unique([idx for t in lhs_terms for idx in term_free_indices(t)])
    rhs_free = ordered_unique([idx for t in rhs_terms for idx in term_free_indices(t)])

    lhs_is_constant = (len(lhs_free) == 0)
    rhs_is_constant = (len(rhs_free) == 0)

    if not lhs_is_constant and not rhs_is_constant:
        if lhs_free != rhs_free:
            errors.append(
                f"Los índices libres del lado izquierdo ({lhs_free}) no coinciden con los del lado derecho ({rhs_free})."
            )
        if lhs_free != forall:
            errors.append(
                f"Los índices libres de la restricción ({lhs_free}) no coinciden con el 'para todo' definido ({forall})."
            )

    elif lhs_is_constant and not rhs_is_constant:
        if rhs_free != forall:
            errors.append(
                f"El lado izquierdo es constante, por lo que los índices libres del lado derecho ({rhs_free}) "
                f"deben coincidir con el 'para todo' ({forall})."
            )

    elif not lhs_is_constant and rhs_is_constant:
        if lhs_free != forall:
            errors.append(
                f"El lado derecho es constante, por lo que los índices libres del lado izquierdo ({lhs_free}) "
                f"deben coincidir con el 'para todo' ({forall})."
            )

    else:
        if len(forall) > 0:
            errors.append(
                f"Ambos lados son constantes, así que no debería existir 'para todo' ({forall})."
            )

    return errors


def build_constraint_family_text(family: Dict[str, Any]) -> str:
    lhs = build_expression_text(family.get("lhs_terms", []))
    rhs = build_expression_text(family.get("rhs_terms", []))
    sense = family.get("sense", "<=")
    forall = family.get("forall", [])

    txt = f"{lhs} {sense} {rhs}"
    if len(forall) > 0:
        txt += "   ∀ " + ", ".join(forall)
    return txt


# ============================================================
# UTILIDADES LATEX
# ============================================================

def format_number_latex(val: float) -> str:
    val = float(val)
    if val.is_integer():
        return str(int(val))
    return f"{val:.2f}"


def factor_to_latex(factor: Dict[str, Any]) -> str:
    if factor["type"] == "object":
        name = factor["name"]
        indices = factor["indices"]

        if len(indices) == 0:
            return name

        idx_str = ",".join(indices)
        return rf"{name}_{{{idx_str}}}"

    elif factor["type"] == "constant":
        return format_number_latex(factor["value"])

    return "?"


def build_term_latex(term: Dict[str, Any]) -> str:
    sign = term.get("sign", "+")
    factors = term.get("factors", [])

    if len(factors) == 0:
        body = "0"
    else:
        body = r" \cdot ".join(factor_to_latex(f) for f in factors)

    sum_over = term.get("sum_over", [])
    if len(sum_over) > 0:
        sums = " ".join([rf"\sum_{{{idx}}}" for idx in sum_over])
        body = f"{sums}\\left({body}\\right)"

    if sign == "-":
        return f"- {body}"
    return f"+ {body}"


def build_expression_latex(terms: List[Dict[str, Any]]) -> str:
    if len(terms) == 0:
        return "0"

    parts = [build_term_latex(t) for t in terms]
    expr = " ".join(parts).strip()

    if expr.startswith("+ "):
        expr = expr[2:]

    return expr


def build_constraint_family_latex(family: Dict[str, Any]) -> str:
    lhs = build_expression_latex(family.get("lhs_terms", []))
    rhs = build_expression_latex(family.get("rhs_terms", []))
    sense = family.get("sense", "<=")
    forall = family.get("forall", [])

    sense_map = {
        "<=": r"\leq",
        ">=": r"\geq",
        "=": "="
    }

    txt = f"{lhs} {sense_map[sense]} {rhs}"

    if len(forall) > 0:
        txt += r"\quad \forall " + ", ".join(forall)

    return txt


# ============================================================
# UTILIDADES PARA CONSTRUIR Y RESOLVER EN PYOMO
# ============================================================

def pyomo_domain_from_code(domain_code: str):
    domain_map = {
        "Binary": pyo.Binary,
        "NonNegativeReals": pyo.NonNegativeReals,
        "NonNegativeIntegers": pyo.NonNegativeIntegers,
    }
    if domain_code not in domain_map:
        raise ValueError(f"Dominio no soportado: {domain_code}")
    return domain_map[domain_code]


def get_component_value(model, factor: Dict[str, Any], env: Dict[str, Any]):
    if factor["type"] == "constant":
        return float(factor["value"])

    if factor["type"] != "object":
        raise ValueError("Tipo de factor no soportado.")

    name = factor["name"]
    idxs = factor["indices"]
    kind = factor["kind"]

    if kind == "parameter":
        comp = getattr(model, f"par_{name}")
    elif kind == "variable":
        comp = getattr(model, f"var_{name}")
    else:
        raise ValueError(f"Kind no soportado: {kind}")

    if len(idxs) == 0:
        return comp

    try:
        key = tuple(env[idx] for idx in idxs)
    except KeyError as e:
        raise ValueError(
            f"Falta el índice `{e.args[0]}` en el entorno al evaluar `{name}[{', '.join(idxs)}]`."
        )

    if len(key) == 1:
        return comp[key[0]]
    return comp[key]


def count_variable_factors(term: Dict[str, Any]) -> int:
    c = 0
    for fac in term.get("factors", []):
        if fac["type"] == "object" and fac.get("kind") == "variable":
            c += 1
    return c


def validate_linearity_of_term(term: Dict[str, Any], context: str = "") -> List[str]:
    errors = []
    num_var_factors = count_variable_factors(term)
    if num_var_factors > 1:
        errors.append(
            f"{context} tiene {num_var_factors} factores variables en un mismo término. "
            "Eso genera no linealidad y esta versión solo soporta modelos lineales."
        )
    return errors


def validate_linearity_of_model(spec: Dict[str, Any]) -> List[str]:
    errors = []

    obj = spec.get("objective", None)
    if obj is not None:
        for i, term in enumerate(obj.get("terms", []), start=1):
            errors.extend(validate_linearity_of_term(term, context=f"FO término {i}"))

    for r, fam in enumerate(spec.get("constraints", []), start=1):
        for i, term in enumerate(fam.get("lhs_terms", []), start=1):
            errors.extend(validate_linearity_of_term(term, context=f"Restricción {fam.get('name', f'R{r}')}, LHS término {i}"))
        for i, term in enumerate(fam.get("rhs_terms", []), start=1):
            errors.extend(validate_linearity_of_term(term, context=f"Restricción {fam.get('name', f'R{r}')}, RHS término {i}"))

    return errors


def build_base_term_value(model, term: Dict[str, Any], env: Dict[str, Any]):
    factors = term.get("factors", [])
    if len(factors) == 0:
        val = 0.0
    else:
        val = 1
        for fac in factors:
            val = val * get_component_value(model, fac, env)

    sign = term.get("sign", "+")
    if sign == "-":
        val = -val

    return val


def evaluate_term_with_sums(model, term: Dict[str, Any], env: Dict[str, Any]):
    sum_over = term.get("sum_over", [])

    def recurse(pos: int, local_env: Dict[str, Any]):
        if pos == len(sum_over):
            return build_base_term_value(model, term, local_env)

        idx_name = sum_over[pos]
        pyomo_set = getattr(model, f"set_{idx_name}")

        return sum(
            recurse(pos + 1, {**local_env, idx_name: idx_val})
            for idx_val in pyomo_set
        )

    return recurse(0, dict(env))


def build_expression_pyomo(model, terms: List[Dict[str, Any]], env: Dict[str, Any]):
    if len(terms) == 0:
        return 0
    expr = 0
    for term in terms:
        expr += evaluate_term_with_sums(model, term, env)
    return expr


def build_pyomo_model_from_spec(spec: Dict[str, Any]):
    model = pyo.ConcreteModel()

    index_specs = spec["indices"]
    for idx_name, idx_spec in index_specs.items():
        setattr(
            model,
            f"set_{idx_name}",
            pyo.Set(initialize=idx_spec["elements"], ordered=True)
        )

    for pname, pspec in spec["parameters"].items():
        idxs = pspec["indices"]
        values = pspec["values"]

        if len(idxs) == 0:
            comp = pyo.Param(initialize=float(values["__scalar__"]), mutable=False)
        else:
            pyomo_sets = [getattr(model, f"set_{idx}") for idx in idxs]

            init_dict = {}
            combos = cartesian_labels(idxs, index_specs)
            for combo in combos:
                val = float(values.get(str(combo), 0.0))
                if len(combo) == 1:
                    init_dict[combo[0]] = val
                else:
                    init_dict[combo] = val

            comp = pyo.Param(*pyomo_sets, initialize=init_dict, mutable=False)

        setattr(model, f"par_{pname}", comp)

    for vname, vspec in spec["variables"].items():
        idxs = vspec["indices"]
        domain = pyomo_domain_from_code(vspec["domain"])

        if len(idxs) == 0:
            comp = pyo.Var(domain=domain)
        else:
            pyomo_sets = [getattr(model, f"set_{idx}") for idx in idxs]
            comp = pyo.Var(*pyomo_sets, domain=domain)

        setattr(model, f"var_{vname}", comp)

    obj = spec.get("objective", None)
    if obj is None:
        raise ValueError("No hay función objetivo definida.")

    objective_terms = obj.get("terms", [])
    sense = obj.get("sense", "minimize")

    obj_expr = build_expression_pyomo(model, objective_terms, env={})
    obj_sense = pyo.minimize if sense == "minimize" else pyo.maximize

    model.OBJ = pyo.Objective(expr=obj_expr, sense=obj_sense)

    for c_idx, fam in enumerate(spec.get("constraints", []), start=1):
        fname = fam.get("name", f"R{c_idx}")
        lhs_terms = fam.get("lhs_terms", [])
        rhs_terms = fam.get("rhs_terms", [])
        forall = fam.get("forall", [])
        sense_f = fam.get("sense", "<=")

        if len(forall) == 0:
            lhs_expr = build_expression_pyomo(model, lhs_terms, env={})
            rhs_expr = build_expression_pyomo(model, rhs_terms, env={})

            if sense_f == "<=":
                con = pyo.Constraint(expr=lhs_expr <= rhs_expr)
            elif sense_f == ">=":
                con = pyo.Constraint(expr=lhs_expr >= rhs_expr)
            else:
                con = pyo.Constraint(expr=lhs_expr == rhs_expr)

        else:
            pyomo_sets = [getattr(model, f"set_{idx}") for idx in forall]

            def make_rule(lhs_terms_local, rhs_terms_local, forall_local, sense_local):
                def _rule(m, *args):
                    env = dict(zip(forall_local, args))
                    lhs_expr = build_expression_pyomo(m, lhs_terms_local, env)
                    rhs_expr = build_expression_pyomo(m, rhs_terms_local, env)

                    if sense_local == "<=":
                        return lhs_expr <= rhs_expr
                    elif sense_local == ">=":
                        return lhs_expr >= rhs_expr
                    else:
                        return lhs_expr == rhs_expr
                return _rule

            con = pyo.Constraint(
                *pyomo_sets,
                rule=make_rule(lhs_terms, rhs_terms, forall, sense_f)
            )

        setattr(model, f"con_{fname}", con)

    return model


def solver_factory_from_name(solver_name: str):
    if solver_name == "appsi_highs":
        return pyo.SolverFactory("appsi_highs")
    elif solver_name == "glpk":
        return pyo.SolverFactory("glpk")
    elif solver_name == "cbc":
        return pyo.SolverFactory("cbc")
    else:
        raise ValueError(f"Solver no soportado: {solver_name}")


def variable_solution_to_dataframe(model, vname: str, vspec: Dict[str, Any], index_specs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    comp = getattr(model, f"var_{vname}")
    idxs = vspec["indices"]

    if len(idxs) == 0:
        return pd.DataFrame({
            "variable": [vname],
            "value": [pyo.value(comp)]
        })

    combos = cartesian_labels(idxs, index_specs)
    rows = []

    for combo in combos:
        if len(combo) == 1:
            val = pyo.value(comp[combo[0]])
        else:
            val = pyo.value(comp[combo])

        row = {}
        for pos, idx in enumerate(idxs):
            row[idx] = combo[pos]
        row["value"] = val
        rows.append(row)

    return pd.DataFrame(rows)


def all_variables_solution_flat(model, spec: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for vname, vspec in spec["variables"].items():
        df = variable_solution_to_dataframe(model, vname, vspec, spec["indices"])
        df.insert(0, "variable_name", vname)
        rows.append(df)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def nonzero_variables_solution_flat(model, spec: Dict[str, Any], tol: float = 1e-9) -> pd.DataFrame:
    df = all_variables_solution_flat(model, spec)
    if df.empty:
        return df
    return df[df["value"].abs() > tol].reset_index(drop=True)


# ============================================================
# COMPONENTES VISUALES
# ============================================================

def info_hero(title: str, text: str):
    st.markdown(
        f"""
        <div class="top-hero">
            <h2>{title}</h2>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(title: str, value: Any):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def result_card(title: str, value: Any):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">{title}</div>
            <div class="result-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def result_wide_card(title: str, value: Any):
    st.markdown(
        f"""
        <div class="result-wide">
            <div class="result-title">{title}</div>
            <div class="result-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
def count_expanded_constraints(spec: Dict[str, Any]) -> int:
    """
    Cuenta el número total de restricciones instanciadas a partir de las familias,
    expandiendo los índices del 'forall'.
    """
    total = 0
    index_specs = spec.get("indices", {})
    constraint_families = spec.get("constraints", [])

    for fam in constraint_families:
        forall = fam.get("forall", [])

        if len(forall) == 0:
            total += 1
        else:
            num = 1
            valid_family = True

            for idx in forall:
                if idx not in index_specs:
                    valid_family = False
                    break
                num *= int(index_specs[idx]["size"])

            if valid_family:
                total += num

    return total

def count_expanded_variables(spec: Dict[str, Any]) -> int:
    """
    Cuenta el número total de variables instanciadas,
    expandiendo sus índices.
    """
    total = 0
    index_specs = spec.get("indices", {})
    variables = spec.get("variables", {})

    for _, vspec in variables.items():
        idxs = vspec.get("indices", [])

        if len(idxs) == 0:
            total += 1
        else:
            num = 1
            valid_var = True

            for idx in idxs:
                if idx not in index_specs:
                    valid_var = False
                    break
                num *= int(index_specs[idx]["size"])

            if valid_var:
                total += num

    return total
# ============================================================
# BARRA LATERAL / NAVEGACIÓN
# ============================================================

spec = st.session_state["model_spec"]

num_indices = len(spec["indices"])
num_variables = count_expanded_variables(spec)
num_restricciones = count_expanded_constraints(spec)

st.sidebar.markdown("""
    <div style="
        padding: 0.4rem 0 1rem 0;
        border-bottom: 1px solid rgba(61, 132, 255, 0.18);
        margin-bottom: 1rem;
    ">
        <div style="
            font-size: 1.45rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0.35rem;
        ">
            Navegación
        </div>
        <div style="
            color: #b9c9e8;
            font-size: 0.92rem;
            line-height: 1.4;
        ">
            Explora cada etapa de construcción y solución del modelo.
        </div>
    </div>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Ir a:",
    [
        "Ingreso de información",
        "Definición del modelo",
        "Salidas del modelo",
    ],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
    <div style="
        font-size: 1.05rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.8rem;
    ">
        Estado actual
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(8,22,55,0.95), rgba(3,10,28,0.98));
            border: 1px solid rgba(61, 132, 255, 0.22);
            border-radius: 14px;
            padding: 12px 14px;
            margin-bottom: 10px;
            box-shadow: 0 0 0 1px rgba(61, 132, 255, 0.05), 0 8px 20px rgba(0,0,0,0.25);
        ">
            <div style="font-size: 0.88rem; color: #cfe0ff; font-weight: 700;">Índices</div>
            <div style="font-size: 1.8rem; color: white; font-weight: 800; line-height: 1.1;">{num_indices}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(8,22,55,0.95), rgba(3,10,28,0.98));
            border: 1px solid rgba(61, 132, 255, 0.22);
            border-radius: 14px;
            padding: 12px 14px;
            margin-bottom: 10px;
            box-shadow: 0 0 0 1px rgba(61, 132, 255, 0.05), 0 8px 20px rgba(0,0,0,0.25);
        ">
            <div style="font-size: 0.88rem; color: #cfe0ff; font-weight: 700;">Variables</div>
            <div style="font-size: 1.8rem; color: white; font-weight: 800; line-height: 1.1;">{num_variables}</div>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(8,22,55,0.95), rgba(3,10,28,0.98));
        border: 1px solid rgba(61, 132, 255, 0.22);
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 2px;
        margin-bottom: 12px;
        box-shadow: 0 0 0 1px rgba(61, 132, 255, 0.05), 0 8px 20px rgba(0,0,0,0.25);
    ">
        <div style="font-size: 0.9rem; color: #cfe0ff; font-weight: 700;">Restricciones definidas</div>
        <div style="font-size: 1.95rem; color: white; font-weight: 800; line-height: 1.1;">{num_restricciones}</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style="
        margin-top: 0.6rem;
        padding: 0.75rem 0.85rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(61, 132, 255, 0.12);
        border-radius: 12px;
        color: #b9c9e8;
        font-size: 0.87rem;
        line-height: 1.45;
    ">
        El número de restricciones corresponde al total expandido según los índices definidos en cada familia.
    </div>
""", unsafe_allow_html=True)

st.title("Solucionador de Modelos Lineales")
st.caption("Esta aplicación está diseñada para solucionar modelos lineales con un solo objetivo.")

if section == "Ingreso de información":

    info_hero(
        "1. Ingreso de información",
        "Define los índices, parámetros y variables que conformarán la estructura base del modelo."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Índices", len(st.session_state["model_spec"]["indices"]))
    with c2:
        metric_card("Parámetros", len(st.session_state["model_spec"]["parameters"]))
    with c3:
        metric_card("Variables", len(st.session_state["model_spec"]["variables"]))

    st.markdown("<br>", unsafe_allow_html=True)

    tab_ind, tab_par, tab_var = st.tabs(["Índices", "Parámetros", "Variables"])
    # ========================================================
    # TAB 1: ÍNDICES
    # ========================================================
    with tab_ind:
        st.markdown(
            "<div class='section-box'><div class='section-subtitle'>Configuración de índices</div>"
            "<div class='section-text'>Define los conjuntos base del modelo.</div></div>",
            unsafe_allow_html=True
        )

        num_indices = st.number_input(
            "Número de índices",
            min_value=1,
            max_value=10,
            value=max(1, len(st.session_state["model_spec"]["indices"]) or 3),
            step=1,
            key="num_indices"
        )

        index_rows = []
        used_names = set()
        index_errors = []

        for r in range(int(num_indices)):
            col1, col2 = st.columns([2, 2])

            default_name = f"idx_{r+1}"
            if r < len(st.session_state["model_spec"]["indices"]):
                existing_names = list(st.session_state["model_spec"]["indices"].keys())
                default_name = existing_names[r]

            with col1:
                idx_name = st.text_input(
                    f"Nombre del índice {r+1}",
                    value=default_name,
                    key=f"index_name_{r}"
                ).strip()

            with col2:
                existing_size = 3
                if r < len(st.session_state["model_spec"]["indices"]):
                    existing_names = list(st.session_state["model_spec"]["indices"].keys())
                    if r < len(existing_names):
                        existing_size = st.session_state["model_spec"]["indices"][existing_names[r]]["size"]

                idx_size = st.number_input(
                    f"Tamaño de {idx_name or f'índice {r+1}'}",
                    min_value=1,
                    max_value=1000,
                    value=int(existing_size),
                    step=1,
                    key=f"index_size_{r}"
                )

            if not is_valid_symbol(idx_name):
                index_errors.append(f"El nombre `{idx_name}` no es válido.")
            elif idx_name in used_names:
                index_errors.append(f"El índice `{idx_name}` está repetido.")
            else:
                used_names.add(idx_name)
                index_rows.append({
                    "name": sanitize_symbol(idx_name),
                    "size": int(idx_size),
                    "elements": build_index_elements(int(idx_size), sanitize_symbol(idx_name))
                })

        for err in index_errors:
            st.error(err)

        valid_indices = len(index_errors) == 0

        index_specs = {}
        if valid_indices:
            for row in index_rows:
                index_specs[row["name"]] = {
                    "size": row["size"],
                    "elements": row["elements"]
                }
            st.session_state["model_spec"]["indices"] = index_specs
            reset_parameters_if_invalid(index_specs)
            reset_variables_if_invalid(index_specs)

        if valid_indices and len(index_specs) > 0:
            st.write("**Vista previa de índices**")
            preview_idx = []
            for idx_name, idx_spec in index_specs.items():
                preview_idx.append({
                    "Índice": idx_name,
                    "Tamaño": idx_spec["size"],
                    "Elementos": ", ".join(idx_spec["elements"])
                })
            st.dataframe(pd.DataFrame(preview_idx), use_container_width=True, hide_index=True)

    # ========================================================
    # TAB 2: PARÁMETROS
    # ========================================================
    with tab_par:
        st.markdown(
            "<div class='section-box'><div class='section-subtitle'>Configuración de parámetros</div>"
            "<div class='section-text'>Define parámetros escalares, vectores, matrices o tensores.</div></div>",
            unsafe_allow_html=True
        )

        spec = st.session_state["model_spec"]
        index_specs = spec["indices"]

        if len(index_specs) == 0:
            st.info("Primero define índices válidos.")
        else:
            current_params = spec["parameters"]
            default_num_params = max(1, len(current_params)) if len(current_params) > 0 else 1

            num_params = st.number_input(
                "Número de parámetros",
                min_value=0,
                max_value=30,
                value=default_num_params,
                step=1,
                key="num_params"
            )

            new_params_spec = {}
            index_name_options = list(index_specs.keys())

            for p in range(int(num_params)):
                st.markdown(f"#### Parámetro {p+1}")

                existing_param_names = list(current_params.keys())
                old_name = existing_param_names[p] if p < len(existing_param_names) else f"param_{p+1}"

                col1, col2 = st.columns([2, 3])

                with col1:
                    pname = st.text_input(
                        f"Nombre del parámetro {p+1}",
                        value=old_name,
                        key=f"param_name_{p}"
                    ).strip()

                with col2:
                    default_indices = current_params.get(old_name, {}).get("indices", [])
                    p_indices = st.multiselect(
                        f"Índices de {pname or f'parámetro {p+1}'}",
                        options=index_name_options,
                        default=default_indices,
                        key=f"param_indices_{p}"
                    )

                if not is_valid_symbol(pname):
                    st.error(f"El nombre `{pname}` no es válido.")
                    continue

                if pname in new_params_spec:
                    st.error(f"El parámetro `{pname}` está repetido.")
                    continue

                num_elems = total_elements_for_dims(p_indices, index_specs)
                st.write(
                    f"**Firma:** `{pretty_param_signature(pname, p_indices)}`  \n"
                    f"**Número total de elementos:** `{num_elems}`"
                )

                mode_options = ["Manual", "Aleatorio"] if num_elems <= 12 else ["Aleatorio"]

                default_mode = current_params.get(old_name, {}).get("mode", mode_options[0])
                if default_mode not in mode_options:
                    default_mode = mode_options[0]

                mode = st.radio(
                    f"Modo de carga para {pname}",
                    options=mode_options,
                    index=mode_options.index(default_mode),
                    horizontal=True,
                    key=f"param_mode_{p}"
                )

                existing_values = current_params.get(old_name, {}).get("values", {})
                param_record = {
                    "indices": p_indices,
                    "mode": mode,
                    "values": {},
                }

                if len(p_indices) == 0:
                    if mode == "Manual":
                        val = st.number_input(
                            f"Valor de {pname}",
                            value=scalar_from_values_dict(existing_values, 0.0),
                            key=f"param_scalar_manual_{p}"
                        )
                        param_record["values"] = values_dict_from_scalar(val)
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            low = st.number_input(f"Mínimo {pname}", value=0.0, key=f"param_scalar_low_{p}")
                        with c2:
                            high = st.number_input(f"Máximo {pname}", value=10.0, key=f"param_scalar_high_{p}")
                        with c3:
                            integer_mode = st.checkbox(f"Entero {pname}", value=False, key=f"param_scalar_integer_{p}")
                        with c4:
                            seed = st.number_input(f"Semilla {pname}", value=123, step=1, key=f"param_scalar_seed_{p}")

                        if low > high:
                            st.error("El mínimo no puede ser mayor que el máximo.")
                            continue

                        if st.button(f"Generar {pname}", key=f"btn_generate_scalar_{p}"):
                            st.session_state[f"generated_scalar_values_{pname}"] = random_scalar(
                                low=low, high=high, integer_mode=integer_mode, seed=int(seed)
                            )

                        generated_values = st.session_state.get(
                            f"generated_scalar_values_{pname}",
                            existing_values if existing_values else random_scalar(low, high, integer_mode, int(seed))
                        )
                        param_record["values"] = generated_values
                        st.write(f"Valor generado: **{scalar_from_values_dict(generated_values):.4f}**")

                elif len(p_indices) == 1:
                    idx = p_indices[0]
                    labels = index_specs[idx]["elements"]

                    if mode == "Manual":
                        df0 = dataframe_1d_from_values_dict(labels, existing_values)
                        edited_df = st.data_editor(
                            df0,
                            use_container_width=True,
                            num_rows="fixed",
                            hide_index=True,
                            disabled=["label"],
                            key=f"param_manual_1d_{p}"
                        )
                        param_record["values"] = values_dict_from_dataframe_1d(edited_df, labels)
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            low = st.number_input(f"Mínimo {pname}", value=0.0, key=f"param_random_low_{p}")
                        with c2:
                            high = st.number_input(f"Máximo {pname}", value=10.0, key=f"param_random_high_{p}")
                        with c3:
                            integer_mode = st.checkbox(f"Entero {pname}", value=False, key=f"param_random_integer_{p}")
                        with c4:
                            seed = st.number_input(f"Semilla {pname}", value=123, step=1, key=f"param_random_seed_{p}")

                        if low > high:
                            st.error("El mínimo no puede ser mayor que el máximo.")
                            continue

                        combos = cartesian_labels(p_indices, index_specs)

                        if st.button(f"Generar valores de {pname}", key=f"btn_generate_{p}"):
                            st.session_state[f"generated_values_{pname}"] = random_values_dict(
                                combos=combos, low=low, high=high, integer_mode=integer_mode, seed=int(seed)
                            )

                        generated_values = st.session_state.get(
                            f"generated_values_{pname}",
                            existing_values if existing_values else random_values_dict(
                                combos=combos, low=low, high=high, integer_mode=integer_mode, seed=int(seed)
                            )
                        )
                        param_record["values"] = generated_values
                        preview_df = dataframe_1d_from_values_dict(labels, generated_values)
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)

                else:
                    combos = cartesian_labels(p_indices, index_specs)

                    if mode == "Manual":
                        df0 = dataframe_nd_from_values_dict(p_indices, combos, existing_values)
                        edited_df = st.data_editor(
                            df0,
                            use_container_width=True,
                            num_rows="fixed",
                            hide_index=True,
                            disabled=list(p_indices),
                            key=f"param_manual_nd_{p}"
                        )
                        param_record["values"] = values_dict_from_dataframe_nd(edited_df, p_indices)
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            low = st.number_input(f"Mínimo {pname}", value=0.0, key=f"param_random_low_{p}")
                        with c2:
                            high = st.number_input(f"Máximo {pname}", value=10.0, key=f"param_random_high_{p}")
                        with c3:
                            integer_mode = st.checkbox(f"Entero {pname}", value=False, key=f"param_random_integer_{p}")
                        with c4:
                            seed = st.number_input(f"Semilla {pname}", value=123, step=1, key=f"param_random_seed_{p}")

                        if low > high:
                            st.error("El mínimo no puede ser mayor que el máximo.")
                            continue

                        if st.button(f"Generar valores de {pname}", key=f"btn_generate_nd_{p}"):
                            st.session_state[f"generated_values_{pname}"] = random_values_dict(
                                combos=combos, low=low, high=high, integer_mode=integer_mode, seed=int(seed)
                            )

                        generated_values = st.session_state.get(
                            f"generated_values_{pname}",
                            existing_values if existing_values else random_values_dict(
                                combos=combos, low=low, high=high, integer_mode=integer_mode, seed=int(seed)
                            )
                        )
                        param_record["values"] = generated_values
                        preview_df = dataframe_nd_from_values_dict(p_indices, combos, generated_values)
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)

                new_params_spec[pname] = param_record

            st.session_state["model_spec"]["parameters"] = new_params_spec

            if len(new_params_spec) > 0:
                st.write("**Resumen de parámetros**")
                rows = []
                for pname, pspec in new_params_spec.items():
                    rows.append({
                        "Parámetro": pretty_param_signature(pname, pspec["indices"]),
                        "Modo": pspec["mode"],
                        "Elementos": total_elements_for_dims(pspec["indices"], index_specs)
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ========================================================
    # TAB 3: VARIABLES
    # ========================================================
    with tab_var:
        st.markdown(
            "<div class='section-box'><div class='section-subtitle'>Configuración de variables</div>"
            "<div class='section-text'>Define las variables de decisión del modelo y su dominio.</div></div>",
            unsafe_allow_html=True
        )

        spec = st.session_state["model_spec"]
        index_specs = spec["indices"]

        if len(index_specs) == 0:
            st.info("Primero define índices válidos.")
        else:
            current_vars = spec["variables"]
            default_num_vars = max(1, len(current_vars)) if len(current_vars) > 0 else 1

            num_vars = st.number_input(
                "Número de variables",
                min_value=0,
                max_value=30,
                value=default_num_vars,
                step=1,
                key="num_vars"
            )

            new_vars_spec = {}
            index_name_options = list(index_specs.keys())

            for v in range(int(num_vars)):
                st.markdown(f"#### Variable {v+1}")

                existing_var_names = list(current_vars.keys())
                old_name = existing_var_names[v] if v < len(existing_var_names) else f"x_{v+1}"

                col1, col2, col3 = st.columns([2, 3, 2])

                with col1:
                    vname = st.text_input(
                        f"Nombre de la variable {v+1}",
                        value=old_name,
                        key=f"var_name_{v}"
                    ).strip()

                with col2:
                    default_indices = current_vars.get(old_name, {}).get("indices", [])
                    v_indices = st.multiselect(
                        f"Índices de {vname or f'variable {v+1}'}",
                        options=index_name_options,
                        default=default_indices,
                        key=f"var_indices_{v}"
                    )

                domain_options = ["Binary", "NonNegativeReals", "NonNegativeIntegers"]
                default_domain = current_vars.get(old_name, {}).get("domain", "NonNegativeReals")
                if default_domain not in domain_options:
                    default_domain = "NonNegativeReals"

                with col3:
                    v_domain = st.selectbox(
                        f"Dominio de {vname or f'variable {v+1}'}",
                        options=domain_options,
                        index=domain_options.index(default_domain),
                        key=f"var_domain_{v}"
                    )

                if not is_valid_symbol(vname):
                    st.error(f"El nombre `{vname}` no es válido.")
                    continue

                if vname in new_vars_spec:
                    st.error(f"La variable `{vname}` está repetida.")
                    continue

                new_vars_spec[vname] = {
                    "indices": v_indices,
                    "domain": v_domain
                }

            st.session_state["model_spec"]["variables"] = new_vars_spec

            if len(new_vars_spec) > 0:
                st.write("**Resumen de variables**")
                rows = []
                for vname, vspec in new_vars_spec.items():
                    rows.append({
                        "Variable": pretty_var_signature(vname, vspec["indices"]),
                        "Dominio": infer_domain_text(vspec["domain"]),
                        "Componentes": total_elements_for_dims(vspec["indices"], index_specs)
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif section == "Definición del modelo":

    info_hero(
        "2. Definición del modelo",
        "Defina la función objetivo, restricciones y visualice su modelo de manera matematica"
    )

    spec = st.session_state["model_spec"]
    index_specs = spec["indices"]
    var_specs = spec["variables"]

    if len(index_specs) == 0:
        st.warning("Primero debes definir al menos un índice.")
    elif len(var_specs) == 0:
        st.warning("Primero debes definir al menos una variable.")
    else:
        tab_obj, tab_rest, tab_resumen = st.tabs(
            ["Función Objetivo", "Restricciones", "Modelo matemático"]
        )

        # ====================================================
        # TAB FUNCIÓN OBJETIVO
        # ====================================================
        with tab_obj:
            current_obj = spec.get("objective", None)
            default_sense = "minimize"
            if current_obj is not None:
                default_sense = current_obj.get("sense", "minimize")

            st.markdown("### Objetivo:")

            sense_options = ["minimize", "maximize"]
            sense = st.radio(
                "",
                options=sense_options,
                index=sense_options.index(default_sense),
                format_func=lambda x: "minimize" if x == "minimize" else "maximize",
                horizontal=True,
                key="objective_sense"
            )

            obj_catalog = object_catalog(spec)
            obj_catalog_labels = [obj["label"] for obj in obj_catalog]
            obj_label_to_item = {obj["label"]: obj for obj in obj_catalog}

            old_obj_terms = []
            if current_obj is not None:
                old_obj_terms = current_obj.get("terms", [])

            num_obj_terms = st.number_input(
                "Número de términos en la función objetivo",
                min_value=1,
                max_value=20,
                value=max(1, len(old_obj_terms) if len(old_obj_terms) > 0 else 1),
                step=1,
                key="num_obj_terms"
            )

            objective_terms = []

            for t in range(int(num_obj_terms)):
                st.markdown(f"#### Término FO {t+1}")

                old_term = old_obj_terms[t] if t < len(old_obj_terms) else None

                c1, c2, c3 = st.columns([1, 2, 2])

                with c1:
                    sign = st.selectbox(
                        f"Signo término {t+1}",
                        options=["+", "-"],
                        index=0 if old_term is None else (0 if old_term.get("sign", "+") == "+" else 1),
                        key=f"obj_term_sign_{t}"
                    )

                with c2:
                    num_factors = st.number_input(
                        f"Número de factores del término {t+1}",
                        min_value=1,
                        max_value=4,
                        value=2 if old_term is None else max(1, len(old_term.get('factors', []))),
                        step=1,
                        key=f"obj_num_factors_{t}"
                    )

                with c3:
                    default_sum_over = [] if old_term is None else old_term.get("sum_over", [])
                    sum_over = st.multiselect(
                        f"Sumar sobre índices en término {t+1}",
                        options=list(index_specs.keys()),
                        default=default_sum_over,
                        key=f"obj_sum_over_{t}"
                    )

                factors = []
                old_factors = [] if old_term is None else old_term.get("factors", [])

                for f in range(int(num_factors)):
                    st.markdown(f"**Factor {f+1} del término {t+1}**")

                    fc1, fc2, fc3 = st.columns([1.5, 2.5, 2])

                    old_factor = old_factors[f] if f < len(old_factors) else None
                    default_factor_type = "object" if old_factor is None else old_factor.get("type", "object")

                    with fc1:
                        factor_type = st.selectbox(
                            f"Tipo factor {f+1} término {t+1}",
                            options=["object", "constant"],
                            index=0 if default_factor_type == "object" else 1,
                            format_func=lambda x: "Parámetro/Variable" if x == "object" else "Constante",
                            key=f"obj_factor_type_{t}_{f}"
                        )

                    if factor_type == "object":
                        if len(obj_catalog_labels) == 0:
                            st.error("No hay parámetros ni variables disponibles para usar en la FO.")
                        else:
                            default_label = obj_catalog_labels[0]
                            if old_factor is not None and old_factor.get("type") == "object":
                                old_lbl = old_factor.get("label", default_label)
                                if old_lbl in obj_catalog_labels:
                                    default_label = old_lbl

                            with fc2:
                                chosen_label = st.selectbox(
                                    f"Objeto factor {f+1} término {t+1}",
                                    options=obj_catalog_labels,
                                    index=obj_catalog_labels.index(default_label),
                                    key=f"obj_factor_object_{t}_{f}"
                                )

                            item = obj_label_to_item[chosen_label]
                            factors.append({
                                "type": "object",
                                "kind": item["kind"],
                                "name": item["name"],
                                "indices": item["indices"],
                                "label": item["label"]
                            })

                            with fc3:
                                st.write(f"Índices: {', '.join(item['indices']) if len(item['indices']) > 0 else 'ninguno'}")
                    else:
                        default_value = 1.0
                        if old_factor is not None and old_factor.get("type") == "constant":
                            default_value = float(old_factor.get("value", 1.0))

                        with fc2:
                            const_val = st.number_input(
                                f"Valor constante factor {f+1} término {t+1}",
                                value=default_value,
                                key=f"obj_factor_const_{t}_{f}"
                            )

                        factors.append({
                            "type": "constant",
                            "value": float(const_val)
                        })

                term_record = {
                    "sign": sign,
                    "factors": factors,
                    "sum_over": sum_over
                }
                objective_terms.append(term_record)

                st.write("**Vista previa del término:**")
                st.latex(build_term_latex(term_record))

            objective_errors = validate_objective_terms(objective_terms)

            if len(objective_errors) > 0:
                for err in objective_errors:
                    st.error(err)
            else:
                st.success("La función objetivo es estructuralmente consistente.")

            spec["objective"] = {
                "sense": sense,
                "terms": objective_terms
            }

        # ====================================================
        # TAB RESTRICCIONES
        # ====================================================
        with tab_rest:
            old_constraint_families = spec.get("constraints", [])

            num_families = st.number_input(
                "Número de familias de restricciones",
                min_value=0,
                max_value=30,
                value=len(old_constraint_families),
                step=1,
                key="num_constraint_families"
            )

            expr_catalog = object_catalog(spec)
            expr_catalog_labels = [obj["label"] for obj in expr_catalog]
            expr_label_to_item = {obj["label"]: obj for obj in expr_catalog}

            new_constraint_families = []

            for r in range(int(num_families)):
                st.markdown(f"### Familia de restricción {r+1}")

                old_family = old_constraint_families[r] if r < len(old_constraint_families) else None

                c1, c2, c3 = st.columns([2, 2, 2])

                with c1:
                    default_name = f"R{r+1}" if old_family is None else old_family.get("name", f"R{r+1}")
                    family_name = st.text_input(
                        f"Nombre de la familia {r+1}",
                        value=default_name,
                        key=f"constraint_family_name_{r}"
                    ).strip()

                with c2:
                    default_forall = [] if old_family is None else old_family.get("forall", [])
                    forall = st.multiselect(
                        f"Índices para 'para todo' en {family_name or f'familia {r+1}'}",
                        options=list(index_specs.keys()),
                        default=default_forall,
                        key=f"constraint_forall_{r}"
                    )

                with c3:
                    default_sense = "<=" if old_family is None else old_family.get("sense", "<=")
                    sense_family = st.selectbox(
                        f"Operador en {family_name or f'familia {r+1}'}",
                        options=["<=", ">=", "="],
                        index=["<=", ">=", "="].index(default_sense),
                        key=f"constraint_sense_{r}"
                    )

                if not is_valid_symbol(family_name):
                    st.error(f"El nombre `{family_name}` no es válido.")
                    continue

                old_lhs_terms = [] if old_family is None else old_family.get("lhs_terms", [])
                old_rhs_terms = [] if old_family is None else old_family.get("rhs_terms", [])

                colL, colR = st.columns(2)

                with colL:
                    st.markdown(f"#### Lado izquierdo de {family_name}")

                    num_lhs_terms = st.number_input(
                        f"Número de términos LHS en {family_name}",
                        min_value=0,
                        max_value=10,
                        value=len(old_lhs_terms),
                        step=1,
                        key=f"num_lhs_terms_{r}"
                    )

                    lhs_terms = []

                    for t in range(int(num_lhs_terms)):
                        st.markdown(f"**LHS término {t+1}**")
                        old_term = old_lhs_terms[t] if t < len(old_lhs_terms) else None

                        ca, cb, cc = st.columns([1, 2, 2])

                        with ca:
                            sign = st.selectbox(
                                f"Signo LHS término {t+1} en {family_name}",
                                options=["+", "-"],
                                index=0 if old_term is None else (0 if old_term.get("sign", "+") == "+" else 1),
                                key=f"lhs_sign_{r}_{t}"
                            )

                        with cb:
                            num_factors = st.number_input(
                                f"Número de factores LHS término {t+1} en {family_name}",
                                min_value=1,
                                max_value=4,
                                value=2 if old_term is None else max(1, len(old_term.get('factors', []))),
                                step=1,
                                key=f"lhs_num_factors_{r}_{t}"
                            )

                        with cc:
                            default_sum_over = [] if old_term is None else old_term.get("sum_over", [])
                            sum_over = st.multiselect(
                                f"Sumar sobre índices LHS término {t+1} en {family_name}",
                                options=list(index_specs.keys()),
                                default=default_sum_over,
                                key=f"lhs_sum_over_{r}_{t}"
                            )

                        factors = []
                        old_factors = [] if old_term is None else old_term.get("factors", [])

                        for f in range(int(num_factors)):
                            cfa, cfb, cfc = st.columns([1.5, 2.5, 2])

                            old_factor = old_factors[f] if f < len(old_factors) else None
                            default_factor_type = "object" if old_factor is None else old_factor.get("type", "object")

                            with cfa:
                                factor_type = st.selectbox(
                                    f"Tipo LHS factor {f+1} término {t+1} en {family_name}",
                                    options=["object", "constant"],
                                    index=0 if default_factor_type == "object" else 1,
                                    format_func=lambda x: "Parámetro/Variable" if x == "object" else "Constante",
                                    key=f"lhs_factor_type_{r}_{t}_{f}"
                                )

                            if factor_type == "object":
                                if len(expr_catalog_labels) == 0:
                                    st.error("No hay parámetros ni variables disponibles.")
                                else:
                                    default_label = expr_catalog_labels[0]
                                    if old_factor is not None and old_factor.get("type") == "object":
                                        old_lbl = old_factor.get("label", default_label)
                                        if old_lbl in expr_catalog_labels:
                                            default_label = old_lbl

                                    with cfb:
                                        chosen_label = st.selectbox(
                                            f"Objeto LHS factor {f+1} término {t+1} en {family_name}",
                                            options=expr_catalog_labels,
                                            index=expr_catalog_labels.index(default_label),
                                            key=f"lhs_factor_object_{r}_{t}_{f}"
                                        )

                                    item = expr_label_to_item[chosen_label]
                                    factors.append({
                                        "type": "object",
                                        "kind": item["kind"],
                                        "name": item["name"],
                                        "indices": item["indices"],
                                        "label": item["label"]
                                    })

                                    with cfc:
                                        st.write(f"Índices: {', '.join(item['indices']) if len(item['indices']) > 0 else 'ninguno'}")
                            else:
                                default_value = 0.0
                                if old_factor is not None and old_factor.get("type") == "constant":
                                    default_value = float(old_factor.get("value", 0.0))

                                with cfb:
                                    const_val = st.number_input(
                                        f"Valor constante LHS factor {f+1} término {t+1} en {family_name}",
                                        value=default_value,
                                        key=f"lhs_factor_const_{r}_{t}_{f}"
                                    )

                                factors.append({
                                    "type": "constant",
                                    "value": float(const_val)
                                })

                        term_record = {
                            "sign": sign,
                            "factors": factors,
                            "sum_over": sum_over
                        }
                        lhs_terms.append(term_record)
                        st.latex(build_term_latex(term_record))

                with colR:
                    st.markdown(f"#### Lado derecho de {family_name}")

                    num_rhs_terms = st.number_input(
                        f"Número de términos RHS en {family_name}",
                        min_value=0,
                        max_value=10,
                        value=len(old_rhs_terms),
                        step=1,
                        key=f"num_rhs_terms_{r}"
                    )

                    rhs_terms = []

                    for t in range(int(num_rhs_terms)):
                        st.markdown(f"**RHS término {t+1}**")
                        old_term = old_rhs_terms[t] if t < len(old_rhs_terms) else None

                        ca, cb, cc = st.columns([1, 2, 2])

                        with ca:
                            sign = st.selectbox(
                                f"Signo RHS término {t+1} en {family_name}",
                                options=["+", "-"],
                                index=0 if old_term is None else (0 if old_term.get("sign", "+") == "+" else 1),
                                key=f"rhs_sign_{r}_{t}"
                            )

                        with cb:
                            num_factors = st.number_input(
                                f"Número de factores RHS término {t+1} en {family_name}",
                                min_value=1,
                                max_value=4,
                                value=1 if old_term is None else max(1, len(old_term.get('factors', []))),
                                step=1,
                                key=f"rhs_num_factors_{r}_{t}"
                            )

                        with cc:
                            default_sum_over = [] if old_term is None else old_term.get("sum_over", [])
                            sum_over = st.multiselect(
                                f"Sumar sobre índices RHS término {t+1} en {family_name}",
                                options=list(index_specs.keys()),
                                default=default_sum_over,
                                key=f"rhs_sum_over_{r}_{t}"
                            )

                        factors = []
                        old_factors = [] if old_term is None else old_term.get("factors", [])

                        for f in range(int(num_factors)):
                            cfa, cfb, cfc = st.columns([1.5, 2.5, 2])

                            old_factor = old_factors[f] if f < len(old_factors) else None
                            default_factor_type = "constant" if old_factor is None else old_factor.get("type", "constant")

                            with cfa:
                                factor_type = st.selectbox(
                                    f"Tipo RHS factor {f+1} término {t+1} en {family_name}",
                                    options=["object", "constant"],
                                    index=0 if default_factor_type == "object" else 1,
                                    format_func=lambda x: "Parámetro/Variable" if x == "object" else "Constante",
                                    key=f"rhs_factor_type_{r}_{t}_{f}"
                                )

                            if factor_type == "object":
                                if len(expr_catalog_labels) == 0:
                                    st.error("No hay parámetros ni variables disponibles.")
                                else:
                                    default_label = expr_catalog_labels[0]
                                    if old_factor is not None and old_factor.get("type") == "object":
                                        old_lbl = old_factor.get("label", default_label)
                                        if old_lbl in expr_catalog_labels:
                                            default_label = old_lbl

                                    with cfb:
                                        chosen_label = st.selectbox(
                                            f"Objeto RHS factor {f+1} término {t+1} en {family_name}",
                                            options=expr_catalog_labels,
                                            index=expr_catalog_labels.index(default_label),
                                            key=f"rhs_factor_object_{r}_{t}_{f}"
                                        )

                                    item = expr_label_to_item[chosen_label]
                                    factors.append({
                                        "type": "object",
                                        "kind": item["kind"],
                                        "name": item["name"],
                                        "indices": item["indices"],
                                        "label": item["label"]
                                    })

                                    with cfc:
                                        st.write(f"Índices: {', '.join(item['indices']) if len(item['indices']) > 0 else 'ninguno'}")
                            else:
                                default_value = 0.0
                                if old_factor is not None and old_factor.get("type") == "constant":
                                    default_value = float(old_factor.get("value", 0.0))

                                with cfb:
                                    const_val = st.number_input(
                                        f"Valor constante RHS factor {f+1} término {t+1} en {family_name}",
                                        value=default_value,
                                        key=f"rhs_factor_const_{r}_{t}_{f}"
                                    )

                                factors.append({
                                    "type": "constant",
                                    "value": float(const_val)
                                })

                        term_record = {
                            "sign": sign,
                            "factors": factors,
                            "sum_over": sum_over
                        }
                        rhs_terms.append(term_record)
                        st.latex(build_term_latex(term_record))

                family_record = {
                    "name": family_name,
                    "forall": forall,
                    "sense": sense_family,
                    "lhs_terms": lhs_terms,
                    "rhs_terms": rhs_terms
                }

                family_errors = validate_constraint_family(family_record)

                st.markdown(f"### Vista previa de la familia {family_name}")
                st.latex(build_constraint_family_latex(family_record))

                if len(family_errors) > 0:
                    for err in family_errors:
                        st.error(err)
                else:
                    st.success("La familia de restricciones está estructuralmente consistente.")

                new_constraint_families.append(family_record)

            spec["constraints"] = new_constraint_families

        # ====================================================
        # TAB RESUMEN LATEX
        # ====================================================
        with tab_resumen:
            st.markdown("### Modelo estructurado")

            if spec["objective"] is not None:
                sense_resume = spec["objective"]["sense"]
                terms_resume = spec["objective"]["terms"]
                sense_symbol_resume = r"\min" if sense_resume == "minimize" else r"\max"
                st.latex(rf"{sense_symbol_resume}\ Z = {build_expression_latex(terms_resume)}")
            else:
                st.info("Aún no has definido la función objetivo.")

            st.markdown("**Sujeto a:**")
            if len(spec["constraints"]) == 0:
                st.info("No hay familias de restricciones definidas.")
            else:
                for fam in spec["constraints"]:
                    st.latex(build_constraint_family_latex(fam))
elif section == "Salidas del modelo":

    info_hero(
        "3.Resultados",
        "Se muestra la solución óptima y la configuración de las variables"
    )

    spec = st.session_state["model_spec"]

    validation_errors = []

    if spec["objective"] is None:
        validation_errors.append("No hay función objetivo definida.")
    else:
        validation_errors.extend(validate_objective_terms(spec["objective"].get("terms", [])))

    for fam in spec.get("constraints", []):
        validation_errors.extend(validate_constraint_family(fam))

    validation_errors.extend(validate_linearity_of_model(spec))

    if len(spec["variables"]) == 0:
        validation_errors.append("No hay variables definidas.")

    if len(spec["indices"]) == 0:
        validation_errors.append("No hay índices definidos.")

    if len(validation_errors) > 0:
        for err in validation_errors:
            st.error(err)
        st.stop()
    else:
        st.success("La especificación es válida para intentar construir y resolver el modelo.")

    tab_resolver, tab_vars = st.tabs(
        ["Resolver", "Variables solución"]
    )

    with tab_resolver:
        st.subheader("Resolver modelo")

        solver_name = "appsi_highs"

        solve_button = st.button("Resolver modelo", type="primary")

        if solve_button:
            try:
                model = build_pyomo_model_from_spec(spec)
                solver = solver_factory_from_name(solver_name)
                result = solver.solve(model)

                termination = str(result.solver.termination_condition)
                status = str(result.solver.status)
                obj_value = pyo.value(model.OBJ)

                st.session_state["model_spec"]["results"] = {
                    "solver_name": solver_name,
                    "termination_condition": termination,
                    "status": status,
                    "objective_value": obj_value,
                }

                st.session_state["solved_model_object"] = model

                st.success("Modelo resuelto correctamente.")

            except Exception as e:
                st.error(f"Error al construir o resolver el modelo: {e}")
                st.stop()

        results = st.session_state["model_spec"].get("results", None)
        solved_model = st.session_state.get("solved_model_object", None)

        if results is None or solved_model is None:
            st.info("Aún no has resuelto el modelo.")
        else:
            c2, c3 = st.columns(2)
            with c2:
                result_card("Status", results["status"])
            with c3:
                result_card("Termination", results["termination_condition"])

            result_wide_card("Valor óptimo", f"{results['objective_value']:,.6f}")

    with tab_vars:
        results = st.session_state["model_spec"].get("results", None)
        solved_model = st.session_state.get("solved_model_object", None)

        if results is None or solved_model is None:
            st.info("Primero resuelve el modelo.")
        else:
            subtab_var, subtab_nz = st.tabs(["Seleccionar variable", "Variables no nulas"])

            with subtab_var:
                st.subheader("Solución por variable")

                variable_names = list(spec["variables"].keys())
                selected_var = st.selectbox(
                    "Selecciona una variable para ver su solución",
                    options=variable_names
                )

                var_df = variable_solution_to_dataframe(
                    solved_model,
                    selected_var,
                    spec["variables"][selected_var],
                    spec["indices"]
                )
                st.dataframe(var_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "Descargar solución de la variable seleccionada en CSV",
                    data=var_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_var}_solucion.csv",
                    mime="text/csv"
                )

            with subtab_nz:
                st.subheader("Variables no nulas")

                nz_df = nonzero_variables_solution_flat(solved_model, spec, tol=1e-9)

                if nz_df.empty:
                    st.info("No hay variables no nulas.")
                else:
                    st.dataframe(nz_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "Descargar variables no nulas en CSV",
                    data=nz_df.to_csv(index=False).encode("utf-8"),
                    file_name="variables_no_nulas.csv",
                    mime="text/csv"
                )
