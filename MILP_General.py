import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Constructor de Modelos Algebraicos",
    page_icon="📐",
    layout="wide",
)

SOLVER_OPTIONS = ["appsi_highs", "glpk", "cbc"]
DOMAIN_OPTIONS = ["Binary", "NonNegativeReals", "NonNegativeIntegers"]


# ============================================================
# ESTILOS
# ============================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2.0rem;
                max-width: 1450px;
            }

            .main-title {
                font-size: 2.25rem;
                font-weight: 800;
                margin-bottom: 0.15rem;
                line-height: 1.1;
            }

            .main-subtitle {
                color: #9ca3af;
                font-size: 0.98rem;
                margin-bottom: 1.25rem;
            }

            .section-box {
                background: linear-gradient(180deg, rgba(10,17,33,0.94), rgba(4,10,22,0.96));
                border: 1px solid rgba(86, 110, 158, 0.30);
                border-radius: 18px;
                padding: 1.1rem 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.18);
            }

            .section-title {
                font-size: 1.15rem;
                font-weight: 750;
                margin-bottom: 0.3rem;
                color: #f3f4f6;
            }

            .section-text {
                color: #cbd5e1;
                font-size: 0.98rem;
                line-height: 1.5;
            }

            .metric-card {
                background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(5,10,20,0.96));
                border: 1px solid rgba(110, 126, 160, 0.25);
                border-radius: 16px;
                padding: 1rem;
            }

            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(5,10,20,0.96));
                border: 1px solid rgba(110,126,160,0.22);
                border-radius: 16px;
                padding: 0.9rem 1rem;
            }

            .small-note {
                color: #94a3b8;
                font-size: 0.9rem;
            }

            .pill {
                display: inline-block;
                padding: 0.22rem 0.6rem;
                border-radius: 999px;
                font-size: 0.8rem;
                border: 1px solid rgba(148,163,184,0.20);
                background: rgba(255,255,255,0.04);
                margin-right: 0.35rem;
            }

            .watermark {
                position: fixed;
                top: 16px;
                right: 22px;
                z-index: 9999;
                color: #ff4b4b;
                opacity: 0.95;
                font-weight: 900;
                font-size: 0.95rem;
                text-shadow: 1px 1px 2px #000;
                pointer-events: none;
            }

            section[data-testid="stSidebar"] {
                border-right: 1px solid rgba(148,163,184,0.10);
            }
        </style>

        <div class="watermark">by M.Sc. Dilan Mogollón</div>
        """,
        unsafe_allow_html=True,
    )


inject_css()


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
        },
        "solved_model_object": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ============================================================
# BLOQUES VISUALES
# ============================================================

def section_banner(title: str, text: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-box">
            <div class="section-title">{title}</div>
            <div class="section-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_header() -> None:
    st.markdown('<div class="main-title">Constructor de Modelos Algebraicos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Definición estructural, construcción y resolución de modelos en Pyomo.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def is_valid_symbol(name: str) -> bool:
    if not isinstance(name, str):
        return False
    name = name.strip()
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in name[1:])


def sanitize_symbol(name: str) -> str:
    return name.strip()


def build_index_elements(size: int, prefix: str) -> List[str]:
    return [f"{prefix}{i}" for i in range(1, size + 1)]


def ordered_unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def total_elements_for_dims(index_names: List[str], index_specs: Dict[str, Dict[str, Any]]) -> int:
    if not index_names:
        return 1
    total = 1
    for idx in index_names:
        total *= int(index_specs[idx]["size"])
    return total


def cartesian_labels(index_names: List[str], index_specs: Dict[str, Dict[str, Any]]) -> List[Tuple[str, ...]]:
    if not index_names:
        return [tuple()]
    arrays = [index_specs[idx]["elements"] for idx in index_names]
    return list(itertools.product(*arrays))


def pretty_signature(name: str, index_names: List[str]) -> str:
    return name if not index_names else f"{name}[{', '.join(index_names)}]"


def infer_domain_text(domain_code: str) -> str:
    mapper = {
        "Binary": "Binarias",
        "NonNegativeReals": "Reales ≥ 0",
        "NonNegativeIntegers": "Enteras ≥ 0",
    }
    return mapper.get(domain_code, domain_code)


def reset_parameters_if_invalid(index_specs: Dict[str, Dict[str, Any]]) -> None:
    valid_index_names = set(index_specs.keys())
    cleaned = {}
    for name, spec in st.session_state["model_spec"]["parameters"].items():
        if all(idx in valid_index_names for idx in spec.get("indices", [])):
            cleaned[name] = spec
    st.session_state["model_spec"]["parameters"] = cleaned


def reset_variables_if_invalid(index_specs: Dict[str, Dict[str, Any]]) -> None:
    valid_index_names = set(index_specs.keys())
    cleaned = {}
    for name, spec in st.session_state["model_spec"]["variables"].items():
        if all(idx in valid_index_names for idx in spec.get("indices", [])):
            cleaned[name] = spec
    st.session_state["model_spec"]["variables"] = cleaned


# ============================================================
# UTILIDADES DE DATOS
# ============================================================

def values_dict_from_scalar(value: float) -> Dict[str, float]:
    return {"__scalar__": float(value)}


def scalar_from_values_dict(values: Dict[str, float], default: float = 0.0) -> float:
    return float(values.get("__scalar__", default))


def dataframe_1d_from_values_dict(labels: List[str], values_dict: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": labels,
            "value": [float(values_dict.get(str((lbl,)), 0.0)) for lbl in labels],
        }
    )


def values_dict_from_dataframe_1d(df: pd.DataFrame, labels: List[str]) -> Dict[str, float]:
    out = {}
    for lbl in labels:
        val = df.loc[df["label"] == lbl, "value"].iloc[0]
        out[str((lbl,))] = float(val)
    return out


def dataframe_nd_from_values_dict(
    index_names: List[str],
    combos: List[Tuple[str, ...]],
    values_dict: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for combo in combos:
        row = {idx: combo[pos] for pos, idx in enumerate(index_names)}
        row["value"] = float(values_dict.get(str(combo), 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def values_dict_from_dataframe_nd(df: pd.DataFrame, index_names: List[str]) -> Dict[str, float]:
    out = {}
    for _, row in df.iterrows():
        key = tuple(str(row[idx]) for idx in index_names)
        out[str(key)] = float(row["value"])
    return out


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
            value = int(rng.integers(int(low), int(high) + 1))
        else:
            value = float(rng.uniform(low, high))
        out[str(combo)] = float(value)
    return out


def random_scalar(low: float, high: float, integer_mode: bool, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    value = int(rng.integers(int(low), int(high) + 1)) if integer_mode else float(rng.uniform(low, high))
    return {"__scalar__": float(value)}


# ============================================================
# CATALOGO DE OBJETOS
# ============================================================

def object_catalog(model_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for pname, pspec in model_spec["parameters"].items():
        items.append(
            {
                "kind": "parameter",
                "name": pname,
                "indices": pspec["indices"],
                "label": pretty_signature(pname, pspec["indices"]),
            }
        )
    for vname, vspec in model_spec["variables"].items():
        items.append(
            {
                "kind": "variable",
                "name": vname,
                "indices": vspec["indices"],
                "label": pretty_signature(vname, vspec["indices"]),
            }
        )
    return items


# ============================================================
# TEXTO / LATEX
# ============================================================

def format_number_latex(val: float) -> str:
    val = float(val)
    return str(int(val)) if val.is_integer() else f"{val:.4f}"


def factor_to_text(factor: Dict[str, Any]) -> str:
    if factor["type"] == "object":
        return factor["label"]
    return str(factor["value"])


def factor_to_latex(factor: Dict[str, Any]) -> str:
    if factor["type"] == "constant":
        return format_number_latex(factor["value"])
    name = factor["name"]
    idxs = factor["indices"]
    return name if not idxs else rf"{name}_{{{','.join(idxs)}}}"


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


def build_term_latex(term: Dict[str, Any]) -> str:
    factors = term.get("factors", [])
    body = r" \cdot ".join(factor_to_latex(f) for f in factors) if factors else "0"
    if term.get("sum_over"):
        sums = " ".join([rf"\sum_{{{idx}}}" for idx in term["sum_over"]])
        body = f"{sums}\\left({body}\\right)"
    sign = "- " if term.get("sign", "+") == "-" else "+ "
    return f"{sign}{body}"


def build_expression_latex(terms: List[Dict[str, Any]]) -> str:
    if not terms:
        return "0"
    expr = " ".join(build_term_latex(t) for t in terms).strip()
    return expr[2:] if expr.startswith("+ ") else expr


def build_constraint_family_latex(family: Dict[str, Any]) -> str:
    lhs = build_expression_latex(family.get("lhs_terms", []))
    rhs = build_expression_latex(family.get("rhs_terms", []))
    sense_map = {"<=": r"\leq", ">=": r"\geq", "=": "="}
    out = f"{lhs} {sense_map[family['sense']]} {rhs}"
    if family.get("forall"):
        out += r"\quad \forall " + ", ".join(family["forall"])
    return out


# ============================================================
# VALIDACIÓN
# ============================================================

def validate_objective_terms(terms: List[Dict[str, Any]]) -> List[str]:
    errors = []
    for pos, term in enumerate(terms, start=1):
        free = term_free_indices(term)
        if free:
            errors.append(
                f"En el término {pos} de la función objetivo quedaron índices libres: {', '.join(free)}."
            )
    return errors


def validate_constraint_family(family: Dict[str, Any]) -> List[str]:
    errors = []
    lhs_free = ordered_unique([idx for t in family.get("lhs_terms", []) for idx in term_free_indices(t)])
    rhs_free = ordered_unique([idx for t in family.get("rhs_terms", []) for idx in term_free_indices(t)])
    forall = family.get("forall", [])

    lhs_is_constant = len(lhs_free) == 0
    rhs_is_constant = len(rhs_free) == 0

    if not lhs_is_constant and not rhs_is_constant:
        if lhs_free != rhs_free:
            errors.append(
                f"Los índices libres del lado izquierdo ({lhs_free}) no coinciden con los del lado derecho ({rhs_free})."
            )
        if lhs_free != forall:
            errors.append(
                f"Los índices libres de la restricción ({lhs_free}) no coinciden con el 'para todo' ({forall})."
            )
    elif lhs_is_constant and not rhs_is_constant:
        if rhs_free != forall:
            errors.append(
                f"El lado izquierdo es constante; los índices libres del lado derecho ({rhs_free}) deben coincidir con el 'para todo' ({forall})."
            )
    elif not lhs_is_constant and rhs_is_constant:
        if lhs_free != forall:
            errors.append(
                f"El lado derecho es constante; los índices libres del lado izquierdo ({lhs_free}) deben coincidir con el 'para todo' ({forall})."
            )
    else:
        if forall:
            errors.append(
                f"Ambos lados son constantes, por lo que no debería existir 'para todo' ({forall})."
            )

    return errors


def count_variable_factors(term: Dict[str, Any]) -> int:
    return sum(
        1
        for fac in term.get("factors", [])
        if fac["type"] == "object" and fac.get("kind") == "variable"
    )


def validate_linearity_of_term(term: Dict[str, Any], context: str = "") -> List[str]:
    n = count_variable_factors(term)
    if n > 1:
        return [f"{context} tiene {n} factores variables en un mismo término. Eso genera no linealidad."]
    return []


def validate_linearity_of_model(spec: Dict[str, Any]) -> List[str]:
    errors = []
    obj = spec.get("objective")
    if obj is not None:
        for i, term in enumerate(obj.get("terms", []), start=1):
            errors.extend(validate_linearity_of_term(term, f"FO término {i}"))
    for r, fam in enumerate(spec.get("constraints", []), start=1):
        rname = fam.get("name", f"R{r}")
        for i, term in enumerate(fam.get("lhs_terms", []), start=1):
            errors.extend(validate_linearity_of_term(term, f"Restricción {rname}, LHS término {i}"))
        for i, term in enumerate(fam.get("rhs_terms", []), start=1):
            errors.extend(validate_linearity_of_term(term, f"Restricción {rname}, RHS término {i}"))
    return errors


# ============================================================
# PYOMO
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

    name = factor["name"]
    idxs = factor["indices"]
    kind = factor["kind"]

    comp = getattr(model, f"par_{name}") if kind == "parameter" else getattr(model, f"var_{name}")

    if not idxs:
        return comp

    key = tuple(env[idx] for idx in idxs)
    return comp[key[0]] if len(key) == 1 else comp[key]


def build_base_term_value(model, term: Dict[str, Any], env: Dict[str, Any]):
    factors = term.get("factors", [])
    val = 0.0 if not factors else 1
    for fac in factors:
        val *= get_component_value(model, fac, env)
    if term.get("sign", "+") == "-":
        val = -val
    return val


def evaluate_term_with_sums(model, term: Dict[str, Any], env: Dict[str, Any]):
    sum_over = term.get("sum_over", [])

    def recurse(pos: int, local_env: Dict[str, Any]):
        if pos == len(sum_over):
            return build_base_term_value(model, term, local_env)
        idx_name = sum_over[pos]
        pyomo_set = getattr(model, f"set_{idx_name}")
        return sum(recurse(pos + 1, {**local_env, idx_name: idx_val}) for idx_val in pyomo_set)

    return recurse(0, dict(env))


def build_expression_pyomo(model, terms: List[Dict[str, Any]], env: Dict[str, Any]):
    return sum(evaluate_term_with_sums(model, term, env) for term in terms) if terms else 0


def build_pyomo_model_from_spec(spec: Dict[str, Any]):
    model = pyo.ConcreteModel()
    index_specs = spec["indices"]

    for idx_name, idx_spec in index_specs.items():
        setattr(model, f"set_{idx_name}", pyo.Set(initialize=idx_spec["elements"], ordered=True))

    for pname, pspec in spec["parameters"].items():
        idxs = pspec["indices"]
        values = pspec["values"]

        if not idxs:
            comp = pyo.Param(initialize=float(values["__scalar__"]), mutable=False)
        else:
            pyomo_sets = [getattr(model, f"set_{idx}") for idx in idxs]
            init_dict = {}
            for combo in cartesian_labels(idxs, index_specs):
                val = float(values.get(str(combo), 0.0))
                init_dict[combo[0] if len(combo) == 1 else combo] = val
            comp = pyo.Param(*pyomo_sets, initialize=init_dict, mutable=False)

        setattr(model, f"par_{pname}", comp)

    for vname, vspec in spec["variables"].items():
        idxs = vspec["indices"]
        domain = pyomo_domain_from_code(vspec["domain"])
        if not idxs:
            comp = pyo.Var(domain=domain)
        else:
            pyomo_sets = [getattr(model, f"set_{idx}") for idx in idxs]
            comp = pyo.Var(*pyomo_sets, domain=domain)
        setattr(model, f"var_{vname}", comp)

    obj = spec.get("objective")
    if obj is None:
        raise ValueError("No hay función objetivo definida.")

    obj_expr = build_expression_pyomo(model, obj["terms"], env={})
    obj_sense = pyo.minimize if obj["sense"] == "minimize" else pyo.maximize
    model.OBJ = pyo.Objective(expr=obj_expr, sense=obj_sense)

    for c_idx, fam in enumerate(spec.get("constraints", []), start=1):
        fname = fam.get("name", f"R{c_idx}")
        lhs_terms = fam.get("lhs_terms", [])
        rhs_terms = fam.get("rhs_terms", [])
        forall = fam.get("forall", [])
        sense_f = fam.get("sense", "<=")

        if not forall:
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
                    return lhs_expr == rhs_expr
                return _rule

            con = pyo.Constraint(*pyomo_sets, rule=make_rule(lhs_terms, rhs_terms, forall, sense_f))

        setattr(model, f"con_{fname}", con)

    return model


def solver_factory_from_name(solver_name: str):
    if solver_name == "appsi_highs":
        return pyo.SolverFactory("appsi_highs")
    if solver_name == "glpk":
        return pyo.SolverFactory("glpk")
    if solver_name == "cbc":
        return pyo.SolverFactory("cbc")
    raise ValueError(f"Solver no soportado: {solver_name}")


def variable_solution_to_dataframe(
    model,
    vname: str,
    vspec: Dict[str, Any],
    index_specs: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    comp = getattr(model, f"var_{vname}")
    idxs = vspec["indices"]

    if not idxs:
        return pd.DataFrame({"variable": [vname], "value": [pyo.value(comp)]})

    rows = []
    for combo in cartesian_labels(idxs, index_specs):
        value = pyo.value(comp[combo[0]]) if len(combo) == 1 else pyo.value(comp[combo])
        row = {idx: combo[pos] for pos, idx in enumerate(idxs)}
        row["value"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def all_variables_solution_flat(model, spec: Dict[str, Any]) -> pd.DataFrame:
    pieces = []
    for vname, vspec in spec["variables"].items():
        df = variable_solution_to_dataframe(model, vname, vspec, spec["indices"])
        df.insert(0, "variable_name", vname)
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def nonzero_variables_solution_flat(model, spec: Dict[str, Any], tol: float = 1e-9) -> pd.DataFrame:
    df = all_variables_solution_flat(model, spec)
    return df[df["value"].abs() > tol].reset_index(drop=True) if not df.empty else df


# ============================================================
# RENDER HELPERS
# ============================================================

def render_spec_summary(spec: Dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Índices", len(spec["indices"]))
    with c2:
        st.metric("Parámetros", len(spec["parameters"]))
    with c3:
        st.metric("Variables", len(spec["variables"]))


def render_factor_builder(
    prefix: str,
    term_idx: int,
    factor_idx: int,
    catalog_labels: List[str],
    label_to_item: Dict[str, Dict[str, Any]],
    old_factor: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    default_type = "object" if old_factor is None else old_factor.get("type", "object")
    c1, c2, c3 = st.columns([1.25, 2.4, 2.1])

    with c1:
        factor_type = st.selectbox(
            f"Tipo factor {factor_idx + 1}",
            options=["object", "constant"],
            index=0 if default_type == "object" else 1,
            format_func=lambda x: "Parámetro/Variable" if x == "object" else "Constante",
            key=f"{prefix}_factor_type_{term_idx}_{factor_idx}",
        )

    if factor_type == "object":
        if not catalog_labels:
            st.error("No hay parámetros ni variables disponibles.")
            return {"type": "constant", "value": 0.0}

        default_label = catalog_labels[0]
        if old_factor is not None and old_factor.get("type") == "object":
            old_label = old_factor.get("label", default_label)
            if old_label in catalog_labels:
                default_label = old_label

        with c2:
            chosen_label = st.selectbox(
                f"Objeto factor {factor_idx + 1}",
                options=catalog_labels,
                index=catalog_labels.index(default_label),
                key=f"{prefix}_factor_object_{term_idx}_{factor_idx}",
            )

        item = label_to_item[chosen_label]
        with c3:
            st.caption(f"Índices: {', '.join(item['indices']) if item['indices'] else 'ninguno'}")

        return {
            "type": "object",
            "kind": item["kind"],
            "name": item["name"],
            "indices": item["indices"],
            "label": item["label"],
        }

    default_value = 1.0 if old_factor is None else float(old_factor.get("value", 1.0))
    with c2:
        const_val = st.number_input(
            f"Valor factor {factor_idx + 1}",
            value=default_value,
            key=f"{prefix}_factor_const_{term_idx}_{factor_idx}",
        )

    return {"type": "constant", "value": float(const_val)}


def render_term_builder(
    title: str,
    prefix: str,
    term_idx: int,
    index_options: List[str],
    catalog_labels: List[str],
    label_to_item: Dict[str, Dict[str, Any]],
    old_term: Dict[str, Any] | None = None,
    default_num_factors: int = 2,
) -> Dict[str, Any]:
    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns([1, 1.4, 2.2])

    with c1:
        sign = st.selectbox(
            "Signo",
            options=["+", "-"],
            index=0 if old_term is None or old_term.get("sign", "+") == "+" else 1,
            key=f"{prefix}_sign_{term_idx}",
        )

    with c2:
        num_factors = st.number_input(
            "Número de factores",
            min_value=1,
            max_value=4,
            value=default_num_factors if old_term is None else max(1, len(old_term.get("factors", []))),
            step=1,
            key=f"{prefix}_num_factors_{term_idx}",
        )

    with c3:
        sum_over = st.multiselect(
            "Sumar sobre",
            options=index_options,
            default=[] if old_term is None else old_term.get("sum_over", []),
            key=f"{prefix}_sum_over_{term_idx}",
        )

    factors = []
    old_factors = [] if old_term is None else old_term.get("factors", [])
    for f in range(int(num_factors)):
        with st.container(border=True):
            factor = render_factor_builder(
                prefix=prefix,
                term_idx=term_idx,
                factor_idx=f,
                catalog_labels=catalog_labels,
                label_to_item=label_to_item,
                old_factor=old_factors[f] if f < len(old_factors) else None,
            )
            factors.append(factor)

    term = {"sign": sign, "factors": factors, "sum_over": sum_over}
    st.latex(build_term_latex(term))

    free = term_free_indices(term)
    if free:
        st.warning(f"Índices libres: {', '.join(free)}")
    else:
        st.success("Sin índices libres.")

    return term


def render_parameter_editor(
    pname: str,
    p_indices: List[str],
    existing_values: Dict[str, float],
    index_specs: Dict[str, Dict[str, Any]],
    mode: str,
    ui_key: str,
) -> Dict[str, float]:
    num_elems = total_elements_for_dims(p_indices, index_specs)

    if not p_indices:
        if mode == "Manual":
            value = st.number_input(
                f"Valor de {pname}",
                value=scalar_from_values_dict(existing_values, 0.0),
                key=f"{ui_key}_scalar_manual",
            )
            return values_dict_from_scalar(value)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            low = st.number_input("Mínimo", value=0.0, key=f"{ui_key}_low")
        with c2:
            high = st.number_input("Máximo", value=10.0, key=f"{ui_key}_high")
        with c3:
            integer_mode = st.checkbox("Entero", value=False, key=f"{ui_key}_integer")
        with c4:
            seed = st.number_input("Semilla", value=123, step=1, key=f"{ui_key}_seed")

        if low > high:
            st.error("El mínimo no puede ser mayor que el máximo.")
            return existing_values or values_dict_from_scalar(0.0)

        if st.button(f"Generar {pname}", key=f"{ui_key}_generate"):
            st.session_state[f"{ui_key}_generated"] = random_scalar(low, high, integer_mode, int(seed))

        generated = st.session_state.get(
            f"{ui_key}_generated",
            existing_values if existing_values else random_scalar(low, high, integer_mode, int(seed)),
        )
        st.write(f"Valor generado: **{scalar_from_values_dict(generated):.4f}**")
        return generated

    if len(p_indices) == 1:
        labels = index_specs[p_indices[0]]["elements"]
        if mode == "Manual":
            df0 = dataframe_1d_from_values_dict(labels, existing_values)
            edited = st.data_editor(
                df0,
                use_container_width=True,
                hide_index=True,
                disabled=["label"],
                num_rows="fixed",
                key=f"{ui_key}_manual_1d",
            )
            return values_dict_from_dataframe_1d(edited, labels)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            low = st.number_input("Mínimo", value=0.0, key=f"{ui_key}_low")
        with c2:
            high = st.number_input("Máximo", value=10.0, key=f"{ui_key}_high")
        with c3:
            integer_mode = st.checkbox("Entero", value=False, key=f"{ui_key}_integer")
        with c4:
            seed = st.number_input("Semilla", value=123, step=1, key=f"{ui_key}_seed")

        if low > high:
            st.error("El mínimo no puede ser mayor que el máximo.")
            return existing_values or {}

        combos = cartesian_labels(p_indices, index_specs)
        if st.button(f"Generar valores de {pname}", key=f"{ui_key}_generate"):
            st.session_state[f"{ui_key}_generated"] = random_values_dict(combos, low, high, integer_mode, int(seed))

        generated = st.session_state.get(
            f"{ui_key}_generated",
            existing_values if existing_values else random_values_dict(combos, low, high, integer_mode, int(seed)),
        )
        st.dataframe(dataframe_1d_from_values_dict(labels, generated), use_container_width=True, hide_index=True)
        return generated

    combos = cartesian_labels(p_indices, index_specs)
    if mode == "Manual":
        df0 = dataframe_nd_from_values_dict(p_indices, combos, existing_values)
        edited = st.data_editor(
            df0,
            use_container_width=True,
            hide_index=True,
            disabled=p_indices,
            num_rows="fixed",
            key=f"{ui_key}_manual_nd",
        )
        return values_dict_from_dataframe_nd(edited, p_indices)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        low = st.number_input("Mínimo", value=0.0, key=f"{ui_key}_low")
    with c2:
        high = st.number_input("Máximo", value=10.0, key=f"{ui_key}_high")
    with c3:
        integer_mode = st.checkbox("Entero", value=False, key=f"{ui_key}_integer")
    with c4:
        seed = st.number_input("Semilla", value=123, step=1, key=f"{ui_key}_seed")

    if low > high:
        st.error("El mínimo no puede ser mayor que el máximo.")
        return existing_values or {}

    if st.button(f"Generar valores de {pname}", key=f"{ui_key}_generate"):
        st.session_state[f"{ui_key}_generated"] = random_values_dict(combos, low, high, integer_mode, int(seed))

    generated = st.session_state.get(
        f"{ui_key}_generated",
        existing_values if existing_values else random_values_dict(combos, low, high, integer_mode, int(seed)),
    )
    st.dataframe(dataframe_nd_from_values_dict(p_indices, combos, generated), use_container_width=True, hide_index=True)
    return generated


# ============================================================
# ENCABEZADO Y SIDEBAR
# ============================================================

top_header()
spec = st.session_state["model_spec"]

st.sidebar.markdown("## Navegación")
section = st.sidebar.radio(
    "Ir a:",
    ["Ingreso de información", "Definición del modelo", "Resolver modelo"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Estado actual")
st.sidebar.markdown(f'<span class="pill">Índices: {len(spec["indices"])}</span>', unsafe_allow_html=True)
st.sidebar.markdown(f'<span class="pill">Parámetros: {len(spec["parameters"])}</span>', unsafe_allow_html=True)
st.sidebar.markdown(f'<span class="pill">Variables: {len(spec["variables"])}</span>', unsafe_allow_html=True)


# ============================================================
# 1. INGRESO DE INFORMACIÓN
# ============================================================

if section == "Ingreso de información":
    section_banner(
        "1. Ingreso de información",
        "Define los índices, parámetros y variables que conformarán la estructura base del modelo."
    )
    render_spec_summary(spec)

    with st.container(border=True):
        st.subheader("Índices")

        current_indices = spec["indices"]
        num_indices = st.number_input(
            "Número de índices",
            min_value=1,
            max_value=10,
            value=max(1, len(current_indices) if current_indices else 3),
            step=1,
        )

        index_rows = []
        used_names = set()
        index_errors = []

        existing_names = list(current_indices.keys())

        for r in range(int(num_indices)):
            col1, col2 = st.columns([2, 2])
            default_name = existing_names[r] if r < len(existing_names) else f"idx_{r+1}"

            with col1:
                idx_name = st.text_input(
                    f"Nombre del índice {r+1}",
                    value=default_name,
                    key=f"index_name_{r}",
                ).strip()

            with col2:
                existing_size = current_indices[default_name]["size"] if default_name in current_indices else 3
                idx_size = st.number_input(
                    f"Tamaño de {idx_name or f'índice {r+1}'}",
                    min_value=1,
                    max_value=1000,
                    value=int(existing_size),
                    step=1,
                    key=f"index_size_{r}",
                )

            if not is_valid_symbol(idx_name):
                index_errors.append(f"El nombre `{idx_name}` no es válido.")
            elif idx_name in used_names:
                index_errors.append(f"El índice `{idx_name}` está repetido.")
            else:
                used_names.add(idx_name)
                index_rows.append(
                    {
                        "name": sanitize_symbol(idx_name),
                        "size": int(idx_size),
                        "elements": build_index_elements(int(idx_size), sanitize_symbol(idx_name)),
                    }
                )

        for err in index_errors:
            st.error(err)

        valid_indices = len(index_errors) == 0
        index_specs = {}

        if valid_indices:
            for row in index_rows:
                index_specs[row["name"]] = {"size": row["size"], "elements": row["elements"]}
            st.session_state["model_spec"]["indices"] = index_specs
            reset_parameters_if_invalid(index_specs)
            reset_variables_if_invalid(index_specs)
            spec = st.session_state["model_spec"]

        if valid_indices and index_specs:
            preview = pd.DataFrame(
                [
                    {
                        "Índice": idx_name,
                        "Tamaño": idx_spec["size"],
                        "Elementos": ", ".join(idx_spec["elements"]),
                    }
                    for idx_name, idx_spec in index_specs.items()
                ]
            )
            st.dataframe(preview, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Parámetros")

        if not spec["indices"]:
            st.info("Primero define índices válidos.")
        else:
            current_params = spec["parameters"]
            num_params = st.number_input(
                "Número de parámetros",
                min_value=0,
                max_value=30,
                value=max(1, len(current_params)) if current_params else 1,
                step=1,
            )

            new_params = {}
            index_name_options = list(spec["indices"].keys())
            existing_param_names = list(current_params.keys())

            for p in range(int(num_params)):
                with st.expander(f"Parámetro {p+1}", expanded=(p == 0)):
                    old_name = existing_param_names[p] if p < len(existing_param_names) else f"param_{p+1}"

                    c1, c2 = st.columns([2, 3])
                    with c1:
                        pname = st.text_input(
                            f"Nombre del parámetro {p+1}",
                            value=old_name,
                            key=f"param_name_{p}",
                        ).strip()

                    with c2:
                        default_indices = current_params.get(old_name, {}).get("indices", [])
                        p_indices = st.multiselect(
                            f"Índices de {pname or f'parámetro {p+1}'}",
                            options=index_name_options,
                            default=default_indices,
                            key=f"param_indices_{p}",
                        )

                    if not is_valid_symbol(pname):
                        st.error(f"El nombre `{pname}` no es válido.")
                        continue
                    if pname in new_params:
                        st.error(f"El parámetro `{pname}` está repetido.")
                        continue

                    num_elems = total_elements_for_dims(p_indices, spec["indices"])
                    st.caption(f"Firma: {pretty_signature(pname, p_indices)} | Elementos: {num_elems}")

                    mode_options = ["Manual", "Aleatorio"] if num_elems <= 12 else ["Aleatorio"]
                    default_mode = current_params.get(old_name, {}).get("mode", mode_options[0])
                    if default_mode not in mode_options:
                        default_mode = mode_options[0]

                    mode = st.selectbox(
                        "Modo de carga",
                        options=mode_options,
                        index=mode_options.index(default_mode),
                        key=f"param_mode_{p}",
                    )

                    existing_values = current_params.get(old_name, {}).get("values", {})
                    values = render_parameter_editor(
                        pname=pname,
                        p_indices=p_indices,
                        existing_values=existing_values,
                        index_specs=spec["indices"],
                        mode=mode,
                        ui_key=f"param_ui_{p}_{pname}",
                    )

                    new_params[pname] = {
                        "indices": p_indices,
                        "mode": mode,
                        "values": values,
                    }

            st.session_state["model_spec"]["parameters"] = new_params
            spec = st.session_state["model_spec"]

            if new_params:
                summary = pd.DataFrame(
                    [
                        {
                            "Parámetro": pretty_signature(name, pspec["indices"]),
                            "Modo": pspec["mode"],
                            "Elementos": total_elements_for_dims(pspec["indices"], spec["indices"]),
                        }
                        for name, pspec in new_params.items()
                    ]
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Variables")

        if not spec["indices"]:
            st.info("Primero define índices válidos.")
        else:
            current_vars = spec["variables"]
            num_vars = st.number_input(
                "Número de variables",
                min_value=0,
                max_value=30,
                value=max(1, len(current_vars)) if current_vars else 1,
                step=1,
            )

            new_vars = {}
            index_name_options = list(spec["indices"].keys())
            existing_var_names = list(current_vars.keys())

            for v in range(int(num_vars)):
                with st.expander(f"Variable {v+1}", expanded=(v == 0)):
                    old_name = existing_var_names[v] if v < len(existing_var_names) else f"x_{v+1}"

                    c1, c2, c3 = st.columns([2, 3, 2])
                    with c1:
                        vname = st.text_input(
                            f"Nombre de la variable {v+1}",
                            value=old_name,
                            key=f"var_name_{v}",
                        ).strip()
                    with c2:
                        default_indices = current_vars.get(old_name, {}).get("indices", [])
                        v_indices = st.multiselect(
                            f"Índices de {vname or f'variable {v+1}'}",
                            options=index_name_options,
                            default=default_indices,
                            key=f"var_indices_{v}",
                        )
                    with c3:
                        default_domain = current_vars.get(old_name, {}).get("domain", "NonNegativeReals")
                        if default_domain not in DOMAIN_OPTIONS:
                            default_domain = "NonNegativeReals"
                        v_domain = st.selectbox(
                            "Dominio",
                            options=DOMAIN_OPTIONS,
                            index=DOMAIN_OPTIONS.index(default_domain),
                            key=f"var_domain_{v}",
                        )

                    if not is_valid_symbol(vname):
                        st.error(f"El nombre `{vname}` no es válido.")
                        continue
                    if vname in new_vars:
                        st.error(f"La variable `{vname}` está repetida.")
                        continue

                    st.caption(
                        f"Firma: {pretty_signature(vname, v_indices)} | "
                        f"Componentes: {total_elements_for_dims(v_indices, spec['indices'])}"
                    )

                    new_vars[vname] = {"indices": v_indices, "domain": v_domain}

            st.session_state["model_spec"]["variables"] = new_vars
            spec = st.session_state["model_spec"]

            if new_vars:
                summary = pd.DataFrame(
                    [
                        {
                            "Variable": pretty_signature(name, vspec["indices"]),
                            "Dominio": infer_domain_text(vspec["domain"]),
                            "Componentes": total_elements_for_dims(vspec["indices"], spec["indices"]),
                        }
                        for name, vspec in new_vars.items()
                    ]
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)


# ============================================================
# 2. DEFINICIÓN DEL MODELO
# ============================================================

elif section == "Definición del modelo":
    section_banner(
        "2. Definición del modelo",
        "Construye la función objetivo y las familias de restricciones."
    )
    spec = st.session_state["model_spec"]

    if not spec["indices"]:
        st.warning("Primero debes definir al menos un índice.")
        st.stop()
    if not spec["variables"]:
        st.warning("Primero debes definir al menos una variable.")
        st.stop()

    catalog = object_catalog(spec)
    catalog_labels = [obj["label"] for obj in catalog]
    label_to_item = {obj["label"]: obj for obj in catalog}
    index_options = list(spec["indices"].keys())

    with st.container(border=True):
        st.subheader("Tipo de problema")
        current_obj = spec.get("objective")
        default_sense = "minimize" if current_obj is None else current_obj.get("sense", "minimize")
        sense = st.selectbox(
            "Sentido de la función objetivo",
            options=["minimize", "maximize"],
            index=0 if default_sense == "minimize" else 1,
            format_func=lambda x: "Minimizar" if x == "minimize" else "Maximizar",
        )

    with st.container(border=True):
        st.subheader("Función objetivo")
        old_obj_terms = [] if current_obj is None else current_obj.get("terms", [])

        num_obj_terms = st.number_input(
            "Número de términos en la función objetivo",
            min_value=1,
            max_value=20,
            value=max(1, len(old_obj_terms)) if old_obj_terms else 1,
            step=1,
        )

        objective_terms = []
        for t in range(int(num_obj_terms)):
            with st.expander(f"Término FO {t+1}", expanded=(t == 0)):
                term = render_term_builder(
                    title=f"Término FO {t+1}",
                    prefix="obj",
                    term_idx=t,
                    index_options=index_options,
                    catalog_labels=catalog_labels,
                    label_to_item=label_to_item,
                    old_term=old_obj_terms[t] if t < len(old_obj_terms) else None,
                    default_num_factors=2,
                )
                objective_terms.append(term)

        objective_record = {"sense": sense, "terms": objective_terms}
        obj_errors = validate_objective_terms(objective_terms)

        st.markdown("### Vista previa")
        sense_symbol = r"\min" if sense == "minimize" else r"\max"
        st.latex(rf"{sense_symbol}\ Z = {build_expression_latex(objective_terms)}")

        if obj_errors:
            for err in obj_errors:
                st.error(err)
        else:
            st.success("La función objetivo está estructuralmente consistente.")

        spec["objective"] = objective_record
        st.session_state["model_spec"] = spec

    with st.container(border=True):
        st.subheader("Familias de restricciones")
        old_constraints = spec.get("constraints", [])

        num_families = st.number_input(
            "Número de familias de restricciones",
            min_value=0,
            max_value=30,
            value=len(old_constraints) if old_constraints else 1,
            step=1,
        )

        new_constraint_families = []

        for r in range(int(num_families)):
            old_family = old_constraints[r] if r < len(old_constraints) else None
            fname_default = f"R{r+1}" if old_family is None else old_family.get("name", f"R{r+1}")

            with st.expander(f"Familia {r+1}: {fname_default}", expanded=(r == 0)):
                c1, c2, c3 = st.columns([2, 2.2, 1.2])
                with c1:
                    family_name = st.text_input(
                        "Nombre de la familia",
                        value=fname_default,
                        key=f"family_name_{r}",
                    )
                with c2:
                    forall = st.multiselect(
                        "Para todo",
                        options=index_options,
                        default=[] if old_family is None else old_family.get("forall", []),
                        key=f"family_forall_{r}",
                    )
                with c3:
                    sense_family = st.selectbox(
                        "Sentido",
                        options=["<=", ">=", "="],
                        index=0 if old_family is None else ["<=", ">=", "="].index(old_family.get("sense", "<=")),
                        key=f"family_sense_{r}",
                    )

                st.markdown("#### Lado izquierdo")
                old_lhs = [] if old_family is None else old_family.get("lhs_terms", [])
                lhs_num_terms = st.number_input(
                    "Número de términos LHS",
                    min_value=1,
                    max_value=20,
                    value=max(1, len(old_lhs)) if old_lhs else 1,
                    step=1,
                    key=f"lhs_num_terms_{r}",
                )
                lhs_terms = []
                for t in range(int(lhs_num_terms)):
                    term = render_term_builder(
                        title=f"LHS término {t+1}",
                        prefix=f"lhs_{r}",
                        term_idx=t,
                        index_options=index_options,
                        catalog_labels=catalog_labels,
                        label_to_item=label_to_item,
                        old_term=old_lhs[t] if t < len(old_lhs) else None,
                        default_num_factors=2,
                    )
                    lhs_terms.append(term)

                st.markdown("#### Lado derecho")
                old_rhs = [] if old_family is None else old_family.get("rhs_terms", [])
                rhs_num_terms = st.number_input(
                    "Número de términos RHS",
                    min_value=1,
                    max_value=20,
                    value=max(1, len(old_rhs)) if old_rhs else 1,
                    step=1,
                    key=f"rhs_num_terms_{r}",
                )
                rhs_terms = []
                for t in range(int(rhs_num_terms)):
                    term = render_term_builder(
                        title=f"RHS término {t+1}",
                        prefix=f"rhs_{r}",
                        term_idx=t,
                        index_options=index_options,
                        catalog_labels=catalog_labels,
                        label_to_item=label_to_item,
                        old_term=old_rhs[t] if t < len(old_rhs) else None,
                        default_num_factors=1,
                    )
                    rhs_terms.append(term)

                family_record = {
                    "name": family_name,
                    "forall": forall,
                    "sense": sense_family,
                    "lhs_terms": lhs_terms,
                    "rhs_terms": rhs_terms,
                }

                st.markdown("### Vista previa")
                st.latex(build_constraint_family_latex(family_record))

                family_errors = validate_constraint_family(family_record)
                if family_errors:
                    for err in family_errors:
                        st.error(err)
                else:
                    st.success("La familia de restricciones está estructuralmente consistente.")

                new_constraint_families.append(family_record)

        spec["constraints"] = new_constraint_families
        st.session_state["model_spec"] = spec

    with st.container(border=True):
        st.subheader("Resumen de la definición")
        if spec["objective"] is not None:
            sense_symbol_resume = r"\min" if spec["objective"]["sense"] == "minimize" else r"\max"
            st.latex(rf"{sense_symbol_resume}\ Z = {build_expression_latex(spec['objective']['terms'])}")

        if not spec["constraints"]:
            st.info("No hay familias de restricciones definidas.")
        else:
            for fam in spec["constraints"]:
                st.latex(build_constraint_family_latex(fam))


# ============================================================
# 3. RESOLVER MODELO
# ============================================================

elif section == "Resolver modelo":
    section_banner(
        "3. Resolver modelo",
        "Valida, construye y resuelve el modelo a partir de la estructura definida."
    )
    spec = st.session_state["model_spec"]

    with st.container(border=True):
        st.subheader("Validación previa")

        validation_errors = []

        if not spec["indices"]:
            validation_errors.append("No hay índices definidos.")
        if not spec["variables"]:
            validation_errors.append("No hay variables definidas.")
        if spec["objective"] is None:
            validation_errors.append("No hay función objetivo definida.")
        else:
            validation_errors.extend(validate_objective_terms(spec["objective"].get("terms", [])))

        for fam in spec.get("constraints", []):
            validation_errors.extend(validate_constraint_family(fam))

        validation_errors.extend(validate_linearity_of_model(spec))

        if validation_errors:
            for err in validation_errors:
                st.error(err)
            st.stop()
        else:
            st.success("La especificación es válida para construir e intentar resolver el modelo.")

    with st.container(border=True):
        st.subheader("Resumen del modelo")

        if spec["objective"] is not None:
            sense_symbol = r"\min" if spec["objective"]["sense"] == "minimize" else r"\max"
            st.latex(rf"{sense_symbol}\ Z = {build_expression_latex(spec['objective']['terms'])}")

        if not spec["constraints"]:
            st.info("No hay restricciones definidas.")
        else:
            for fam in spec["constraints"]:
                st.latex(build_constraint_family_latex(fam))

    with st.container(border=True):
        st.subheader("Resolución")

        solver_name = st.selectbox(
            "Selecciona el solver",
            options=SOLVER_OPTIONS,
            index=0,
        )

        solve_button = st.button("Resolver modelo", type="primary", use_container_width=False)

        if solve_button:
            try:
                model = build_pyomo_model_from_spec(spec)
                solver = solver_factory_from_name(solver_name)
                result = solver.solve(model)

                termination = str(result.solver.termination_condition)
                status = str(result.solver.status)
                objective_value = pyo.value(model.OBJ)

                st.session_state["model_spec"]["results"] = {
                    "solver_name": solver_name,
                    "termination_condition": termination,
                    "status": status,
                    "objective_value": objective_value,
                }
                st.session_state["solved_model_object"] = model
                st.success("Modelo resuelto correctamente.")

            except Exception as e:
                st.session_state["model_spec"]["results"] = None
                st.session_state["solved_model_object"] = None
                st.error(f"Error al construir o resolver el modelo: {e}")

    with st.container(border=True):
        st.subheader("Resultados")

        results = st.session_state["model_spec"].get("results")
        solved_model = st.session_state.get("solved_model_object")

        if results is None or solved_model is None:
            st.info("Aún no has resuelto el modelo.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Solver", results["solver_name"])
            with c2:
                st.metric("Status", results["status"])
            with c3:
                st.metric("Termination", results["termination_condition"])

            st.metric("Valor óptimo", f"{results['objective_value']:,.6f}")

    if st.session_state["model_spec"].get("results") is not None and st.session_state.get("solved_model_object") is not None:
        solved_model = st.session_state["solved_model_object"]

        with st.container(border=True):
            st.subheader("Solución por variable")

            variable_names = list(spec["variables"].keys())
            selected_var = st.selectbox("Selecciona una variable", options=variable_names)

            var_df = variable_solution_to_dataframe(
                solved_model,
                selected_var,
                spec["variables"][selected_var],
                spec["indices"],
            )

            st.dataframe(var_df, use_container_width=True, hide_index=True)

            st.download_button(
                f"Descargar solución de {selected_var}",
                data=var_df.to_csv(index=False).encode("utf-8"),
                file_name=f"solucion_{selected_var}.csv",
                mime="text/csv",
            )

        with st.container(border=True):
            st.subheader("Variables no nulas")

            nz_df = nonzero_variables_solution_flat(solved_model, spec)
            if nz_df.empty:
                st.info("No hay variables no nulas.")
            else:
                st.dataframe(nz_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Descargar variables no nulas",
                data=nz_df.to_csv(index=False).encode("utf-8"),
                file_name="solucion_variables_no_nulas.csv",
                mime="text/csv",
            )
