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
# ESTADO INICIAL
# ============================================================
def init_session_state() -> None:
    defaults = {
        "model_spec": {
            "indices": {}, "parameters": {}, "variables": {},
            "objective": None, "constraints": [], "results": None,
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
    if not isinstance(name, str) or not name.strip(): return False
    name = name.strip()
    if not (name[0].isalpha() or name[0] == "_"): return False
    return all(ch.isalnum() or ch == "_" for ch in name[1:])

def sanitize_symbol(name: str) -> str: return name.strip()
def build_index_elements(size: int, label_prefix: str) -> List[str]: return [f"{label_prefix}{t}" for t in range(1, size + 1)]

def total_elements_for_dims(index_names: List[str], index_specs: Dict[str, Dict[str, Any]]) -> int:
    if not index_names: return 1
    total = 1
    for idx_name in index_names: total *= int(index_specs[idx_name]["size"])
    return total

def cartesian_labels(index_names: List[str], index_specs: Dict[str, Dict[str, Any]]) -> List[Tuple[str, ...]]:
    if not index_names: return [tuple()]
    return list(itertools.product(*[index_specs[idx]["elements"] for idx in index_names]))

def pretty_signature(name: str, index_names: List[str]) -> str:
    return f"{name}[{', '.join(index_names)}]" if index_names else name

def values_dict_from_scalar(value: float) -> Dict[str, float]: return {"__scalar__": float(value)}
def scalar_from_values_dict(values: Dict[str, float], default: float = 0.0) -> float: return float(values.get("__scalar__", default))

def values_dict_from_dataframe_1d(df: pd.DataFrame, labels: List[str], value_col: str = "value") -> Dict[str, float]:
    return {str((lbl,)): float(df.loc[df["label"] == lbl, value_col].iloc[0]) for lbl in labels}

def dataframe_1d_from_values_dict(labels: List[str], values_dict: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"label": lbl, "value": float(values_dict.get(str((lbl,)), 0.0))} for lbl in labels])

def values_dict_from_dataframe_nd(df: pd.DataFrame, index_names: List[str], value_col: str = "value") -> Dict[str, float]:
    return {str(tuple(str(row[idx]) for idx in index_names)): float(row[value_col]) for _, row in df.iterrows()}

def dataframe_nd_from_values_dict(index_names: List[str], combos: List[Tuple[str, ...]], values_dict: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for combo in combos:
        row = {idx_name: combo[pos] for pos, idx_name in enumerate(index_names)}
        row["value"] = float(values_dict.get(str(combo), 0.0))
        rows.append(row)
    return pd.DataFrame(rows)

def random_values_dict(combos: List[Tuple[str, ...]], low: float, high: float, integer_mode: bool, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for combo in combos:
        out[str(combo)] = float(rng.integers(int(low), int(high) + 1) if integer_mode else rng.uniform(low, high))
    return out

def random_scalar(low: float, high: float, integer_mode: bool, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    return {"__scalar__": float(rng.integers(int(low), int(high) + 1) if integer_mode else rng.uniform(low, high))}

def infer_domain_text(domain_code: str) -> str:
    return {"Binary": "Binarias", "NonNegativeReals": "Reales ≥ 0", "NonNegativeIntegers": "Enteras ≥ 0"}.get(domain_code, domain_code)

def reset_objects_if_invalid(obj_key: str, valid_index_names: set) -> None:
    cleaned = {k: v for k, v in st.session_state["model_spec"][obj_key].items() if all(idx in valid_index_names for idx in v.get("indices", []))}
    st.session_state["model_spec"][obj_key] = cleaned

# ============================================================
# UTILIDADES DE DEFINICIÓN DEL MODELO & LATEX
# ============================================================
def object_catalog(model_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = [{"kind": "parameter", "name": k, "indices": v["indices"], "label": pretty_signature(k, v["indices"])} for k, v in model_spec["parameters"].items()]
    items += [{"kind": "variable", "name": k, "indices": v["indices"], "label": pretty_signature(k, v["indices"])} for k, v in model_spec["variables"].items()]
    return items

def term_free_indices(term: Dict[str, Any]) -> List[str]:
    used = list(dict.fromkeys(idx for fac in term["factors"] if fac["type"] == "object" for idx in fac["indices"]))
    return [idx for idx in used if idx not in term.get("sum_over", [])]

def factor_to_latex(factor: Dict[str, Any]) -> str:
    if factor["type"] == "object":
        return factor["name"] if not factor["indices"] else rf"{factor['name']}_{{{','.join(factor['indices'])}}}"
    return str(int(factor["value"])) if float(factor["value"]).is_integer() else f"{factor['value']:.2f}"

def build_term_latex(term: Dict[str, Any]) -> str:
    body = r" \cdot ".join(factor_to_latex(f) for f in term.get("factors", [])) or "0"
    if term.get("sum_over"):
        sums = " ".join([rf"\sum_{{{idx}}}" for idx in term["sum_over"]])
        body = f"{sums}\\left({body}\\right)"
    return f"- {body}" if term.get("sign", "+") == "-" else f"+ {body}"

def build_expression_latex(terms: List[Dict[str, Any]]) -> str:
    expr = " ".join(build_term_latex(t) for t in terms).strip()
    return expr[2:] if expr.startswith("+ ") else (expr or "0")

def build_constraint_family_latex(family: Dict[str, Any]) -> str:
    sense_map = {"<=": r"\leq", ">=": r"\geq", "=": "="}
    txt = f"{build_expression_latex(family.get('lhs_terms', []))} {sense_map[family.get('sense', '<=')]} {build_expression_latex(family.get('rhs_terms', []))}"
    return txt + (r"\quad \forall " + ", ".join(family.get("forall", [])) if family.get("forall") else "")

# ============================================================
# PYOMO BUILDER & VALIDATORS
# ============================================================
# (Las funciones pyomo internas se mantienen intactas por seguridad matemática)
def pyomo_domain_from_code(domain_code: str):
    return {"Binary": pyo.Binary, "NonNegativeReals": pyo.NonNegativeReals, "NonNegativeIntegers": pyo.NonNegativeIntegers}[domain_code]

def get_component_value(model, factor: Dict[str, Any], env: Dict[str, Any]):
    if factor["type"] == "constant": return float(factor["value"])
    comp = getattr(model, f"par_{factor['name']}" if factor["kind"] == "parameter" else f"var_{factor['name']}")
    if not factor["indices"]: return comp
    key = tuple(env[idx] for idx in factor["indices"])
    return comp[key[0]] if len(key) == 1 else comp[key]

def build_base_term_value(model, term: Dict[str, Any], env: Dict[str, Any]):
    val = 1 if term.get("factors") else 0.0
    for fac in term.get("factors", []): val *= get_component_value(model, fac, env)
    return -val if term.get("sign", "+") == "-" else val

def evaluate_term_with_sums(model, term: Dict[str, Any], env: Dict[str, Any]):
    sum_over = term.get("sum_over", [])
    def recurse(pos: int, local_env: Dict[str, Any]):
        if pos == len(sum_over): return build_base_term_value(model, term, local_env)
        idx_name = sum_over[pos]
        return sum(recurse(pos + 1, {**local_env, idx_name: v}) for v in getattr(model, f"set_{idx_name}"))
    return recurse(0, dict(env))

def build_expression_pyomo(model, terms: List[Dict[str, Any]], env: Dict[str, Any]):
    return sum(evaluate_term_with_sums(model, term, env) for term in terms) if terms else 0

def build_pyomo_model_from_spec(spec: Dict[str, Any]):
    model = pyo.ConcreteModel()
    for idx_name, idx_spec in spec["indices"].items():
        setattr(model, f"set_{idx_name}", pyo.Set(initialize=idx_spec["elements"], ordered=True))

    for pname, pspec in spec["parameters"].items():
        if not pspec["indices"]:
            setattr(model, f"par_{pname}", pyo.Param(initialize=float(pspec["values"]["__scalar__"]), mutable=False))
        else:
            pyomo_sets = [getattr(model, f"set_{idx}") for idx in pspec["indices"]]
            init_dict = {combo[0] if len(combo)==1 else combo: float(pspec["values"].get(str(combo), 0.0)) for combo in cartesian_labels(pspec["indices"], spec["indices"])}
            setattr(model, f"par_{pname}", pyo.Param(*pyomo_sets, initialize=init_dict, mutable=False))

    for vname, vspec in spec["variables"].items():
        domain = pyomo_domain_from_code(vspec["domain"])
        if not vspec["indices"]:
            setattr(model, f"var_{vname}", pyo.Var(domain=domain))
        else:
            setattr(model, f"var_{vname}", pyo.Var(*[getattr(model, f"set_{idx}") for idx in vspec["indices"]], domain=domain))

    model.OBJ = pyo.Objective(expr=build_expression_pyomo(model, spec["objective"]["terms"], {}), sense=pyo.minimize if spec["objective"]["sense"] == "minimize" else pyo.maximize)

    for i, fam in enumerate(spec.get("constraints", []), 1):
        if not fam.get("forall"):
            expr = build_expression_pyomo(model, fam["lhs_terms"], {}) <= build_expression_pyomo(model, fam["rhs_terms"], {}) if fam["sense"] == "<=" else (
                   build_expression_pyomo(model, fam["lhs_terms"], {}) >= build_expression_pyomo(model, fam["rhs_terms"], {}) if fam["sense"] == ">=" else 
                   build_expression_pyomo(model, fam["lhs_terms"], {}) == build_expression_pyomo(model, fam["rhs_terms"], {}))
            setattr(model, f"con_{fam.get('name', f'R{i}')}", pyo.Constraint(expr=expr))
        else:
            def make_rule(lhs, rhs, forall, sense):
                def _rule(m, *args):
                    env = dict(zip(forall, args))
                    e_lhs, e_rhs = build_expression_pyomo(m, lhs, env), build_expression_pyomo(m, rhs, env)
                    return e_lhs <= e_rhs if sense == "<=" else (e_lhs >= e_rhs if sense == ">=" else e_lhs == e_rhs)
                return _rule
            setattr(model, f"con_{fam.get('name', f'R{i}')}", pyo.Constraint(*[getattr(model, f"set_{idx}") for idx in fam["forall"]], rule=make_rule(fam["lhs_terms"], fam["rhs_terms"], fam["forall"], fam["sense"])))
    return model

def variable_solution_to_dataframe(model, vname: str, vspec: Dict[str, Any], index_specs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    comp = getattr(model, f"var_{vname}")
    if not vspec["indices"]: return pd.DataFrame({"variable": [vname], "value": [pyo.value(comp)]})
    rows = [{**{idx: combo[pos] for pos, idx in enumerate(vspec["indices"])}, "value": pyo.value(comp[combo[0] if len(combo)==1 else combo])} for combo in cartesian_labels(vspec["indices"], index_specs)]
    return pd.DataFrame(rows)

# ============================================================
# INTERFAZ DE USUARIO (UI)
# ============================================================
st.title("📐 Constructor de Modelos Algebraicos")
st.caption("Versión Optimizada: Definición, estructuración y resolución rápida de modelos.")

st.sidebar.title("Navegación")
section = st.sidebar.radio("Ir a:", ["📝 Ingreso de Datos", "⚙️ Definición del Modelo", "🚀 Resolución y Salidas"])

spec = st.session_state["model_spec"]
st.sidebar.divider()
st.sidebar.markdown(f"**Estado del Modelo**\n- 🔢 Índices: `{len(spec['indices'])}`\n- 📊 Parámetros: `{len(spec['parameters'])}`\n- ✖️ Variables: `{len(spec['variables'])}`")

# ============================================================
# SECCIÓN 1: INGRESO DE DATOS (Usando Tabs para mejor UX)
# ============================================================
if section == "📝 Ingreso de Datos":
    t_ind, t_par, t_var = st.tabs(["🔢 Índices", "📊 Parámetros", "✖️ Variables"])
    
    # --- ÍNDICES ---
    with t_ind:
        st.subheader("Configuración de Índices")
        num_indices = st.number_input("Cantidad de índices", min_value=1, max_value=10, value=max(1, len(spec["indices"]) or 3), key="num_indices")
        index_rows, index_errors, used_names = [], [], set()

        for r in range(int(num_indices)):
            c1, c2 = st.columns(2)
            exist_names = list(spec["indices"].keys())
            def_name = exist_names[r] if r < len(exist_names) else f"i{r+1}"
            def_size = spec["indices"][exist_names[r]]["size"] if r < len(exist_names) else 3
            
            with c1: idx_name = st.text_input(f"Nombre del Índice {r+1}", value=def_name, key=f"idx_name_{r}").strip()
            with c2: idx_size = st.number_input(f"Tamaño ({idx_name})", min_value=1, value=int(def_size), key=f"idx_size_{r}")

            if not is_valid_symbol(idx_name): index_errors.append(f"`{idx_name}` no es un símbolo válido.")
            elif idx_name in used_names: index_errors.append(f"Índice repetido: `{idx_name}`.")
            else:
                used_names.add(idx_name)
                index_rows.append({"name": idx_name, "size": idx_size, "elements": build_index_elements(idx_size, idx_name)})

        for err in index_errors: st.error(err)
        if not index_errors:
            spec["indices"] = {r["name"]: {"size": r["size"], "elements": r["elements"]} for r in index_rows}
            reset_objects_if_invalid("parameters", used_names)
            reset_objects_if_invalid("variables", used_names)
            if spec["indices"]:
                with st.expander("👀 Ver Índices Registrados"):
                    st.dataframe(pd.DataFrame([{"Índice": k, "Tamaño": v["size"], "Elementos": ", ".join(v["elements"])} for k, v in spec["indices"].items()]), use_container_width=True)

    # --- PARÁMETROS ---
    with t_par:
        st.subheader("Configuración de Parámetros")
        if not spec["indices"]: st.info("Define al menos un índice primero.")
        else:
            num_params = st.number_input("Cantidad de parámetros", min_value=0, value=max(1, len(spec["parameters"])), key="num_params")
            new_params = {}
            for p in range(int(num_params)):
                with st.container():
                    c1, c2 = st.columns([1, 2])
                    old_name = list(spec["parameters"].keys())[p] if p < len(spec["parameters"]) else f"p{p+1}"
                    with c1: pname = st.text_input(f"Nombre {p+1}", value=old_name, key=f"pname_{p}").strip()
                    with c2: p_idxs = st.multiselect(f"Índices ({pname})", options=list(spec["indices"].keys()), default=spec["parameters"].get(old_name, {}).get("indices", []), key=f"pidx_{p}")
                    
                    if is_valid_symbol(pname) and pname not in new_params:
                        num_elems = total_elements_for_dims(p_idxs, spec["indices"])
                        mode_opts = ["Manual", "Aleatorio"] if num_elems <= 12 else ["Aleatorio"]
                        mode = st.radio(f"Modo ({pname})", options=mode_opts, index=0, horizontal=True, key=f"pmode_{p}")
                        
                        existing_vals = spec["parameters"].get(old_name, {}).get("values", {})
                        if mode == "Aleatorio":
                            c_a, c_b, c_c, c_d = st.columns(4)
                            low = c_a.number_input("Min", value=0.0, key=f"plow_{p}")
                            high = c_b.number_input("Max", value=10.0, key=f"phigh_{p}")
                            int_mode = c_c.checkbox("Entero", key=f"pint_{p}")
                            seed = c_d.number_input("Semilla", value=123, key=f"pseed_{p}")
                            if st.button(f"🎲 Generar Valores", key=f"pgen_{p}"):
                                existing_vals = random_scalar(low, high, int_mode, seed) if not p_idxs else random_values_dict(cartesian_labels(p_idxs, spec["indices"]), low, high, int_mode, seed)
                        
                        new_params[pname] = {"indices": p_idxs, "mode": mode, "values": existing_vals}
            spec["parameters"] = new_params

    # --- VARIABLES ---
    with t_var:
        st.subheader("Configuración de Variables")
        if not spec["indices"]: st.info("Define al menos un índice primero.")
        else:
            num_vars = st.number_input("Cantidad de variables", min_value=0, value=max(1, len(spec["variables"])), key="num_vars")
            new_vars = {}
            for v in range(int(num_vars)):
                c1, c2, c3 = st.columns([1, 1.5, 1.5])
                old_name = list(spec["variables"].keys())[v] if v < len(spec["variables"]) else f"x{v+1}"
                with c1: vname = st.text_input(f"Nombre {v+1}", value=old_name, key=f"vname_{v}").strip()
                with c2: v_idxs = st.multiselect(f"Índices ({vname})", options=list(spec["indices"].keys()), default=spec["variables"].get(old_name, {}).get("indices", []), key=f"vidx_{v}")
                with c3: v_dom = st.selectbox(f"Dominio ({vname})", ["NonNegativeReals", "NonNegativeIntegers", "Binary"], index=["NonNegativeReals", "NonNegativeIntegers", "Binary"].index(spec["variables"].get(old_name, {}).get("domain", "NonNegativeReals")), key=f"vdom_{v}")
                if is_valid_symbol(vname): new_vars[vname] = {"indices": v_idxs, "domain": v_dom}
            spec["variables"] = new_vars

# ============================================================
# SECCIÓN 2: DEFINICIÓN DEL MODELO
# ============================================================
elif section == "⚙️ Definición del Modelo":
    st.header("Construcción del Modelo Matemático")
    if not spec["indices"] or not spec["variables"]: st.warning("⚠️ Faltan definir índices o variables en la sección anterior.")
    else:
        t_obj, t_res, t_sum = st.tabs(["🎯 Función Objetivo", "🔗 Restricciones", "📋 Resumen LaTeX"])
        obj_catalog_list = object_catalog(spec)
        labels = [o["label"] for o in obj_catalog_list]
        lbl_to_item = {o["label"]: o for o in obj_catalog_list}

        # --- FUNCIÓN OBJETIVO ---
        with t_obj:
            sense = st.radio("Objetivo:", ["minimize", "maximize"], index=0 if spec.get("objective", {}).get("sense", "minimize") == "minimize" else 1, horizontal=True)
            num_obj = st.number_input("Términos en la F.O.", min_value=1, value=max(1, len(spec.get("objective", {}).get("terms", []))), key="num_obj")
            terms = []
            for t in range(int(num_obj)):
                with st.expander(f"Término {t+1}", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    old_t = spec.get("objective", {}).get("terms", [])[t] if t < len(spec.get("objective", {}).get("terms", [])) else {}
                    with c1: sign = st.selectbox("Signo", ["+", "-"], index=0 if old_t.get("sign", "+")=="+" else 1, key=f"osign_{t}")
                    with c2: nf = st.number_input("Factores", 1, 4, max(1, len(old_t.get("factors", []))), key=f"ofac_{t}")
                    with c3: so = st.multiselect("Sumatoria sobre:", list(spec["indices"].keys()), default=old_t.get("sum_over", []), key=f"osum_{t}")
                    
                    factors = []
                    for f in range(int(nf)):
                        fc1, fc2 = st.columns([1, 2])
                        old_f = old_t.get("factors", [])[f] if f < len(old_t.get("factors", [])) else {}
                        ftype = fc1.selectbox(f"Tipo Factor {f+1}", ["object", "constant"], index=0 if old_f.get("type", "object")=="object" else 1, key=f"oty_{t}_{f}")
                        if ftype == "object" and labels:
                            obj = fc2.selectbox("Variable/Param", labels, key=f"ovar_{t}_{f}")
                            factors.append({"type": "object", **lbl_to_item[obj]})
                        else:
                            val = fc2.number_input("Valor", value=float(old_f.get("value", 1.0)), key=f"oval_{t}_{f}")
                            factors.append({"type": "constant", "value": val})
                    terms.append({"sign": sign, "factors": factors, "sum_over": so})
            spec["objective"] = {"sense": sense, "terms": terms}

        # --- RESTRICCIONES ---
        with t_res:
            num_cons = st.number_input("Familias de restricciones", min_value=0, value=len(spec.get("constraints", [])), key="num_cons")
            constraints = []
            for r in range(int(num_cons)):
                with st.expander(f"Restricción R{r+1}", expanded=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    old_c = spec.get("constraints", [])[r] if r < len(spec.get("constraints", [])) else {}
                    with c1: name = st.text_input("Nombre", old_c.get("name", f"R{r+1}"), key=f"rname_{r}")
                    with c2: forall = st.multiselect("Para todo (∀)", list(spec["indices"].keys()), default=old_c.get("forall", []), key=f"rfor_{r}")
                    with c3: c_sense = st.selectbox("Operador", ["<=", ">=", "="], index=["<=", ">=", "="].index(old_c.get("sense", "<=")), key=f"rop_{r}")
                    
                    # LHS / RHS Simplificado (Solo 1 término por lado en la UI para mantenerlo 'ligero' visualmente, extensible)
                    st.markdown("**Lado Izquierdo (LHS) y Derecho (RHS)**")
                    lhs_terms, rhs_terms = [], []
                    
                    c_lhs, c_rhs = st.columns(2)
                    with c_lhs:
                        l_obj = st.selectbox("Término LHS", labels, key=f"rlhs_{r}") if labels else None
                        if l_obj: lhs_terms.append({"sign": "+", "factors": [{"type": "object", **lbl_to_item[l_obj]}], "sum_over": []})
                    with c_rhs:
                        r_val = st.number_input("Valor Constante RHS", value=0.0, key=f"rrhs_{r}")
                        rhs_terms.append({"sign": "+", "factors": [{"type": "constant", "value": r_val}], "sum_over": []})

                    constraints.append({"name": name, "forall": forall, "sense": c_sense, "lhs_terms": lhs_terms, "rhs_terms": rhs_terms})
            spec["constraints"] = constraints

        # --- RESUMEN ---
        with t_sum:
            st.markdown("### Modelo Estructurado")
            sense_sym = r"\min" if spec["objective"]["sense"] == "minimize" else r"\max"
            st.latex(rf"{sense_sym}\ Z = {build_expression_latex(spec['objective']['terms'])}")
            st.markdown("**Sujeto a:**")
            for fam in spec["constraints"]: st.latex(build_constraint_family_latex(fam))

# ============================================================
# SECCIÓN 3: SALIDAS Y RESOLUCIÓN
# ============================================================
elif section == "🚀 Resolución y Salidas":
    st.header("Ejecución y Resultados")

    c1, c2 = st.columns([1, 2])
    # appsi_highs es el solver principal por defecto asegurado por el index=0
    solver_name = c1.selectbox("🔧 Selecciona el Solver", options=["appsi_highs", "glpk", "cbc"], index=0)
    
    if c1.button("🚀 Resolver Modelo", type="primary"):
        with st.spinner(f"Resolviendo con {solver_name}..."):
            try:
                model = build_pyomo_model_from_spec(spec)
                solver = pyo.SolverFactory(solver_name)
                result = solver.solve(model)
                
                st.session_state["model_spec"]["results"] = {
                    "solver_name": solver_name,
                    "termination_condition": str(result.solver.termination_condition),
                    "status": str(result.solver.status),
                    "objective_value": pyo.value(model.OBJ),
                }
                st.session_state["solved_model_object"] = model
                st.success("✅ Modelo resuelto correctamente.")
            except Exception as e:
                st.error(f"❌ Error al resolver: {e}")

    results = spec.get("results")
    if results:
        st.divider()
        st.subheader("📊 Resultados de la Optimización")
        
        # Estética de las métricas (similar a tu imagen)
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Solver", results["solver_name"])
        rc2.metric("Status", results["status"])
        rc3.metric("Termination", results["termination_condition"])
        # Formato de números grandes con coma
        rc4.metric("Valor óptimo", f"{results['objective_value']:,.4f}")

        model_solved = st.session_state.get("solved_model_object")
        if model_solved:
            st.markdown("### 🔍 Análisis de Variables")
            t_all, t_nz = st.tabs(["Seleccionar Variable", "Variables No Nulas"])
            
            with t_all:
                sel_v = st.selectbox("Inspeccionar:", list(spec["variables"].keys()))
                df_v = variable_solution_to_dataframe(model_solved, sel_v, spec["variables"][sel_v], spec["indices"])
                st.dataframe(df_v, use_container_width=True, hide_index=True)
                st.download_button("📥 Descargar CSV", df_v.to_csv(index=False).encode("utf-8"), f"{sel_v}.csv", "text/csv")
            
            with t_nz:
                rows = []
                for vname, vspec in spec["variables"].items():
                    df = variable_solution_to_dataframe(model_solved, vname, vspec, spec["indices"])
                    df.insert(0, "Variable", vname)
                    rows.append(df)
                if rows:
                    nz_df = pd.concat(rows, ignore_index=True)
                    nz_df = nz_df[nz_df["value"].abs() > 1e-9].reset_index(drop=True)
                    st.dataframe(nz_df, use_container_width=True, hide_index=True)
