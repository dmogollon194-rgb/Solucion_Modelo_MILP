import streamlit as st
import pandas as pd
import numpy as np
import itertools
import json
from typing import Dict, List, Tuple, Any


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Constructor de modelos algebraicos",
    layout="wide",
)

st.title("Constructor de modelos algebraicos")
st.caption("Versión base: índices, parámetros y variables indexadas.")


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
# UTILIDADES
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


def parameter_preview_dataframe(
    param_name: str,
    param_spec: Dict[str, Any],
    index_specs: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    idxs = param_spec["indices"]
    values = param_spec["values"]

    if len(idxs) == 0:
        return pd.DataFrame({"parameter": [param_name], "value": [scalar_from_values_dict(values)]})

    combos = cartesian_labels(idxs, index_specs)
    return dataframe_nd_from_values_dict(idxs, combos, values)


def infer_domain_text(domain_code: str) -> str:
    mapper = {
        "Binary": "Binarias",
        "NonNegativeReals": "Reales ≥ 0",
        "NonNegativeIntegers": "Enteras ≥ 0",
    }
    return mapper.get(domain_code, domain_code)


def model_spec_as_jsonable(spec: Dict[str, Any]) -> Dict[str, Any]:
    return spec


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
# BARRA LATERAL / NAVEGACIÓN
# ============================================================

st.sidebar.title("Navegación")

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
st.sidebar.subheader("Estado actual")

spec = st.session_state["model_spec"]
st.sidebar.write(f"Índices: {len(spec['indices'])}")
st.sidebar.write(f"Parámetros: {len(spec['parameters'])}")
st.sidebar.write(f"Variables: {len(spec['variables'])}")


# ============================================================
# SECCIÓN 1: INGRESO DE INFORMACIÓN
# ============================================================

if section == "Ingreso de información":

    st.header("Ingreso de información")
    st.write("Aquí se definen índices, parámetros y variables del modelo.")

    # ========================================================
    # SUBSECCIÓN: ÍNDICES
    # ========================================================

    st.markdown("---")
    st.subheader("1. Índices")

    st.write("Define los índices del modelo. Ejemplos: `i`, `j`, `k`.")

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

    if len(index_errors) > 0:
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
    # SUBSECCIÓN: PARÁMETROS
    # ========================================================

    st.markdown("---")
    st.subheader("2. Parámetros")

    if not valid_indices or len(index_specs) == 0:
        st.info("Primero define índices válidos.")
    else:
        st.write(
            "Define parámetros escalares, vectores, matrices o tensores. "
            "Si el número total de elementos es mayor que 12, se habilita generación aleatoria."
        )

        current_params = st.session_state["model_spec"]["parameters"]
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

            if num_elems <= 12:
                mode_options = ["Manual", "Aleatorio"]
            else:
                mode_options = ["Aleatorio"]

            default_mode = current_params.get(old_name, {}).get("mode", mode_options[0])

            if default_mode not in mode_options:
                default_mode = mode_options[0]

            mode = st.selectbox(
                f"Modo de carga para {pname}",
                options=mode_options,
                index=mode_options.index(default_mode),
                key=f"param_mode_{p}"
            )

            existing_values = current_params.get(old_name, {}).get("values", {})
            param_record = {
                "indices": p_indices,
                "mode": mode,
                "values": {},
            }

            # ----------------------------------------------------
            # ESCALAR
            # ----------------------------------------------------
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
                        low = st.number_input(
                            f"Mínimo {pname}",
                            value=0.0,
                            key=f"param_scalar_low_{p}"
                        )
                    with c2:
                        high = st.number_input(
                            f"Máximo {pname}",
                            value=10.0,
                            key=f"param_scalar_high_{p}"
                        )
                    with c3:
                        integer_mode = st.checkbox(
                            f"Entero {pname}",
                            value=False,
                            key=f"param_scalar_integer_{p}"
                        )
                    with c4:
                        seed = st.number_input(
                            f"Semilla {pname}",
                            value=123,
                            step=1,
                            key=f"param_scalar_seed_{p}"
                        )

                    if low > high:
                        st.error("El mínimo no puede ser mayor que el máximo.")
                        continue

                    if st.button(f"Generar {pname}", key=f"btn_generate_scalar_{p}"):
                        st.session_state[f"generated_scalar_values_{pname}"] = random_scalar(
                            low=low,
                            high=high,
                            integer_mode=integer_mode,
                            seed=int(seed),
                        )

                    generated_values = st.session_state.get(
                        f"generated_scalar_values_{pname}",
                        existing_values if existing_values else random_scalar(low, high, integer_mode, int(seed))
                    )
                    param_record["values"] = generated_values
                    st.write(f"Valor generado: **{scalar_from_values_dict(generated_values):.4f}**")

            # ----------------------------------------------------
            # VECTOR 1D
            # ----------------------------------------------------
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
                        low = st.number_input(
                            f"Mínimo {pname}",
                            value=0.0,
                            key=f"param_random_low_{p}"
                        )
                    with c2:
                        high = st.number_input(
                            f"Máximo {pname}",
                            value=10.0,
                            key=f"param_random_high_{p}"
                        )
                    with c3:
                        integer_mode = st.checkbox(
                            f"Entero {pname}",
                            value=False,
                            key=f"param_random_integer_{p}"
                        )
                    with c4:
                        seed = st.number_input(
                            f"Semilla {pname}",
                            value=123,
                            step=1,
                            key=f"param_random_seed_{p}"
                        )

                    if low > high:
                        st.error("El mínimo no puede ser mayor que el máximo.")
                        continue

                    combos = cartesian_labels(p_indices, index_specs)

                    if st.button(f"Generar valores de {pname}", key=f"btn_generate_{p}"):
                        st.session_state[f"generated_values_{pname}"] = random_values_dict(
                            combos=combos,
                            low=low,
                            high=high,
                            integer_mode=integer_mode,
                            seed=int(seed),
                        )

                    generated_values = st.session_state.get(
                        f"generated_values_{pname}",
                        existing_values if existing_values else random_values_dict(
                            combos=combos,
                            low=low,
                            high=high,
                            integer_mode=integer_mode,
                            seed=int(seed),
                        )
                    )
                    param_record["values"] = generated_values
                    preview_df = dataframe_1d_from_values_dict(labels, generated_values)
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)

            # ----------------------------------------------------
            # MATRIZ / TENSOR
            # ----------------------------------------------------
            else:
                combos = cartesian_labels(p_indices, index_specs)

                if mode == "Manual":
                    df0 = dataframe_nd_from_values_dict(
                        p_indices,
                        combos,
                        existing_values
                    )

                    disabled_cols = list(p_indices)

                    edited_df = st.data_editor(
                        df0,
                        use_container_width=True,
                        num_rows="fixed",
                        hide_index=True,
                        disabled=disabled_cols,
                        key=f"param_manual_nd_{p}"
                    )

                    param_record["values"] = values_dict_from_dataframe_nd(
                        edited_df,
                        p_indices
                    )

                else:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        low = st.number_input(
                            f"Mínimo {pname}",
                            value=0.0,
                            key=f"param_random_low_{p}"
                        )
                    with c2:
                        high = st.number_input(
                            f"Máximo {pname}",
                            value=10.0,
                            key=f"param_random_high_{p}"
                        )
                    with c3:
                        integer_mode = st.checkbox(
                            f"Entero {pname}",
                            value=False,
                            key=f"param_random_integer_{p}"
                        )
                    with c4:
                        seed = st.number_input(
                            f"Semilla {pname}",
                            value=123,
                            step=1,
                            key=f"param_random_seed_{p}"
                        )

                    if low > high:
                        st.error("El mínimo no puede ser mayor que el máximo.")
                        continue

                    if st.button(f"Generar valores de {pname}", key=f"btn_generate_{p}"):
                        st.session_state[f"generated_values_{pname}"] = random_values_dict(
                            combos=combos,
                            low=low,
                            high=high,
                            integer_mode=integer_mode,
                            seed=int(seed),
                        )

                    generated_values = st.session_state.get(
                        f"generated_values_{pname}",
                        existing_values if existing_values else random_values_dict(
                            combos=combos,
                            low=low,
                            high=high,
                            integer_mode=integer_mode,
                            seed=int(seed),
                        )
                    )

                    param_record["values"] = generated_values
                    preview_df = dataframe_nd_from_values_dict(
                        p_indices,
                        combos,
                        generated_values
                    )
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
    # SUBSECCIÓN: VARIABLES
    # ========================================================

    st.markdown("---")
    st.subheader("3. Variables")

    if not valid_indices or len(index_specs) == 0:
        st.info("Primero define índices válidos.")
    else:
        current_vars = st.session_state["model_spec"]["variables"]
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

            domain_options = [
                "Binary",
                "NonNegativeReals",
                "NonNegativeIntegers"
            ]

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


# ============================================================
# SECCIÓN 2: DEFINICIÓN DEL MODELO
# ============================================================

elif section == "Definición del modelo":

    st.header("Definición del modelo")

    st.info(
        "Esta sección queda preparada para la siguiente fase. "
        "Aquí construiremos la función objetivo y las familias de restricciones."
    )

    spec = st.session_state["model_spec"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Índices definidos", len(spec["indices"]))
    with col2:
        st.metric("Parámetros definidos", len(spec["parameters"]))
    with col3:
        st.metric("Variables definidas", len(spec["variables"]))

    st.markdown("---")
    st.subheader("Vista preliminar de la estructura disponible")

    if len(spec["indices"]) == 0:
        st.warning("Aún no hay índices definidos.")
    else:
        st.write("**Índices disponibles:**")
        st.write(", ".join(spec["indices"].keys()))

    if len(spec["parameters"]) == 0:
        st.warning("Aún no hay parámetros definidos.")
    else:
        st.write("**Parámetros disponibles:**")
        for pname, pspec in spec["parameters"].items():
            st.write(f"- {pretty_param_signature(pname, pspec['indices'])}")

    if len(spec["variables"]) == 0:
        st.warning("Aún no hay variables definidas.")
    else:
        st.write("**Variables disponibles:**")
        for vname, vspec in spec["variables"].items():
            st.write(f"- {pretty_var_signature(vname, vspec['indices'])}")


# ============================================================
# SECCIÓN 3: SALIDAS DEL MODELO
# ============================================================

elif section == "Salidas del modelo":

    st.header("Salidas del modelo")

    st.info(
        "Esta sección queda preparada para la fase en la que el modelo ya tenga "
        "función objetivo, restricciones y solución."
    )

    st.markdown("---")
    st.subheader("Especificación actual del modelo")

    model_spec = st.session_state["model_spec"]

    tab1, tab2 = st.tabs(["Resumen algebraico", "JSON del modelo"])

    with tab1:
        st.subheader("Índices")
        if len(model_spec["indices"]) == 0:
            st.info("No hay índices definidos.")
        else:
            for idx_name, idx_spec in model_spec["indices"].items():
                st.write(
                    f"- `{idx_name}` con tamaño `{idx_spec['size']}` "
                    f"y elementos: {', '.join(idx_spec['elements'])}"
                )

        st.subheader("Parámetros")
        if len(model_spec["parameters"]) == 0:
            st.info("No hay parámetros definidos.")
        else:
            for pname, pspec in model_spec["parameters"].items():
                st.write(
                    f"- `{pretty_param_signature(pname, pspec['indices'])}` "
                    f"({pspec['mode']})"
                )

        st.subheader("Variables")
        if len(model_spec["variables"]) == 0:
            st.info("No hay variables definidas.")
        else:
            for vname, vspec in model_spec["variables"].items():
                st.write(
                    f"- `{pretty_var_signature(vname, vspec['indices'])}` "
                    f"en dominio `{infer_domain_text(vspec['domain'])}`"
                )

    with tab2:
        spec_json = json.dumps(
            model_spec_as_jsonable(model_spec),
            indent=2,
            ensure_ascii=False
        )
        st.code(spec_json, language="json")

        st.download_button(
            "Descargar especificación JSON",
            data=spec_json.encode("utf-8"),
            file_name="model_spec.json",
            mime="application/json"
        )

    st.markdown("---")
    st.subheader("Exportaciones rápidas")

    if len(model_spec["parameters"]) == 0:
        st.info("Define al menos un parámetro para habilitar exportaciones.")
    else:
        param_names = list(model_spec["parameters"].keys())
        selected_param_for_csv = st.selectbox(
            "Selecciona un parámetro para exportar sus valores a CSV",
            options=param_names,
            key="selected_param_for_csv"
        )

        pspec = model_spec["parameters"][selected_param_for_csv]
        export_df = parameter_preview_dataframe(
            selected_param_for_csv,
            pspec,
            model_spec["indices"]
        )

        st.dataframe(export_df, use_container_width=True, hide_index=True)

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar parámetro en CSV",
            data=csv_bytes,
            file_name=f"{selected_param_for_csv}.csv",
            mime="text/csv"
        )
