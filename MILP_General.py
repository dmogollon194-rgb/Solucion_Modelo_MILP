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

    spec = st.session_state["model_spec"]
    index_specs = spec["indices"]
    param_specs = spec["parameters"]
    var_specs = spec["variables"]

    if len(index_specs) == 0:
        st.warning("Primero debes definir al menos un índice.")
    elif len(var_specs) == 0:
        st.warning("Primero debes definir al menos una variable.")
    else:
        # ====================================================
        # 2.1 TIPO DE PROBLEMA
        # ====================================================
        st.subheader("1. Tipo de problema")

        current_obj = spec.get("objective", None)
        default_sense = "minimize"
        if current_obj is not None:
            default_sense = current_obj.get("sense", "minimize")

        sense_options = ["minimize", "maximize"]
        sense = st.selectbox(
            "Sentido de la función objetivo",
            options=sense_options,
            index=sense_options.index(default_sense),
            format_func=lambda x: "Minimizar" if x == "minimize" else "Maximizar",
            key="objective_sense"
        )

        # ====================================================
        # 2.2 FUNCIÓN OBJETIVO
        # ====================================================
        st.markdown("---")
        st.subheader("2. Función objetivo")

        st.write(
            "Construye la función objetivo como suma de términos. "
            "Cada término puede tener constantes, parámetros y variables, "
            "y puede incluir sumatorias."
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
                    value=2 if old_term is None else max(1, len(old_term.get("factors", []))),
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
                    default_label = obj_catalog_labels[0] if len(obj_catalog_labels) > 0 else None
                    if old_factor is not None and old_factor.get("type") == "object":
                        default_label = old_factor.get("label", default_label)

                    if len(obj_catalog_labels) == 0:
                        st.error("No hay parámetros ni variables disponibles para usar en la FO.")
                    else:
                        with fc2:
                            chosen_label = st.selectbox(
                                f"Objeto factor {f+1} término {t+1}",
                                options=obj_catalog_labels,
                                index=obj_catalog_labels.index(default_label) if default_label in obj_catalog_labels else 0,
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

            term_text = build_term_text(term_record)
            free_idxs = term_free_indices(term_record)

            st.write(f"**Vista previa del término:** `{term_text}`")
            if len(free_idxs) > 0:
                st.warning(f"Índices libres en este término: {', '.join(free_idxs)}")
            else:
                st.success("Este término no deja índices libres.")

        objective_record = {
            "sense": sense,
            "terms": objective_terms
        }

        obj_errors = validate_objective_terms(objective_terms)

        st.markdown("### Vista previa de la función objetivo")
        sense_symbol = "min" if sense == "minimize" else "max"
        obj_text = build_expression_text(objective_terms)
        st.code(f"{sense_symbol}  {obj_text}", language="text")

        if len(obj_errors) > 0:
            for err in obj_errors:
                st.error(err)
        else:
            st.success("La función objetivo está estructuralmente consistente.")

        spec["objective"] = objective_record

        # ====================================================
        # 2.3 FAMILIAS DE RESTRICCIONES
        # ====================================================
        st.markdown("---")
        st.subheader("3. Familias de restricciones")

        st.write(
            "Cada familia se construye con un lado izquierdo, un sentido, "
            "un lado derecho y un conjunto de índices 'para todo'."
        )

        old_constraints = spec.get("constraints", [])

        num_families = st.number_input(
            "Número de familias de restricciones",
            min_value=0,
            max_value=30,
            value=len(old_constraints) if len(old_constraints) > 0 else 1,
            step=1,
            key="num_constraint_families"
        )

        new_constraint_families = []

        expr_catalog = object_catalog(spec)
        expr_catalog_labels = [obj["label"] for obj in expr_catalog]
        expr_label_to_item = {obj["label"]: obj for obj in expr_catalog}

        for r in range(int(num_families)):
            st.markdown(f"## Familia de restricción {r+1}")

            old_family = old_constraints[r] if r < len(old_constraints) else None
            old_forall = [] if old_family is None else old_family.get("forall", [])
            old_sense = "<=" if old_family is None else old_family.get("sense", "<=")

            fc1, fc2 = st.columns([2, 2])

            with fc1:
                family_name = st.text_input(
                    f"Nombre de la familia {r+1}",
                    value=f"R{r+1}" if old_family is None else old_family.get("name", f"R{r+1}"),
                    key=f"constraint_family_name_{r}"
                )

            with fc2:
                forall = st.multiselect(
                    f"Para todo en familia {r+1}",
                    options=list(index_specs.keys()),
                    default=old_forall,
                    key=f"constraint_family_forall_{r}"
                )

            sense = st.selectbox(
                f"Sentido de la familia {r+1}",
                options=["<=", ">=", "="],
                index=["<=", ">=", "="].index(old_sense),
                key=f"constraint_family_sense_{r}"
            )

            # -----------------------
            # LADO IZQUIERDO
            # -----------------------
            st.markdown(f"### Lado izquierdo de {family_name}")

            old_lhs = [] if old_family is None else old_family.get("lhs_terms", [])

            lhs_num_terms = st.number_input(
                f"Número de términos LHS en {family_name}",
                min_value=1,
                max_value=20,
                value=max(1, len(old_lhs) if len(old_lhs) > 0 else 1),
                step=1,
                key=f"lhs_num_terms_{r}"
            )

            lhs_terms = []

            for t in range(int(lhs_num_terms)):
                st.markdown(f"**Término LHS {t+1}**")

                old_term = old_lhs[t] if t < len(old_lhs) else None

                c1, c2, c3 = st.columns([1, 2, 2])

                with c1:
                    sign = st.selectbox(
                        f"Signo LHS término {t+1} en {family_name}",
                        options=["+", "-"],
                        index=0 if old_term is None else (0 if old_term.get("sign", "+") == "+" else 1),
                        key=f"lhs_sign_{r}_{t}"
                    )

                with c2:
                    num_factors = st.number_input(
                        f"Número de factores LHS término {t+1} en {family_name}",
                        min_value=1,
                        max_value=4,
                        value=2 if old_term is None else max(1, len(old_term.get("factors", []))),
                        step=1,
                        key=f"lhs_num_factors_{r}_{t}"
                    )

                with c3:
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
                    old_factor = old_factors[f] if f < len(old_factors) else None
                    default_factor_type = "object" if old_factor is None else old_factor.get("type", "object")

                    cfa, cfb, cfc = st.columns([1.5, 2.5, 2])

                    with cfa:
                        factor_type = st.selectbox(
                            f"Tipo factor LHS {f+1} término {t+1} en {family_name}",
                            options=["object", "constant"],
                            index=0 if default_factor_type == "object" else 1,
                            format_func=lambda x: "Parámetro/Variable" if x == "object" else "Constante",
                            key=f"lhs_factor_type_{r}_{t}_{f}"
                        )

                    if factor_type == "object":
                        default_label = expr_catalog_labels[0] if len(expr_catalog_labels) > 0 else None
                        if old_factor is not None and old_factor.get("type") == "object":
                            default_label = old_factor.get("label", default_label)

                        if len(expr_catalog_labels) == 0:
                            st.error("No hay objetos disponibles.")
                        else:
                            with cfb:
                                chosen_label = st.selectbox(
                                    f"Objeto LHS factor {f+1} término {t+1} en {family_name}",
                                    options=expr_catalog_labels,
                                    index=expr_catalog_labels.index(default_label) if default_label in expr_catalog_labels else 0,
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
                        default_value = 1.0
                        if old_factor is not None and old_factor.get("type") == "constant":
                            default_value = float(old_factor.get("value", 1.0))

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

                st.write(f"Vista previa LHS término {t+1}: `{build_term_text(term_record)}`")

            # -----------------------
            # LADO DERECHO
            # -----------------------
            st.markdown(f"### Lado derecho de {family_name}")

            old_rhs = [] if old_family is None else old_family.get("rhs_terms", [])

            rhs_num_terms = st.number_input(
                f"Número de términos RHS en {family_name}",
                min_value=1,
                max_value=20,
                value=max(1, len(old_rhs) if len(old_rhs) > 0 else 1),
                step=1,
                key=f"rhs_num_terms_{r}"
            )

            rhs_terms = []

            for t in range(int(rhs_num_terms)):
                st.markdown(f"**Término RHS {t+1}**")

                old_term = old_rhs[t] if t < len(old_rhs) else None

                c1, c2, c3 = st.columns([1, 2, 2])

                with c1:
                    sign = st.selectbox(
                        f"Signo RHS término {t+1} en {family_name}",
                        options=["+", "-"],
                        index=0 if old_term is None else (0 if old_term.get("sign", "+") == "+" else 1),
                        key=f"rhs_sign_{r}_{t}"
                    )

                with c2:
                    num_factors = st.number_input(
                        f"Número de factores RHS término {t+1} en {family_name}",
                        min_value=1,
                        max_value=4,
                        value=1 if old_term is None else max(1, len(old_term.get("factors", []))),
                        step=1,
                        key=f"rhs_num_factors_{r}_{t}"
                    )

                with c3:
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
                    old_factor = old_factors[f] if f < len(old_factors) else None
                    default_factor_type = "object" if old_factor is None else old_factor.get("type", "object")

                    cfa, cfb, cfc = st.columns([1.5, 2.5, 2])

                    with cfa:
                        factor_type = st.selectbox(
                            f"Tipo factor RHS {f+1} término {t+1} en {family_name}",
                            options=["object", "constant"],
                            index=0 if default_factor_type == "object" else 1,
                            format_func=lambda x: "Parámetro/Variable" if x == "object" else "Constante",
                            key=f"rhs_factor_type_{r}_{t}_{f}"
                        )

                    if factor_type == "object":
                        default_label = expr_catalog_labels[0] if len(expr_catalog_labels) > 0 else None
                        if old_factor is not None and old_factor.get("type") == "object":
                            default_label = old_factor.get("label", default_label)

                        if len(expr_catalog_labels) == 0:
                            st.error("No hay objetos disponibles.")
                        else:
                            with cfb:
                                chosen_label = st.selectbox(
                                    f"Objeto RHS factor {f+1} término {t+1} en {family_name}",
                                    options=expr_catalog_labels,
                                    index=expr_catalog_labels.index(default_label) if default_label in expr_catalog_labels else 0,
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

                st.write(f"Vista previa RHS término {t+1}: `{build_term_text(term_record)}`")

            family_record = {
                "name": family_name,
                "forall": forall,
                "sense": sense,
                "lhs_terms": lhs_terms,
                "rhs_terms": rhs_terms
            }

            family_errors = validate_constraint_family(family_record)

            st.markdown(f"### Vista previa de la familia {family_name}")
            st.code(build_constraint_family_text(family_record), language="text")

            if len(family_errors) > 0:
                for err in family_errors:
                    st.error(err)
            else:
                st.success("La familia de restricciones está estructuralmente consistente.")

            new_constraint_families.append(family_record)

        spec["constraints"] = new_constraint_families

        # ====================================================
        # 2.4 RESUMEN GENERAL
        # ====================================================
        st.markdown("---")
        st.subheader("4. Resumen de la definición del modelo")

        st.write("**Función objetivo:**")
        st.code(f"{sense_symbol}  {obj_text}", language="text")

        st.write("**Familias de restricciones:**")
        if len(new_constraint_families) == 0:
            st.info("No hay familias de restricciones definidas.")
        else:
            for fam in new_constraint_families:
                st.code(build_constraint_family_text(fam), language="text")

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
