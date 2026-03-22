import streamlit as st
import pandas as pd
import pyomo.environ as pyo

st.title("Optimizador algebraico")

tipo = st.selectbox("Tipo de problema", ["Minimizar", "Maximizar"])

m = st.number_input("Número de filas", min_value=1, value=3)
n = st.number_input("Número de columnas", min_value=1, value=3)

st.subheader("Matriz de costos C")
C = pd.DataFrame([[0.0]*n for _ in range(m)])
C_edit = st.data_editor(C, use_container_width=True, key="C")

st.subheader("Vector oferta")
oferta = pd.DataFrame({"oferta": [0.0]*m})
oferta_edit = st.data_editor(oferta, use_container_width=True, key="oferta")

st.subheader("Vector demanda")
demanda = pd.DataFrame({"demanda": [0.0]*n})
demanda_edit = st.data_editor(demanda, use_container_width=True, key="demanda")

if st.button("Resolver"):
    model = pyo.ConcreteModel()

    I = range(m)
    J = range(n)

    model.x = pyo.Var(I, J, domain=pyo.NonNegativeReals)

    if tipo == "Minimizar":
        model.obj = pyo.Objective(
            expr=sum(float(C_edit.iloc[i, j]) * model.x[i, j] for i in I for j in J),
            sense=pyo.minimize
        )
    else:
        model.obj = pyo.Objective(
            expr=sum(float(C_edit.iloc[i, j]) * model.x[i, j] for i in I for j in J),
            sense=pyo.maximize
        )

    model.oferta = pyo.ConstraintList()
    for i in I:
        model.oferta.add(sum(model.x[i, j] for j in J) <= float(oferta_edit.iloc[i, 0]))

    model.demanda = pyo.ConstraintList()
    for j in J:
        model.demanda.add(sum(model.x[i, j] for i in I) >= float(demanda_edit.iloc[j, 0]))

    solver = pyo.SolverFactory("appsi_highs")
    result = solver.solve(model)

    st.write("Estado:", result.solver.termination_condition)
    st.write("Valor óptimo:", pyo.value(model.obj))

    X_sol = pd.DataFrame(
        [[pyo.value(model.x[i, j]) for j in J] for i in I],
        index=[f"i{i+1}" for i in I],
        columns=[f"j{j+1}" for j in J]
    )

    st.subheader("Matriz solución")
    st.dataframe(X_sol, use_container_width=True)

    csv = X_sol.to_csv().encode("utf-8")
    st.download_button(
        "Descargar solución CSV",
        data=csv,
        file_name="solucion.csv",
        mime="text/csv"
    )
