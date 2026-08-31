# MILP General

**MILP General** is a Streamlit application for building, validating, solving, and inspecting single-objective Mixed-Integer Linear Programming (MILP) models through a graphical interface.

The application dynamically constructs the optimization model with **Pyomo** and solves it using **HiGHS**.

---

## Project Structure

```text
MILP_General/
├── MILP_General.py
├── README.md
└── requirements.txt
```

### Files

- `MILP_General.py` — main Streamlit application.
- `README.md` — project documentation.
- `requirements.txt` — Python dependencies required to run the application.

---

## Main Features

The application is organized into three main sections:

1. **Data Input**
2. **Model Definition**
3. **Model Outputs**

---

# 1. Data Input

This section is used to define the basic components of the optimization model.

It contains three tabs:

- **Indices**
- **Parameters**
- **Variables**

## Indices

Each index is defined by:

- a symbolic name;
- a finite size.

Example:

```text
Index name: j
Size: 21
```

The application internally generates:

```text
j1, j2, ..., j21
```

The corresponding positions are interpreted as:

\[
j=1,\ldots,21.
\]

The size of an index can be referenced in dynamic expressions using:

```text
N_j
```

For example, if `j` contains 21 elements:

\[
N_j=21.
\]

---

## Parameters

Parameters may be:

- scalar;
- one-dimensional;
- multidimensional.

Available input modes are:

- **Manual**
- **Excel/CSV**
- **Random**

When a parameter contains more than **12 elements**, manual entry is disabled and the application offers:

- **Excel/CSV**
- **Random**

### Manual Input

Manual input is intended for parameters with a relatively small number of elements.

The application displays an editable table according to the indices associated with the parameter.

### Excel/CSV Input

Parameter values can be uploaded from:

- `.csv`
- `.xlsx`
- `.xls`

The uploaded file must contain **numeric values only**.

Do not include:

- column headers;
- row titles;
- index labels;
- index names;
- additional text.

CSV files may use:

- comma `,`;
- semicolon `;`;
- tab separators.

Values may be arranged horizontally:

```text
3;2;1;3;3;1;4
```

or vertically:

```text
3
2
1
3
3
1
4
```

Values are read:

1. from left to right;
2. then from top to bottom.

The number of values in the file must exactly match the number of elements required by the parameter.

The application also provides downloadable CSV and Excel templates.

### Random Input

Parameter values may also be randomly generated using:

- minimum value;
- maximum value;
- integer/continuous option;
- random seed.

---

## Variables

Decision variables are defined by:

- variable name;
- associated indices;
- mathematical domain.

Available domains:

- **Binary**
- **Nonnegative Reals**
- **Nonnegative Integers**

Example:

```text
x[i,j]
```

with:

```text
Domain: Binary
```

represents:

\[
x_{ij}\in\{0,1\}.
\]

---

# 2. Model Definition

This section is used to construct the optimization model.

It contains three tabs:

- **Objective Function**
- **Constraints**
- **Mathematical Model**

---

## Objective Function

The application supports one objective function.

Available objective senses:

- **minimize**
- **maximize**

The objective function is constructed from multiple terms.

### Operators Between Terms

Complete terms may be connected using:

- `+` Add
- `−` Subtract
- `×` Multiply
- `÷` Divide

For example:

\[
WH\cdot TN
-
8\sum_i\sum_jx_{ij}.
\]

### Operators Inside Terms

Factors inside a term may also be connected using:

- `×` Multiply
- `÷` Divide

This allows expressions involving:

- parameters;
- decision variables;
- constants;
- summations.

The application checks whether the resulting expression remains linear.

### Collapsible Terms

Each objective-function term is displayed inside a collapsible panel.

The active term remains open when its configuration is modified, avoiding unnecessary reopening after each Streamlit rerun.

---

## Summations

Each summation is defined by:

- **Sum index**
- **Lower bound**
- **Upper bound**

### Complete Index

For:

\[
\sum_{i=1}^{N_i},
\]

enter:

```text
Sum index: i
Lower bound: 1
Upper bound: N_i
```

`N_i` represents the total size of index `i`.

### Dynamic Bounds

Summation bounds may contain expressions such as:

```text
j+2
k+2
2*j+1
N_i-1
k
```

Example:

```text
Sum index: j
Lower bound: k
Upper bound: k+2
```

represents:

\[
\sum_{j=k}^{k+2}.
\]

A bound may depend on another free or outer index.

A bound cannot depend on its own summation index.

For nested summations:

- **Summation 1** is the outermost;
- **Summation 2** is inside Summation 1;
- subsequent summations continue inward.

---

## Constraints

Constraints are organized into **constraint families**.

Each family contains:

- family name;
- `For all` indices;
- relational operator;
- left-hand side;
- right-hand side.

Available relational operators:

- `<=`
- `>=`
- `=`

Example:

\[
\sum_{j=k}^{k+2}x_{ij}\leq1
\qquad \forall i,k.
\]

The corresponding configuration is:

```text
For all:
i, k

Sum index:
j

Lower bound:
k

Upper bound:
k+2

Operator:
<=

Right-hand side:
1
```

The application validates:

- undefined indices;
- duplicated summation indices;
- invalid dynamic bounds;
- free indices;
- `For all` consistency;
- structural compatibility of both sides;
- linearity.

---

## Dynamic Index Example

Suppose:

\[
i=1,\ldots,10,
\]

\[
j=1,\ldots,21,
\]

and:

\[
k=1,\ldots,19.
\]

Let:

\[
x_{ij}\in\{0,1\}.
\]

The family:

\[
\sum_{j=k}^{k+2}x_{ij}\leq1
\qquad\forall i,k
\]

generates moving windows of three consecutive positions.

For \(k=1\):

\[
x_{i1}+x_{i2}+x_{i3}\leq1.
\]

For \(k=2\):

\[
x_{i2}+x_{i3}+x_{i4}\leq1.
\]

For \(k=19\):

\[
x_{i19}+x_{i20}+x_{i21}\leq1.
\]

Because:

\[
19+2=21=N_j,
\]

the final window remains inside the valid range of index `j`.

---

## Mathematical Model

The **Mathematical Model** tab displays the complete structured formulation before solving.

It includes:

- objective function;
- constraint families;
- summation limits;
- relational operators;
- `For all` indices.

This tab should be reviewed before solving the model.

---

# 3. Model Outputs

Before solving, the application validates the complete model specification.

The application is designed for **linear and mixed-integer linear models**.

Expressions that introduce nonlinearity are rejected.

Examples of nonlinear expressions:

\[
x_i x_j
\]

and:

\[
\frac{x_i}{x_j}.
\]

Multiplication and division by constants or parameters are permitted when the resulting mathematical expression remains linear.

---

## Solver

The application uses:

### HiGHS

HiGHS is the only solver configured in the application.

It is accessed through Pyomo using:

```text
appsi_highs
```

HiGHS supports the continuous and mixed-integer linear models handled by the application.

---

## Solution Information

After solving, the application displays:

- solver status;
- termination condition;
- optimal objective value.

---

## Solution Variables

The **Solution Variables** tab contains two views.

### Select Variable

Displays all components of a selected decision variable.

### Nonzero Variables

Displays only components satisfying:

```text
abs(value) > 1e-9
```

This view is useful for binary and sparse optimization models.

Results can be exported as CSV files.

---

# Installation

## Python

Python 3.10 or later is recommended.

## Install Dependencies

From the project directory, run:

```bash
pip install -r requirements.txt
```

The project dependencies are:

```text
streamlit
pandas
numpy
pyomo
highspy
openpyxl
xlsxwriter
```

No separate GLPK or CBC installation is required.

---

# Running the Application

From the project directory:

```bash
streamlit run MILP_General.py
```

Streamlit will start the application and normally provide a local URL similar to:

```text
http://localhost:8501
```

---

# Recommended Workflow

1. Open **Data Input**.
2. Define the model indices.
3. Define the parameters.
4. Enter, upload, or randomly generate parameter values.
5. Define the decision variables.
6. Open **Model Definition**.
7. Construct the objective function.
8. Define the constraint families.
9. Review the formulation in **Mathematical Model**.
10. Open **Model Outputs**.
11. Solve the model with **HiGHS**.
12. Inspect the optimal objective value.
13. Review or export the solution variables.

---

# Model Validation

Before solving, the application checks for:

- missing objective function;
- missing indices;
- missing variables;
- invalid symbolic names;
- duplicated names;
- undefined indices;
- duplicated summation indices;
- invalid dynamic summation expressions;
- inconsistent free indices;
- inconsistent `For all` definitions;
- division by zero where detectable;
- nonlinear products involving decision variables;
- division by expressions containing decision variables.

The model must pass validation before it is sent to Pyomo and HiGHS.

---

# Parameter Upload Example

Suppose:

```text
MIN[j]
```

contains 21 elements.

A valid CSV may contain:

```text
3;2;1;3;3;1;4;2;1;4;2;1;3;2;1;1;1;1;1;1;1
```

The following format is invalid:

```text
MIN
3
2
1
...
```

The following format is also invalid:

```text
j,value
j1,3
j2,2
...
```

Only numeric values should be uploaded.

---

# Technical Stack

- **Python**
- **Streamlit**
- **Pyomo**
- **HiGHS**
- **Pandas**
- **NumPy**

Optional Excel support uses:

- **openpyxl**
- **xlsxwriter**

---

# Current Scope

The current version supports:

- one objective function;
- finite ordered index sets;
- scalar and multidimensional parameters;
- manual, file-based, and random parameter input;
- continuous, integer, and binary variables;
- dynamic summation bounds;
- constraint families;
- linearity validation;
- mathematical preview;
- MILP solution using HiGHS;
- CSV export of solution variables.

---

# Author

**M.Sc. Dilan Mogollón**
