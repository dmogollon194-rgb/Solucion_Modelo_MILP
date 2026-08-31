# Linear Model Solver

A Streamlit application for building, validating, solving, and inspecting single-objective linear optimization models through a graphical interface.

The application uses **Pyomo** to construct the mathematical model and supports continuous, integer, and binary decision variables.

---

## Features

### 1. Data Input

The **Data Input** section is organized into three tabs:

- **Indices**
- **Parameters**
- **Variables**

#### Indices

Each index is defined by:

- a symbolic name, such as `i`, `j`, or `k`;
- a finite size.

For example, an index `j` with size `21` is internally represented as:

```text
j1, j2, ..., j21
```

Index positions are interpreted numerically from `1` to `N_j`.

#### Parameters

Parameters may be:

- scalar;
- one-dimensional;
- multidimensional.

Three input modes are available:

- **Manual**
- **Excel/CSV**
- **Random**

For parameter arrays with more than **12 elements**, manual input is disabled and the application automatically offers:

- **Excel/CSV**
- **Random**

##### Excel/CSV format

Uploaded parameter files must contain **numeric values only**.

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

Values may be entered in one row or one column. They are read from **left to right and then top to bottom**.

Example:

```text
3;2;1;3;3;1;4
```

or:

```text
3
2
1
3
3
1
4
```

The number of uploaded values must exactly match the number of elements required by the parameter.

The application can also generate downloadable CSV and Excel templates.

#### Variables

Decision variables are defined by:

- name;
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

with domain:

```text
Binary
```

---

## 2. Model Definition

The **Model Definition** section contains:

- **Objective Function**
- **Constraints**
- **Mathematical Model**

### Objective Function

The application supports one objective function with either:

- **minimize**
- **maximize**

The objective is constructed from multiple terms.

Complete terms may be connected using:

- `+` Add
- `−` Subtract
- `×` Multiply
- `÷` Divide

Factors inside each term may also be connected using multiplication or division.

Example:

```text
WH × TN - 8 × sum(i) sum(j) x[i,j]
```

which represents:

\[
\min Z =
WH \cdot TN -
8\sum_i\sum_j x_{ij}.
\]

Objective terms are displayed in **collapsible panels**. The application preserves the active panel when a control inside the term is modified.

### Summations

Each summation is defined by:

- **Sum index**
- **Lower bound**
- **Upper bound**

For a complete index:

```text
Sum index: i
Lower bound: 1
Upper bound: N_i
```

represents:

\[
\sum_{i=1}^{N_i}.
\]

`N_i` means the total size of index `i`.

Dynamic bounds are supported.

Examples:

```text
j + 2
2*j + 1
N_i - 1
k
k + 2
```

For example:

```text
Sum index: j
Lower bound: k
Upper bound: k+2
```

represents:

\[
\sum_{j=k}^{k+2}.
\]

A summation bound may depend on a free or outer index, but it cannot depend on its own summation index.

For nested summations, **Summation 1 is the outermost summation**.

### Constraints

Constraints are organized into collapsible **constraint families**.

Each family contains:

- family name;
- `For all` indices;
- relational operator;
- left-hand side terms;
- right-hand side terms.

Supported relational operators:

- `<=`
- `>=`
- `=`

Example:

\[
\sum_{j=k}^{k+2}x_{ij}\leq 1
\qquad \forall i,k.
\]

The application validates:

- undefined indices;
- duplicated summation indices;
- invalid summation bounds;
- free indices;
- `For all` consistency;
- structural compatibility between both sides of a constraint.

### Mathematical Model

The **Mathematical Model** tab displays the complete model in mathematical notation before it is sent to the solver.

This allows the user to review:

- objective function;
- constraint families;
- summation bounds;
- quantifiers;
- relational operators.

---

## 3. Model Outputs

Before solving, the application validates the model specification.

The current implementation is designed for **linear models**.

It rejects structures such as:

\[
x_i x_j
\]

or:

\[
\frac{x_i}{x_j},
\]

because these expressions are nonlinear.

Multiplication or division by constants and parameters may be used when the resulting expression remains linear.

### Supported Solvers

The interface currently supports:

- **HiGHS**
- **GLPK**
- **CBC**

HiGHS is the recommended default solver.

After solving, the application displays:

- solver;
- solver status;
- termination condition;
- optimal objective value.

### Solution Variables

The solution can be inspected in two ways:

#### Select Variable

Displays the complete solution table for a selected decision variable.

#### Nonzero Variables

Displays only decision-variable components whose absolute value is greater than:

```text
1e-9
```

Solution tables can be exported as CSV files.

---

## Requirements

Python 3.10 or later is recommended.

Core Python packages:

```text
streamlit
pandas
numpy
pyomo
```

For Excel support:

```text
openpyxl
xlsxwriter
```

For HiGHS:

```text
highspy
```

Install the Python dependencies with:

```bash
pip install streamlit pandas numpy pyomo openpyxl xlsxwriter highspy
```

GLPK and CBC are external solvers and must be installed separately if they are going to be used.

### Ubuntu / Debian

GLPK:

```bash
sudo apt update
sudo apt install glpk-utils
```

CBC:

```bash
sudo apt install coinor-cbc
```

---

## Running the Application

Place the application file in the project directory.

Example structure:

```text
linear-model-solver/
├── linear_model_builder_english.py
└── README.md
```

Run:

```bash
streamlit run linear_model_builder_english.py
```

Streamlit will display the local address in the terminal, usually:

```text
http://localhost:8501
```

---

## Recommended Workflow

1. Open **Data Input**.
2. Define all indices.
3. Define and load the parameters.
4. Define the decision variables and their domains.
5. Open **Model Definition**.
6. Construct the objective function.
7. Define each constraint family.
8. Review the complete formulation in **Mathematical Model**.
9. Open **Model Outputs**.
10. Select a solver.
11. Solve the model.
12. Inspect or export the decision-variable values.

---

## Dynamic Index Example

Suppose:

```text
i = 1,...,10
j = 1,...,21
k = 1,...,19
```

and:

```text
x[i,j]
```

is binary.

The constraint:

\[
\sum_{j=k}^{k+2}x_{ij}\leq1
\qquad \forall i,k
\]

can be entered as:

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

For `k = 1`, this generates:

\[
x_{i1}+x_{i2}+x_{i3}\leq1.
\]

For `k = 2`:

\[
x_{i2}+x_{i3}+x_{i4}\leq1.
\]

The final window for `k = 19` is:

\[
x_{i19}+x_{i20}+x_{i21}\leq1.
\]

---

## Parameter Upload Example

If parameter `MIN[j]` has 21 elements, the uploaded CSV must contain exactly 21 numeric values.

Valid:

```text
3;2;1;3;3;1;4;2;1;4;2;1;3;2;1;1;1;1;1;1;1
```

Invalid:

```text
MIN
3
2
1
...
```

Invalid:

```text
j,value
j1,3
j2,2
...
```

The application assigns uploaded values according to the internal order of the parameter indices.

---

## Model Validation

The application performs validation before solving, including:

- missing objective function;
- missing indices;
- missing variables;
- invalid symbol names;
- duplicated names;
- inconsistent free indices;
- invalid `For all` indices;
- invalid dynamic summation expressions;
- division by zero where detectable;
- multiplication of decision-variable expressions that produces nonlinearity;
- division by a decision-variable expression.

A model must pass validation before it is sent to Pyomo.

---

## Notes

- The application currently supports a **single objective function**.
- The optimization model is built dynamically with **Pyomo**.
- Index sets are finite and ordered.
- Parameter values are stored according to the Cartesian product of their associated indices.
- Dynamic summation limits are interpreted using index positions.
- Solver availability depends on the packages and executables installed in the local environment.

---

## Author

**M.Sc. Dilan Mogollón**
