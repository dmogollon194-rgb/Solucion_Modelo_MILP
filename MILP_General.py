import streamlit as st
import pandas as pd
import numpy as np
import itertools
import io
from typing import Any
import pyomo.environ as pyo

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(page_title="Constructor de Modelos Algebraicos", layout="wide")

# ── STYLES + WATERMARK ────────────────────────────────────
st.markdown("""<style>
.watermark{position:fixed;top:150px;right:25px;opacity:.95;font-size:22px;font-weight:900;color:#ff4b4b;text-shadow:1px 1px 2px #000;z-index:2000}
.stApp{background:linear-gradient(180deg,#07101f 0%,#050b16 100%);color:#f3f7ff}
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1400px}
h1,h2,h3,h4,h5,h6,p,label,div,span{color:#f3f7ff}
.top-hero{background:linear-gradient(135deg,rgba(8,22,55,.95),rgba(3,10,28,.98));border:1px solid rgba(61,132,255,.22);border-radius:22px;padding:22px 26px;margin-bottom:16px;box-shadow:0 0 0 1px rgba(61,132,255,.06),0 10px 35px rgba(0,0,0,.35)}
.top-hero h2{margin:0 0 8px;font-size:1.55rem;font-weight:800;color:#fff}
.top-hero p{margin:0;font-size:1rem;color:#d7e6ff}
.kpi-card{background:linear-gradient(135deg,rgba(8,22,55,.95),rgba(3,10,28,.98));border:1px solid rgba(61,132,255,.22);border-radius:18px;padding:16px 18px;min-height:120px;display:flex;flex-direction:column;justify-content:center;box-shadow:0 0 0 1px rgba(61,132,255,.05),0 10px 28px rgba(0,0,0,.28);margin-bottom:10px}
.kpi-title{font-size:1.02rem;font-weight:700;color:#fff;margin-bottom:8px}
.kpi-value{font-size:2.25rem;font-weight:800;color:#fff;line-height:1.05;word-break:break-word}
.section-box{background:rgba(5,12,28,.78);border:1px solid rgba(61,132,255,.16);border-radius:18px;padding:18px 18px 14px}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#08101f 0%,#050b16 100%);border-right:1px solid rgba(61,132,255,.18)}
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,#0c2b69,#0a1f49);color:#fff;border:1px solid rgba(100,162,255,.45);border-radius:12px;font-weight:700;padding:.6rem 1rem}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"]>div,.stMultiSelect div[data-baseweb="select"]>div,.stTextArea textarea{background-color:rgba(7,16,35,.92)!important;color:#fff!important;border-radius:12px!important}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{background:rgba(8,22,55,.72);border-radius:12px 12px 0 0;padding:10px 18px;color:#dfeaff;font-weight:700}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#12306f,#0b1f4a);color:#fff!important}
hr{border-color:rgba(61,132,255,.18)}
</style><div class="watermark">by M.Sc. Dilan Mogollón</div>""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────
if "model_spec" not in st.session_state:
    st.session_state["model_spec"] = {"indices":{},"parameters":{},"variables":{},"objective":None,"constraints":[],"results":None}
if "constraint_family_expander_abierto" not in st.session_state:
    st.session_state["constraint_family_expander_abierto"] = None
spec = st.session_state["model_spec"]

# ── CORE UTILITIES ────────────────────────────────────────
def valid_sym(n):
    n=(n or"").strip(); return bool(n) and (n[0].isalpha() or n[0]=="_") and all(c.isalnum() or c=="_" for c in n[1:])
def idx_elems(size,prefix): return [f"{prefix}{i}" for i in range(1,size+1)]
def combos(names,specs): return list(itertools.product(*[specs[n]["elements"] for n in names])) if names else [tuple()]
def total_elems(names,specs): r=1; [r:=r*specs[k]["size"] for k in names]; return r if names else 1
def sig(name,idxs): return f"{name}[{', '.join(idxs)}]" if idxs else name
def scalar_get(v,d=0.0): return float(v.get("__scalar__",d))
def scalar_set(v): return {"__scalar__":float(v)}

# ── VALUES SERIALIZATION ──────────────────────────────────
def df_to_vals(df,idx_names):
    if len(idx_names)==1: return {str((str(r["label"]),)):float(r["value"]) for _,r in df.iterrows()}
    return {str(tuple(str(r[i]) for i in idx_names)):float(r["value"]) for _,r in df.iterrows()}

def vals_to_df(names,clist,vals):
    return pd.DataFrame([{**{n:c[i] for i,n in enumerate(names)},"value":float(vals.get(str(c),0.0))} for c in clist])

def vals_to_df1d(labels,vals):
    return pd.DataFrame([{"label":l,"value":float(vals.get(str((l,)),0.0))} for l in labels])

def rand_vals(clist,lo,hi,integer,seed):
    rng=np.random.default_rng(seed)
    fn=(lambda:int(rng.integers(int(lo),int(hi)+1))) if integer else (lambda:float(rng.uniform(lo,hi)))
    return {str(c):float(fn()) for c in clist}

def rand_scalar(lo,hi,integer,seed):
    rng=np.random.default_rng(seed)
    return {"__scalar__":float(int(rng.integers(int(lo),int(hi)+1)) if integer else rng.uniform(lo,hi))}

# ── PARAMETER PERSISTENCE ─────────────────────────────────
def _pkey(pos): return f"_pstore_{pos}"
def _psig(name,idxs): return f"{name}__{'_'.join(idxs) or 'scalar'}"
def _empty_vals(idxs,specs): return {"__scalar__":0.0} if not idxs else {str(c):0.0 for c in combos(idxs,specs)}

def _vals_ok(vals,idxs,specs):
    if not isinstance(vals,dict): return False
    return "__scalar__" in vals if not idxs else {str(c) for c in combos(idxs,specs)}.issubset(vals)

def _init_vals(pos,name,idxs,specs,old):
    k=_pkey(pos); s=st.session_state.get(k)
    if _vals_ok(s,idxs,specs): return dict(s)
    if _vals_ok(old,idxs,specs): st.session_state[k]=dict(old); return dict(old)
    fresh=_empty_vals(idxs,specs); st.session_state[k]=dict(fresh); return fresh

def _save_vals(pos,vals): st.session_state[_pkey(pos)]=dict(vals); return dict(vals)

def _tmpl_df(idxs,specs,vals):
    if not idxs: return pd.DataFrame([{"value":scalar_get(vals)}])
    return vals_to_df1d(specs[idxs[0]]["elements"],vals) if len(idxs)==1 else vals_to_df(idxs,combos(idxs,specs),vals)

# ── EXCEL/CSV HELPERS ─────────────────────────────────────
def _to_csv(df): return df.to_csv(index=False).encode("utf-8-sig")

def _to_xlsx(df):
    buf=io.BytesIO()
    try:
        with pd.ExcelWriter(buf,engine="xlsxwriter") as w: df.to_excel(w,index=False,sheet_name="parametro")
        return buf.getvalue()
    except Exception: return None

def _read_upload(f):
    name=f.name.lower()
    try:
        if name.endswith(".csv"): return pd.read_csv(f),None
        if name.endswith((".xlsx",".xls")):
            try: return pd.read_excel(f),None
            except (ImportError,ModuleNotFoundError): return None,"Instala `openpyxl` para cargar .xlsx o usa .csv."
        return None,"Formato no soportado. Usa .csv o .xlsx."
    except Exception as e: return None,f"No se pudo leer el archivo: {e}"

def _validate_upload(df,idxs,specs):
    if df is None or df.empty: return None,["El archivo está vacío."]
    df=df.copy(); df.columns=[str(c).strip() for c in df.columns]
    if "value" not in df.columns: return None,["Falta la columna `value`."]
    try: df["value"]=pd.to_numeric(df["value"],errors="raise")
    except Exception: return None,["La columna `value` debe ser numérica."]
    errs=[]
    if not idxs: return {"__scalar__":float(df.iloc[0]["value"])},[]
    if len(idxs)==1:
        idx=idxs[0]; col="label" if "label" in df.columns else idx if idx in df.columns else None
        if col is None: return None,[f"Falta columna `label` o `{idx}`."]
        w=df[[col,"value"]].copy(); w[col]=w[col].astype(str)
        exp=set(specs[idx]["elements"]); obs=set(w[col])
        if w[col].duplicated().any(): errs.append(f"Etiquetas repetidas: {w.loc[w[col].duplicated(),col].unique().tolist()}")
        if exp-obs: errs.append(f"Faltan etiquetas: {sorted(exp-obs)}")
        if obs-exp: errs.append(f"Etiquetas no válidas: {sorted(obs-exp)}")
        if errs: return None,errs
        return {str((str(r[col]),)):float(r["value"]) for _,r in w.iterrows()},[]
    req=idxs+["value"]; miss=[c for c in req if c not in df.columns]
    if miss: return None,[f"Faltan columnas: {miss}"]
    w=df[req].copy()
    for i in idxs: w[i]=w[i].astype(str)
    if w.duplicated(subset=idxs).any(): errs.append("Hay combinaciones repetidas.")
    exp=set(combos(idxs,specs)); obs={tuple(r[i] for i in idxs) for _,r in w.iterrows()}
    if exp-obs: errs.append(f"Faltan combinaciones: {sorted(exp-obs)}")
    if obs-exp: errs.append(f"Combinaciones no válidas: {sorted(obs-exp)}")
    if errs: return None,errs
    return {str(tuple(str(r[i]) for i in idxs)):float(r["value"]) for _,r in w.iterrows()},[]

def param_upload_ui(pos,name,idxs,specs,cur_vals):
    tmpl=_tmpl_df(idxs,specs,cur_vals); base=f"plantilla_{name}"; wsuf=_psig(name,idxs)
    st.caption("Descarga la plantilla, rellena la columna `value` y vuelve a cargar.")
    d1,d2=st.columns(2)
    d1.download_button("Descargar CSV",_to_csv(tmpl),f"{base}.csv","text/csv",key=f"tcsv_{pos}_{wsuf}")
    xb=_to_xlsx(tmpl)
    if xb: d2.download_button("Descargar Excel",xb,f"{base}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f"txlsx_{pos}_{wsuf}")
    else: d2.info("Instala `xlsxwriter` para descargar .xlsx.")
    up=st.file_uploader(f"Cargar {name}",type=["csv","xlsx","xls"],key=f"up_{pos}_{wsuf}")
    if up is None: st.dataframe(tmpl,use_container_width=True,hide_index=True); return cur_vals
    df,err=_read_upload(up)
    if err: st.error(err); st.dataframe(tmpl,use_container_width=True,hide_index=True); return cur_vals
    vals,errs=_validate_upload(df,idxs,specs)
    if errs:
        for e in errs: st.error(e)
        st.dataframe(df,use_container_width=True,hide_index=True); return cur_vals
    st.success("Valores cargados correctamente.")
    _save_vals(pos,vals); st.dataframe(_tmpl_df(idxs,specs,vals),use_container_width=True,hide_index=True)
    return vals

# ── LATEX BUILDERS ────────────────────────────────────────
DOMAIN_LABELS={"Binary":"Binarias","NonNegativeReals":"Reales ≥ 0","NonNegativeIntegers":"Enteras ≥ 0"}
SENSE_MAP={"<=":r"\leq",">=":r"\geq","=":"="}

def _fac_latex(f):
    if f["type"]=="constant": v=float(f["value"]); return str(int(v)) if v==int(v) else f"{v:.2f}"
    n,ix=f["name"],f["indices"]; return n if not ix else rf"{n}_{{{','.join(ix)}}}"

def term_latex(t):
    body=r" \cdot ".join(_fac_latex(f) for f in t.get("factors",[])) or "0"
    for ix in t.get("sum_over",[]): body=rf"\sum_{{{ix}}}\left({body}\right)"
    return f"- {body}" if t.get("sign")=="-" else f"+ {body}"

def expr_latex(terms):
    if not terms: return "0"
    out=" ".join(term_latex(t) for t in terms).strip()
    return out[2:] if out.startswith("+ ") else out

def family_latex(fam):
    txt=f"{expr_latex(fam.get('lhs_terms',[]))} {SENSE_MAP.get(fam.get('sense','<='),r'\leq')} {expr_latex(fam.get('rhs_terms',[]))}"
    if fam.get("forall"): txt+=r"\quad \forall "+", ".join(fam["forall"])
    return txt

def term_free(t):
    seen,out=set(),[]
    for f in t.get("factors",[]):
        if f["type"]=="object":
            for x in f["indices"]:
                if x not in seen: seen.add(x); out.append(x)
    return [x for x in out if x not in t.get("sum_over",[])]

# ── VALIDATION ────────────────────────────────────────────
def validate_obj(terms):
    return [f"Término FO {i}: índices libres → {', '.join(term_free(t))}" for i,t in enumerate(terms,1) if term_free(t)]

def validate_family(fam):
    errs=[]; lf=list({x for t in fam.get("lhs_terms",[]) for x in term_free(t)}); rf=list({x for t in fam.get("rhs_terms",[]) for x in term_free(t)}); fa=fam.get("forall",[])
    lc,rc=not lf,not rf
    if not lc and not rc:
        if sorted(lf)!=sorted(rf): errs.append(f"LHS {lf} ≠ RHS {rf}")
        if sorted(lf)!=sorted(fa): errs.append(f"Índices libres {lf} ≠ forall {fa}")
    elif lc and not rc:
        if sorted(rf)!=sorted(fa): errs.append(f"RHS {rf} debe coincidir con forall {fa}")
    elif not lc and rc:
        if sorted(lf)!=sorted(fa): errs.append(f"LHS {lf} debe coincidir con forall {fa}")
    elif lc and rc and fa: errs.append(f"Ambos lados constantes pero hay forall {fa}")
    return errs

def validate_linearity(spec):
    errs=[]
    def chk(terms,ctx):
        for i,t in enumerate(terms,1):
            nv=sum(1 for f in t.get("factors",[]) if f["type"]=="object" and f.get("kind")=="variable")
            if nv>1: errs.append(f"{ctx} término {i}: {nv} variables → no lineal")
    obj=spec.get("objective")
    if obj: chk(obj.get("terms",[]),"FO")
    for r,fam in enumerate(spec.get("constraints",[]),1):
        n=fam.get("name",f"R{r}"); chk(fam.get("lhs_terms",[]),f"{n} LHS"); chk(fam.get("rhs_terms",[]),f"{n} RHS")
    return errs

# ── PYOMO ─────────────────────────────────────────────────
_DOMAINS={"Binary":pyo.Binary,"NonNegativeReals":pyo.NonNegativeReals,"NonNegativeIntegers":pyo.NonNegativeIntegers}
SOLVER_OPTIONS={"HiGHS (appsi_highs)":"appsi_highs","GLPK":"glpk","CBC":"cbc"}

def solver_get(label):
    name=SOLVER_OPTIONS.get(label)
    if not name: raise ValueError(f"Solver no soportado: {label}")
    s=pyo.SolverFactory(name)
    try: avail=s.available(exception_flag=False)
    except TypeError: avail=s.available()
    if not avail: raise RuntimeError(f"Solver `{name}` no disponible.")
    return name,s

def _get_val(m,f,env):
    if f["type"]=="constant": return float(f["value"])
    comp=getattr(m,f"{'par' if f['kind']=='parameter' else 'var'}_{f['name']}"); ix=f["indices"]
    if not ix: return comp
    key=tuple(env[i] for i in ix); return comp[key[0]] if len(key)==1 else comp[key]

def _eval_term(m,t,env):
    def rec(pos,e):
        if pos==len(t.get("sum_over",[])):
            val=1
            for f in t.get("factors",[]): val=val*_get_val(m,f,e)
            return -val if t.get("sign")=="-" else val
        ix=t["sum_over"][pos]; return sum(rec(pos+1,{**e,ix:v}) for v in getattr(m,f"set_{ix}"))
    return rec(0,dict(env))

def _expr(m,terms,env): return sum(_eval_term(m,t,env) for t in terms) if terms else 0

def build_model(spec):
    m=pyo.ConcreteModel(); idx=spec["indices"]
    for n,s in idx.items(): setattr(m,f"set_{n}",pyo.Set(initialize=s["elements"],ordered=True))
    for pn,ps in spec["parameters"].items():
        ix,v=ps["indices"],ps["values"]
        if not ix: setattr(m,f"par_{pn}",pyo.Param(initialize=float(v["__scalar__"])))
        else:
            init={c[0] if len(c)==1 else c:float(v.get(str(c),0.0)) for c in combos(ix,idx)}
            setattr(m,f"par_{pn}",pyo.Param(*[getattr(m,f"set_{i}") for i in ix],initialize=init))
    for vn,vs in spec["variables"].items():
        ix=vs["indices"]; dom=_DOMAINS[vs["domain"]]
        if not ix: setattr(m,f"var_{vn}",pyo.Var(domain=dom))
        else: setattr(m,f"var_{vn}",pyo.Var(*[getattr(m,f"set_{i}") for i in ix],domain=dom))
    obj=spec["objective"]; m.OBJ=pyo.Objective(expr=_expr(m,obj["terms"],{}),sense=pyo.minimize if obj["sense"]=="minimize" else pyo.maximize)
    ops={"<=":lambda a,b:a<=b,">=":lambda a,b:a>=b,"=":lambda a,b:a==b}
    for ci,fam in enumerate(spec.get("constraints",[]),1):
        nm=fam.get("name",f"R{ci}"); lt,rt,fa,s=fam.get("lhs_terms",[]),fam.get("rhs_terms",[]),fam.get("forall",[]),fam.get("sense","<=")
        if not fa: con=pyo.Constraint(expr=ops[s](_expr(m,lt,{}),_expr(m,rt,{})))
        else:
            def _rule(mdl,*args,_l=lt,_r=rt,_f=fa,_s=s): env=dict(zip(_f,args)); return ops[_s](_expr(mdl,_l,env),_expr(mdl,_r,env))
            con=pyo.Constraint(*[getattr(m,f"set_{i}") for i in fa],rule=_rule)
        setattr(m,f"con_{nm}",con)
    return m

def var_df(m,vn,vs,idx): 
    comp=getattr(m,f"var_{vn}"); ix=vs["indices"]
    if not ix: return pd.DataFrame({"variable":[vn],"value":[pyo.value(comp)]})
    return pd.DataFrame([{**{i:c[j] for j,i in enumerate(ix)},"value":pyo.value(comp[c[0]] if len(c)==1 else comp[c])} for c in combos(ix,idx)])

def all_vars_df(m,spec):
    dfs=[var_df(m,vn,vs,spec["indices"]).assign(variable_name=vn) for vn,vs in spec["variables"].items()]
    return pd.concat(dfs,ignore_index=True) if dfs else pd.DataFrame()

def count_exp(spec,key):
    ix=spec.get("indices",{}); items=spec.get(key,{}); col=items.values() if isinstance(items,dict) else items; total=0
    for it in col:
        ids=it.get("indices",it.get("forall",[]))
        if not ids: total+=1
        elif all(i in ix for i in ids): n=1; [n:=n*ix[i]["size"] for i in ids]; total+=n
    return total

# ── UI HELPERS ────────────────────────────────────────────
def hero(t,txt): st.markdown(f'<div class="top-hero"><h2>{t}</h2><p>{txt}</p></div>',unsafe_allow_html=True)
def kpi(t,v): st.markdown(f'<div class="kpi-card"><div class="kpi-title">{t}</div><div class="kpi-value">{v}</div></div>',unsafe_allow_html=True)
def sbox(t,txt=""): st.markdown(f'<div class="section-box"><b style="font-size:1.1rem">{t}</b>'+(f'<p style="color:#d7e6ff;margin-top:6px">{txt}</p>' if txt else "")+"</div>",unsafe_allow_html=True)
def _sb_kpi(l,v): return f'<div style="background:linear-gradient(135deg,rgba(8,22,55,.95),rgba(3,10,28,.98));border:1px solid rgba(61,132,255,.22);border-radius:14px;padding:12px 14px;margin-bottom:10px"><div style="font-size:.88rem;color:#cfe0ff;font-weight:700">{l}</div><div style="font-size:1.8rem;color:#fff;font-weight:800">{v}</div></div>'

def _rand_ui(key):
    c1,c2,c3,c4=st.columns(4)
    return c1.number_input("Mínimo",value=0.0,key=f"{key}_lo"),c2.number_input("Máximo",value=10.0,key=f"{key}_hi"),c3.checkbox("Entero",False,key=f"{key}_int"),int(c4.number_input("Semilla",value=123,step=1,key=f"{key}_seed"))

def _factor_ui(tkey,fi,old,catalog,lmap,dtype="object"):
    ca,cb,cc=st.columns([1.5,2.5,2])
    ft=ca.selectbox(f"Tipo {fi+1}",["object","constant"],index=0 if (old or{}).get("type",dtype)=="object" else 1,format_func=lambda x:"Parámetro/Variable" if x=="object" else "Constante",key=f"{tkey}_ft{fi}")
    if ft=="object":
        lbls=[o["label"] for o in catalog]
        if not lbls: st.error("Sin parámetros/variables."); return None
        def_l=lbls[0]
        if old and old.get("type")=="object" and old.get("label") in lbls: def_l=old["label"]
        ch=cb.selectbox(f"Objeto {fi+1}",lbls,index=lbls.index(def_l),key=f"{tkey}_fo{fi}"); it=lmap[ch]
        cc.write(f"Índices: {', '.join(it['indices']) or 'ninguno'}")
        return {"type":"object","kind":it["kind"],"name":it["name"],"indices":it["indices"],"label":it["label"]}
    dv=float((old or{}).get("value",0.0)) if (old or{}).get("type")=="constant" else 0.0
    return {"type":"constant","value":float(cb.number_input(f"Constante {fi+1}",value=dv,key=f"{tkey}_fc{fi}"))}

def _term_ui(tkey,ti,old,catalog,lmap,idx_names,dtype="object"):
    c1,c2,c3=st.columns([1,2,2]); o=old or{}
    sign=c1.selectbox(f"Signo {ti+1}",["+","-"],index=0 if o.get("sign","+")=='+' else 1,key=f"{tkey}_sg")
    nf=int(c2.number_input(f"Factores {ti+1}",min_value=1,max_value=4,value=max(1,len(o.get("factors",[]) or [])) if o.get("factors") else 2,step=1,key=f"{tkey}_nf"))
    so=c3.multiselect(f"Sumar sobre {ti+1}",idx_names,default=o.get("sum_over",[]),key=f"{tkey}_so")
    of=o.get("factors",[]); facs=[f for fi in range(nf) if (f:=_factor_ui(f"{tkey}f{fi}",fi,of[fi] if fi<len(of) else None,catalog,lmap,dtype)) is not None]
    t={"sign":sign,"factors":facs,"sum_over":so}; st.latex(term_latex(t)); return t

def obj_catalog(spec):
    items=[{"kind":"parameter","name":n,"indices":ps["indices"],"label":sig(n,ps["indices"])} for n,ps in spec["parameters"].items()]
    items+=[{"kind":"variable","name":n,"indices":vs["indices"],"label":sig(n,vs["indices"])} for n,vs in spec["variables"].items()]
    return items,{o["label"]:o for o in items}

def _open_fam(r): st.session_state["constraint_family_expander_abierto"]=r

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.markdown("""<div style="padding:.4rem 0 1rem;border-bottom:1px solid rgba(61,132,255,.18);margin-bottom:1rem"><div style="font-size:1.45rem;font-weight:800;color:#fff;margin-bottom:.35rem">Navegación</div><div style="color:#b9c9e8;font-size:.92rem">Explora cada etapa de construcción y solución.</div></div>""",unsafe_allow_html=True)
section=st.sidebar.radio("Ir a:",["Ingreso de información","Definición del modelo","Salidas del modelo"],index=0)
st.sidebar.markdown("---")
st.sidebar.markdown('<div style="font-size:1.05rem;font-weight:800;color:#fff;margin-bottom:.8rem">Estado actual</div>',unsafe_allow_html=True)
sb1,sb2=st.sidebar.columns(2)
sb1.markdown(_sb_kpi("Índices",len(spec["indices"])),unsafe_allow_html=True)
sb2.markdown(_sb_kpi("Variables",count_exp(spec,"variables")),unsafe_allow_html=True)
st.sidebar.markdown(_sb_kpi("Restricciones",count_exp(spec,"constraints")),unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────
st.title("Solucionador de Modelos Lineales")
st.caption("Aplicación para solucionar modelos lineales con un solo objetivo.")

# ═══════════════════════════════════════════════════════════
# SECCIÓN 1: INGRESO DE INFORMACIÓN
# ═══════════════════════════════════════════════════════════
if section=="Ingreso de información":
    hero("1. Ingreso de información","Define los índices, parámetros y variables del modelo.")
    c1,c2,c3=st.columns(3)
    with c1: kpi("Índices",len(spec["indices"]))
    with c2: kpi("Parámetros",len(spec["parameters"]))
    with c3: kpi("Variables",len(spec["variables"]))
    st.markdown("<br>",unsafe_allow_html=True)
    t_ind,t_par,t_var=st.tabs(["Índices","Parámetros","Variables"])

    # ── ÍNDICES ──
    with t_ind:
        sbox("Configuración de índices","Define los conjuntos base del modelo.")
        en=list(spec["indices"].keys()); n=int(st.number_input("Número de índices",1,10,max(1,len(en) or 3),step=1,key="num_idx"))
        new_idx,errs,used={},{},set()  # errs as list
        errs=[]
        for r in range(n):
            dn=en[r] if r<len(en) else f"idx_{r+1}"; ds=spec["indices"].get(dn,{}).get("size",3)
            c1,c2=st.columns(2)
            nm=c1.text_input(f"Nombre {r+1}",value=dn,key=f"idx_n{r}").strip()
            sz=int(c2.number_input(f"Tamaño de {nm or f'idx {r+1}'}",1,1000,int(ds),step=1,key=f"idx_s{r}"))
            if not valid_sym(nm): errs.append(f"`{nm}` no válido.")
            elif nm in used: errs.append(f"`{nm}` repetido.")
            else: used.add(nm); new_idx[nm]={"size":sz,"elements":idx_elems(sz,nm)}
        for e in errs: st.error(e)
        if not errs:
            spec["indices"]=new_idx; valid=set(new_idx)
            spec["parameters"]={k:v for k,v in spec["parameters"].items() if all(i in valid for i in v.get("indices",[]))}
            spec["variables"]={k:v for k,v in spec["variables"].items() if all(i in valid for i in v.get("indices",[]))}
            if new_idx:
                st.write("**Vista previa:**")
                st.dataframe(pd.DataFrame([{"Índice":n,"Tamaño":s["size"],"Elementos":", ".join(s["elements"])} for n,s in new_idx.items()]),use_container_width=True,hide_index=True)

    # ── PARÁMETROS ──
    with t_par:
        sbox("Configuración de parámetros","Define parámetros manualmente, por carga Excel/CSV o aleatoriamente.")
        idx_sp=spec["indices"]
        if not idx_sp: st.info("Primero define índices válidos.")
        else:
            cur=spec["parameters"]; np_=int(st.number_input("Número de parámetros",0,30,max(1,len(cur)) if cur else 1,step=1,key="num_p")); io=list(idx_sp.keys()); new_p={}
            for p in range(np_):
                st.markdown(f"#### Parámetro {p+1}")
                on=list(cur.keys()); oname=on[p] if p<len(on) else f"param_{p+1}"
                c1,c2=st.columns([2,3])
                pn=c1.text_input(f"Nombre {p+1}",value=oname,key=f"pn{p}").strip()
                pi=c2.multiselect(f"Índices de {pn}",io,default=cur.get(oname,{}).get("indices",[]),key=f"pi{p}")
                if not valid_sym(pn): st.error(f"`{pn}` no válido."); continue
                if pn in new_p: st.error(f"`{pn}` repetido."); continue
                ne=total_elems(pi,idx_sp); st.write(f"**Firma:** `{sig(pn,pi)}` — **Elementos:** `{ne}`")
                modes=["Manual","Excel/CSV","Aleatorio"] if ne<=12 else ["Excel/CSV","Aleatorio"]
                om=cur.get(oname,{}).get("mode",modes[0]); om="Excel/CSV" if om=="Excel" else om; om=om if om in modes else modes[0]
                mode=st.radio(f"Modo {pn}",modes,index=modes.index(om),horizontal=True,key=f"pm{p}")
                ov=cur.get(oname,{}).get("values",{}); cv=_init_vals(p,pn,pi,idx_sp,ov); rec={"indices":pi,"mode":mode,"values":dict(cv)}
                if mode=="Manual":
                    if not pi:
                        rec["values"]=_save_vals(p,scalar_set(st.number_input(f"Valor {pn}",value=scalar_get(cv),key=f"psc{p}")))
                    elif len(pi)==1:
                        lbs=idx_sp[pi[0]]["elements"]; ed=st.data_editor(vals_to_df1d(lbs,cv),use_container_width=True,num_rows="fixed",hide_index=True,disabled=["label"],key=f"pm1d{p}_{_psig(pn,pi)}")
                        rec["values"]=_save_vals(p,{str((str(r["label"]),)):float(r["value"]) for _,r in ed.iterrows()})
                    else:
                        cl=combos(pi,idx_sp); ed=st.data_editor(vals_to_df(pi,cl,cv),use_container_width=True,num_rows="fixed",hide_index=True,disabled=list(pi),key=f"pmnd{p}_{_psig(pn,pi)}")
                        rec["values"]=_save_vals(p,df_to_vals(ed,pi))
                elif mode=="Excel/CSV":
                    rec["values"]=param_upload_ui(p,pn,pi,idx_sp,cv)
                else:
                    if not pi:
                        lo,hi,intg,seed=_rand_ui(f"ps{p}")
                        if lo>hi: st.error("Mínimo > máximo."); continue
                        if st.button(f"Generar {pn}",key=f"pg{p}"): cv=_save_vals(p,rand_scalar(lo,hi,intg,seed))
                        st.write(f"Valor: **{scalar_get(cv):.4f}**"); rec["values"]=dict(cv)
                    else:
                        lo,hi,intg,seed=_rand_ui(f"pr{p}")
                        if lo>hi: st.error("Mínimo > máximo."); continue
                        if st.button(f"Generar {pn}",key=f"pg{p}"): cv=_save_vals(p,rand_vals(combos(pi,idx_sp),lo,hi,intg,seed))
                        rec["values"]=dict(cv); st.dataframe(_tmpl_df(pi,idx_sp,cv),use_container_width=True,hide_index=True)
                new_p[pn]=rec
            spec["parameters"]=new_p
            if new_p:
                st.write("**Resumen:**")
                st.dataframe(pd.DataFrame([{"Parámetro":sig(n,v["indices"]),"Modo":v["mode"],"Elementos":total_elems(v["indices"],idx_sp)} for n,v in new_p.items()]),use_container_width=True,hide_index=True)

    # ── VARIABLES ──
    with t_var:
        sbox("Configuración de variables","Variables de decisión y su dominio.")
        idx_sp=spec["indices"]
        if not idx_sp: st.info("Primero define índices válidos.")
        else:
            cur=spec["variables"]; nv=int(st.number_input("Número de variables",0,30,max(1,len(cur)) if cur else 1,step=1,key="num_v")); io=list(idx_sp.keys()); do=["Binary","NonNegativeReals","NonNegativeIntegers"]; new_v={}
            for v in range(nv):
                st.markdown(f"#### Variable {v+1}"); on=list(cur.keys()); oname=on[v] if v<len(on) else f"x_{v+1}"
                c1,c2,c3=st.columns([2,3,2])
                vn=c1.text_input(f"Nombre {v+1}",value=oname,key=f"vn{v}").strip()
                vi=c2.multiselect(f"Índices de {vn}",io,default=cur.get(oname,{}).get("indices",[]),key=f"vi{v}")
                od=cur.get(oname,{}).get("domain","NonNegativeReals"); vd=c3.selectbox(f"Dominio {vn}",do,index=do.index(od if od in do else "NonNegativeReals"),key=f"vd{v}")
                if not valid_sym(vn): st.error(f"`{vn}` no válido."); continue
                if vn in new_v: st.error(f"`{vn}` repetido."); continue
                new_v[vn]={"indices":vi,"domain":vd}
            spec["variables"]=new_v
            if new_v:
                st.write("**Resumen:**")
                st.dataframe(pd.DataFrame([{"Variable":sig(n,v["indices"]),"Dominio":DOMAIN_LABELS.get(v["domain"],v["domain"]),"Componentes":total_elems(v["indices"],idx_sp)} for n,v in new_v.items()]),use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════
# SECCIÓN 2: DEFINICIÓN DEL MODELO
# ═══════════════════════════════════════════════════════════
elif section=="Definición del modelo":
    hero("2. Definición del modelo","Función objetivo, restricciones y modelo matemático.")
    idx_sp=spec["indices"]
    if not idx_sp: st.warning("Primero define al menos un índice.")
    elif not spec["variables"]: st.warning("Primero define al menos una variable.")
    else:
        cat,lmap=obj_catalog(spec); inames=list(idx_sp.keys())
        t_fo,t_r,t_m=st.tabs(["Función Objetivo","Restricciones","Modelo matemático"])

        # ── FUNCIÓN OBJETIVO ──
        with t_fo:
            co=spec.get("objective") or{}; so=["minimize","maximize"]
            sense=st.radio("Objetivo:",so,index=so.index(co.get("sense","minimize")),horizontal=True,key="obj_s")
            ot=co.get("terms",[]); nt=int(st.number_input("Términos en la FO",1,20,max(1,len(ot) or 1),step=1,key="n_ot"))
            obj_terms=[]
            for t in range(nt):
                st.markdown(f"#### Término FO {t+1}")
                obj_terms.append(_term_ui(f"obj_t{t}",t,ot[t] if t<len(ot) else None,cat,lmap,inames))
            errs=validate_obj(obj_terms)
            for e in errs: st.error(e)
            if not errs: st.success("Función objetivo estructuralmente consistente.")
            spec["objective"]={"sense":sense,"terms":obj_terms}

        # ── RESTRICCIONES ──
        with t_r:
            of=spec.get("constraints",[]); nf=int(st.number_input("Familias de restricciones",0,30,len(of),step=1,key="n_fams")); new_fams=[]
            if nf==0: st.info("Sin restricciones definidas.")
            for r in range(nf):
                ofam=of[r] if r<len(of) else None; dn=(ofam or{}).get("name",f"R{r+1}")
                prev={"name":st.session_state.get(f"cfn{r}",dn),"forall":st.session_state.get(f"cff{r}",(ofam or{}).get("forall",[])),"sense":st.session_state.get(f"cfs{r}",(ofam or{}).get("sense","<=")),"lhs_terms":(ofam or{}).get("lhs_terms",[]),"rhs_terms":(ofam or{}).get("rhs_terms",[])}
                exp=(st.session_state.get("constraint_family_expander_abierto")==r or (st.session_state.get("constraint_family_expander_abierto") is None and r==0))
                with st.expander(f"Familia {r+1}: {prev['name']} — {family_latex(prev)}",expanded=exp):
                    st.markdown(f"### Familia {r+1}"); cf1,cf2,cf3=st.columns(3)
                    fn=cf1.text_input(f"Nombre {r+1}",value=dn,key=f"cfn{r}",on_change=_open_fam,args=(r,)).strip()
                    fa=cf2.multiselect(f"Para todo {fn}",inames,default=(ofam or{}).get("forall",[]),key=f"cff{r}",on_change=_open_fam,args=(r,))
                    sf=cf3.selectbox(f"Operador {fn}",["<=",">=","="],index=["<=",">=","="].index((ofam or{}).get("sense","<=")),key=f"cfs{r}",on_change=_open_fam,args=(r,))
                    if not valid_sym(fn): st.error(f"`{fn}` no válido."); continue
                    cL,cR=st.columns(2); ol=(ofam or{}).get("lhs_terms",[]); or_=(ofam or{}).get("rhs_terms",[])
                    with cL:
                        st.markdown(f"#### LHS de {fn}"); nl=int(st.number_input(f"Términos LHS {fn}",0,10,len(ol),step=1,key=f"nl{r}",on_change=_open_fam,args=(r,)))
                        lt=[_term_ui(f"lhs{r}_{t}",t,ol[t] if t<len(ol) else None,cat,lmap,inames) for t in range(nl)]
                    with cR:
                        st.markdown(f"#### RHS de {fn}"); nr=int(st.number_input(f"Términos RHS {fn}",0,10,len(or_),step=1,key=f"nr{r}",on_change=_open_fam,args=(r,)))
                        rt=[_term_ui(f"rhs{r}_{t}",t,or_[t] if t<len(or_) else None,cat,lmap,inames,dtype="constant") for t in range(nr)]
                    fr={"name":fn,"forall":fa,"sense":sf,"lhs_terms":lt,"rhs_terms":rt}
                    st.markdown(f"### Vista previa — {fn}"); st.latex(family_latex(fr))
                    fe=validate_family(fr)
                    for e in fe: st.error(e)
                    if not fe: st.success("Familia estructuralmente consistente.")
                    new_fams.append(fr)
            spec["constraints"]=new_fams
            if new_fams:
                st.markdown("---\n### Familias guardadas")
                for i,fam in enumerate(new_fams):
                    with st.expander(f"Familia {i+1}: {fam.get('name')} — {family_latex(fam)}",expanded=False): st.latex(family_latex(fam))

        # ── MODELO MATEMÁTICO ──
        with t_m:
            st.markdown("### Modelo estructurado"); obj=spec.get("objective")
            if obj: st.latex(rf"{'\\min' if obj['sense']=='minimize' else '\\max'}\ Z = {expr_latex(obj['terms'])}")
            else: st.info("Sin función objetivo definida.")
            st.markdown("**Sujeto a:**")
            if not spec["constraints"]: st.info("Sin restricciones definidas.")
            else: [st.latex(family_latex(fam)) for fam in spec["constraints"]]

# ═══════════════════════════════════════════════════════════
# SECCIÓN 3: SALIDAS DEL MODELO
# ═══════════════════════════════════════════════════════════
elif section=="Salidas del modelo":
    hero("3. Resultados","Solución óptima y configuración de las variables.")
    errs=[]
    if not spec["objective"]: errs.append("Sin función objetivo.")
    else: errs.extend(validate_obj(spec["objective"].get("terms",[])))
    for fam in spec.get("constraints",[]): errs.extend(validate_family(fam))
    errs.extend(validate_linearity(spec))
    if not spec["variables"]: errs.append("Sin variables definidas.")
    if not spec["indices"]: errs.append("Sin índices definidos.")
    for e in errs: st.error(e)
    if errs: st.stop()
    st.success("Especificación válida.")

    ts,tv=st.tabs(["Resolver","Variables solución"])
    with ts:
        st.subheader("Resolver modelo")
        sl=st.selectbox("Solver",list(SOLVER_OPTIONS.keys()),index=0,help="HiGHS recomendado para LP/MIP.")
        if st.button("Resolver modelo",type="primary"):
            try:
                m=build_model(spec); sname,solver=solver_get(sl); res=solver.solve(m)
                try: ov=pyo.value(m.OBJ)
                except Exception: ov=None
                spec["results"]={"solver_name":sname,"termination_condition":str(res.solver.termination_condition),"status":str(res.solver.status),"objective_value":ov}
                st.session_state["solved_model_object"]=m; st.success("Modelo resuelto correctamente.")
            except Exception as e: st.error(f"Error: {e}"); st.stop()
        r=spec.get("results"); m=st.session_state.get("solved_model_object")
        if not r or not m: st.info("Aún no has resuelto el modelo.")
        else:
            c1,c2,c3=st.columns(3)
            with c1: kpi("Solver",r.get("solver_name",""))
            with c2: kpi("Status",r["status"])
            with c3: kpi("Termination",r["termination_condition"])
            kpi("Valor óptimo","No disponible" if r.get("objective_value") is None else f"{r['objective_value']:,.6f}")

    with tv:
        r=spec.get("results"); m=st.session_state.get("solved_model_object")
        if not r or not m: st.info("Primero resuelve el modelo.")
        else:
            sv,snz=st.tabs(["Seleccionar variable","Variables no nulas"])
            with sv:
                st.subheader("Solución por variable"); sel=st.selectbox("Variable",list(spec["variables"].keys()))
                df=var_df(m,sel,spec["variables"][sel],spec["indices"]); st.dataframe(df,use_container_width=True,hide_index=True)
                st.download_button("Descargar CSV",df.to_csv(index=False).encode(),f"{sel}_solucion.csv","text/csv")
            with snz:
                st.subheader("Variables no nulas"); fd=all_vars_df(m,spec)
                nz=fd[fd["value"].abs()>1e-9].reset_index(drop=True) if not fd.empty else fd
                st.info("No hay variables no nulas.") if nz.empty else st.dataframe(nz,use_container_width=True,hide_index=True)
                st.download_button("Descargar CSV",nz.to_csv(index=False).encode(),"variables_no_nulas.csv","text/csv")
