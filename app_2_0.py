# app_2_0.py
# Streamlit – EBAC Módulo 19 (versão consolidada)

import io
import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ------------- Configuração básica -------------
st.set_page_config(page_title="Análise", layout="wide")


# ------------- Utils -------------
@st.cache_data(show_spinner=False)
def load_data(path_or_buffer: str) -> pd.DataFrame:
    """
    Lê CSV de forma robusta (tenta ',' depois ';').
    """
    if not path_or_buffer:
        raise FileNotFoundError("Caminho do arquivo CSV não informado.")
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception:
        df = pd.read_csv(path_or_buffer, sep=";")
    return df


@st.cache_data(show_spinner=False)
def to_excel_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    """
    Gera um único .xlsx com várias abas.
    Fallback para openpyxl se xlsxwriter não estiver instalado.
    """
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for sheet, df in dfs.items():
                df.to_excel(writer, index=False, sheet_name=sheet[:31])
        return output.getvalue()
    except ModuleNotFoundError:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet, df in dfs.items():
                df.to_excel(writer, index=False, sheet_name=sheet[:31])
        return output.getvalue()


def numeric_series(s: pd.Series) -> pd.Series:
    """Coage para numérico com NaN em não-numéricos."""
    return pd.to_numeric(s, errors="coerce")


def pick_default_target(df: pd.DataFrame) -> Optional[str]:
    """Escolhe uma coluna-alvo categórica padrão."""
    for cand in ["Revenue", "y", "Response", "Target"]:
        if cand in df.columns:
            return cand
    # senão, escolhe a primeira categórica
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category") or df[c].nunique() <= 10]
    return cat_cols[0] if cat_cols else None


# ------------- Sidebar (entrada de dados) -------------
st.sidebar.header("Entrada de dados")

uploaded = st.sidebar.file_uploader("Carregar CSV", type=["csv"])

default_path = "/mnt/data/online_shoppers_intention.csv"
path_hint = st.sidebar.text_input("Ou informe o caminho do CSV", value=default_path if os.path.exists(default_path) else "")

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif path_hint:
    df = load_data(path_hint)

if df is None:
    st.info("Carregue um CSV (ou use o caminho padrão se disponível).")
    st.stop()


# ------------- Preparação inicial -------------
st.subheader("Prévia dos dados")
st.dataframe(df.head())

# coluna alvo default
target_col = pick_default_target(df)
cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category") or df[c].nunique() <= 50]
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

st.sidebar.header("Configurações de análise")
col_alvo = st.sidebar.selectbox("Coluna categórica para análise", options=cat_cols or df.columns.tolist(), index=(cat_cols.index(target_col) if target_col in cat_cols else 0))

# filtro simples (opcional) para até 3 colunas comuns do dataset
with st.sidebar.expander("Filtros (opcionais)", expanded=False):
    # filtros populares do dataset online_shoppers_intention
    if "Month" in df.columns:
        sel_month = st.multiselect("Month", sorted(df["Month"].dropna().unique().tolist()))
    else:
        sel_month = []
    if "VisitorType" in df.columns:
        sel_vtype = st.multiselect("VisitorType", sorted(df["VisitorType"].dropna().unique().tolist()))
    else:
        sel_vtype = []
    if "Weekend" in df.columns:
        sel_weekend = st.multiselect("Weekend", sorted(df["Weekend"].dropna().unique().tolist()))
    else:
        sel_weekend = []

graph_type = st.sidebar.radio("Tipo de gráfico", ("Barras", "Pizza"))
metric_mode = st.sidebar.radio("O que mostrar", ("Contagem", "Somar uma métrica"))
metric_col = None
if metric_mode == "Somar uma métrica":
    metric_col = st.sidebar.selectbox("Métrica numérica", options=num_cols or df.columns.tolist())

# aplica filtros
df_filt = df.copy()
if sel_month:
    df_filt = df_filt[df_filt.get("Month").isin(sel_month)]
if sel_vtype:
    df_filt = df_filt[df_filt.get("VisitorType").isin(sel_vtype)]
if sel_weekend:
    df_filt = df_filt[df_filt.get("Weekend").isin(sel_weekend)]

# ------------- Métricas/resumos -------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Linhas (brutos)", len(df))
with col2:
    st.metric("Linhas (filtrados)", len(df_filt))
with col3:
    st.metric("Colunas", df.shape[1])

# ------------- Gráficos (brutos vs filtrados) -------------
st.markdown("## Visualização")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# função que prepara a série (contagem ou soma de métrica)
def make_series(_df: pd.DataFrame) -> pd.Series:
    if metric_mode == "Contagem":
        s = _df[col_alvo].value_counts(dropna=False)
    else:
        if metric_col is None:
            st.warning("Selecione uma métrica numérica.")
            return pd.Series(dtype=float)
        s = (
            _df.assign(_m=numeric_series(_df[metric_col]))
              .groupby(col_alvo)["_m"]
              .sum()
              .dropna()
        )
    # ordena por índice para consistência
    return s.sort_index()

s_raw = make_series(df)
s_filt = make_series(df_filt)

if graph_type == "Pizza":
    # --- Pizza: usar Series.plot.pie em cada eixo ---
    s_raw.plot.pie(autopct="%1.1f%%", startangle=90, counterclock=False,
                   labels=s_raw.index.astype(str), ax=axes[0], ylabel="")
    axes[0].set_title("Dados brutos")

    s_filt.plot.pie(autopct="%1.1f%%", startangle=90, counterclock=False,
                    labels=s_filt.index.astype(str), ax=axes[1], ylabel="")
    axes[1].set_title("Dados filtrados")

else:
    # --- Barras horizontais para facilitar leitura ---
    s_raw.plot.barh(ax=axes[0])
    axes[0].set_title("Dados brutos")
    axes[0].set_xlabel("Valor" if metric_mode == "Somar uma métrica" else "Contagem")
    # anotações
    for i, v in enumerate(s_raw.values):
        axes[0].text(v, i, f" {v:.0f}" if isinstance(v, (int, float, np.number)) else f" {v}", va="center")

    s_filt.plot.barh(ax=axes[1])
    axes[1].set_title("Dados filtrados")
    axes[1].set_xlabel("Valor" if metric_mode == "Somar uma métrica" else "Contagem")
    for i, v in enumerate(s_filt.values):
        axes[1].text(v, i, f" {v:.0f}" if isinstance(v, (int, float, np.number)) else f" {v}", va="center")

plt.tight_layout()
st.pyplot(fig)

# ------------- Download Excel -------------
st.markdown("## Exportar para Excel")
dfs_export = {
    "Dados_brutos": df,
    "Dados_filtrados": df_filt,
    f"Resumo_{col_alvo}": make_series(df).reset_index().rename(columns={"index": col_alvo, 0: "value"})
}

if metric_mode == "Somar uma métrica" and metric_col:
    dfs_export[f"Soma_por_{col_alvo}_{metric_col}_FILTRADO"] = (
        df_filt.assign(_m=numeric_series(df_filt[metric_col]))
              .groupby(col_alvo)["_m"]
              .sum()
              .reset_index()
              .rename(columns={"_m": f"sum_{metric_col}"})
    )
else:
    dfs_export[f"Contagem_por_{col_alvo}_FILTRADO"] = (
        df_filt[col_alvo].value_counts(dropna=False).rename("count").reset_index().rename(columns={"index": col_alvo})
    )

xlsx_bytes = to_excel_bytes(dfs_export)

st.download_button(
    label="Baixar .xlsx",
    data=xlsx_bytes,
    file_name="analise_mod19.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ------------- Rodapé -------------
st.caption(" EBAC – Streamlit")
