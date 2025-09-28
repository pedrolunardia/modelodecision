import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# =======================
# MODELO E BASE
# =======================
modelo = joblib.load("modelo.pkl")
df_join = pd.read_parquet("df_join.parquet")

# =======================
# UTIL: barra HTML (uma única linha, sem quebras)
# =======================
def bar_html(pct: float) -> str:
    pct = float(pct)
    width = int(pct * 100)
    # cores dinâmicas
    if pct >= 0.7:
        color = "#4CAF50"   # verde
    elif pct >= 0.4:
        color = "#FFC107"   # amarelo
    else:
        color = "#F44336"   # vermelho
    text_color = "white" if width > 15 else "black"
    return (
        f"<div style='background:#e5e7eb;border-radius:6px;height:24px;width:100%;position:relative;'>"
        f"<div style='width:{width}%;background:{color};height:24px;border-radius:6px;'></div>"
        f"<div style='position:absolute;top:0;left:0;width:100%;height:24px;"
        f"display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:{text_color};'>"
        f"{width:.0f}%</div></div>"
    )

# =======================
# UI
# =======================
st.title("🎯 Decision: te apoia nas melhores decisões")
st.write("Selecione uma vaga e visualize os candidatos mais bem ranqueados pela Decision!")

col_vaga = "titulo_vaga" if "titulo_vaga" in df_join.columns else "titulo"
vagas = df_join[col_vaga].dropna().unique().tolist()
vaga_escolhida = st.selectbox("Escolha uma vaga:", vagas)

if st.button("Encontrar perfil ideal"):
    cands_vaga = df_join.loc[df_join[col_vaga] == vaga_escolhida].copy()

    # ===== ENTRADA =====
    drop_cols = ["situacao_candidato", "target"]
    X = cands_vaga.drop(columns=[c for c in drop_cols if c in cands_vaga.columns], errors="ignore")

    # LabelEncode rápido (só para rodar)
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # Alinhar colunas com o modelo
    X_encoded = X_encoded.reindex(columns=modelo.feature_name_, fill_value=0)

    # ===== PREVISÃO =====
    probas = modelo.predict_proba(X_encoded)[:, 1]
    cands_vaga["Score previsto"] = probas

    # ===== RESULTADO =====
    col_id = "id_applicant" if "id_applicant" in cands_vaga.columns else "codigo"
    col_nome = "txt_candidato" if "txt_candidato" in cands_vaga.columns else "nome_x"

    df_resultado = (
        cands_vaga[[col_id, col_nome, "Score previsto"]]
        .sort_values("Score previsto", ascending=False)
        .head(10)
        .reset_index(drop=True)
        .rename(columns={col_id: "ID Candidato", col_nome: "Nome", "Score previsto": "Score"})
    )
    df_resultado.insert(0, "Ranking", range(1, len(df_resultado) + 1))

    # ===== TABELA HTML (sem to_html, sem quebras) =====
    # CSS leve para ocupar a largura e deixar bonito
    css = (
        "<style>"
        ".ranktbl{width:100%;border-collapse:collapse;}"
        ".ranktbl th,.ranktbl td{padding:10px 12px;border-bottom:1px solid #2a2a2a;vertical-align:middle;}"
        ".ranktbl th{text-align:left;background:#111318;position:sticky;top:0;}"
        ".center{text-align:center;white-space:nowrap;}"
        ".scorecell{min-width:220px;}"
        "</style>"
    )

    # Cabeçalho
    header = (
        "<tr>"
        "<th class='center'>Ranking</th>"
        "<th class='center'>ID Candidato</th>"
        "<th>Nome</th>"
        "<th class='scorecell'>Score</th>"
        "</tr>"
    )

    # Linhas
    rows = []
    for _, r in df_resultado.iterrows():
        rows.append(
            "<tr>"
            f"<td class='center'>{int(r['Ranking'])}</td>"
            f"<td class='center'>{r['ID Candidato']}</td>"
            f"<td>{r['Nome']}</td>"
            f"<td class='scorecell'>{bar_html(r['Score'])}</td>"
            "</tr>"
        )
    table_html = f"{css}<table class='ranktbl'>{header}{''.join(rows)}</table>"

    st.subheader(f"Melhores candidatos(as) para: {vaga_escolhida}")
    # IMPORTANTE: renderizar só com markdown; não use st.write(table_html) nem st.code(...)
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("Scores previstos pelo algorítimo da Decision (quanto maior, melhor).")