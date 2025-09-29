# 🎯 Decision App

Aplicação Streamlit para apoiar na seleção de candidatos mais bem ranqueados para cada vaga.
Desenvolvido por Alexandre, Gabriel, Matheus e Pedro como trabalho de conclusão de curso da Pós-Graduação da FIAP.

## 🚀 Como funciona
- Faz download da base de dados (`df_join.parquet`) hospedada no GitHub Releases.
- Carrega o modelo treinado (`modelo.pkl`).
- Permite escolher uma vaga e retorna os 10 candidatos mais bem classificados.
- Mostra os scores previstos em formato de barra visual.

## 🛠️ Rodando localmente

1. Clone o repositório:
   ```bash
   git clone https://github.com/pedrolunardia/modelodecision.git
   cd modelodecision
