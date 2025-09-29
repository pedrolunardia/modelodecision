# 🎯 Decision App

Aplicação Streamlit para apoiar na seleção de candidatos mais bem ranqueados para cada vaga.
Desenvolvido por Alexandre, Gabriel, Matheus e Pedro como trabalho de conclusão de curso da Pós-Graduação da FIAP.

##
Notebook com o desenvolvimento do modelo feito no Google Colab está salvo nos nossos releases: https://github.com/pedrolunardia/modelodecision/releases/tag/pipeline

## 🚀 Como funciona
- Faz download da base de dados (`df_join.parquet`) hospedada no GitHub Releases.
- Carrega o modelo treinado (`modelo.pkl`).
- Permite escolher uma vaga e retorna os 10 candidatos mais bem classificados.
- Mostra os scores previstos em formato de barra visual.

## ⚠️ Observação sobre deploy no Streamlit Cloud
Este projeto está publicado no [Streamlit Cloud](https://modelodecision-cfooyqqkd6vtzaaud8djag.streamlit.app/), 
mas por ser uma versão gratuita do serviço, a aplicação pode:

- Cair por falta de recursos (memória/CPU limitada a ~1 GB);
- Ficar "hibernando" após um período de inatividade (precisa clicar em "Yes, get this app back up!" para reativar);
- Apresentar lentidão no carregamento inicial (download da base de dados).

Isso é comportamento esperado do ambiente gratuito, **não um erro do código**.

---

## 💻 Como rodar localmente (recomendado)

1. Clone este repositório:
   ```bash
   git clone https://github.com/pedrolunardia/modelodecision.git
   cd modelodecision

2. Crie um ambiente virtual:
  python -m venv .venv
  source .venv/bin/activate   # Linux/Mac
  .venv\Scripts\activate      # Windows

3. Instale as dependências:
  pip install -r requirements.txt

4. Rode o app:
  streamlit run app.py

5. Abra no navegador o link indicado (geralmente http://localhost:8501).

---

