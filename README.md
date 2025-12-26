# 📰 Deteção de Fake News com Machine Learning e Deep Learning

## 📋 Sobre o Projeto

Este projeto foi desenvolvido no âmbito da unidade curricular de **Text Mining** do ISCTE - Instituto Universitário de Lisboa. O objetivo é criar um sistema de deteção automática de notícias falsas utilizando técnicas de Processamento de Linguagem Natural (NLP), Machine Learning e Deep Learning.

---

## 📁 Estrutura do Projeto

```
PROJETO_FAKE_NEWS/
│
├── 📓 fake_news_detection.ipynb    # Notebook principal com todo o trabalho
├── 🚀 app.py                        # Aplicação Streamlit (Dashboard + Demo)
├── 📊 Fake.csv                      # Dataset de notícias falsas (ISOT)
├── 📊 True.csv                      # Dataset de notícias verdadeiras (ISOT)
├── 📋 requirements_streamlit.txt    # Dependências para o Streamlit
│
└── 📂 exports/                      # Dados exportados para visualização
    ├── clustering/                  # Métricas e resultados de clustering
    ├── data/                        # Estatísticas de texto e palavras
    ├── ir/                          # Métricas de Information Retrieval
    ├── metrics/                     # Métricas dos modelos e matrizes de confusão
    ├── models/                      # Modelo LSTM e histórico de treino
    └── topics/                      # Tópicos LDA e NMF exportados
```

---

## 📓 Notebook Principal

O ficheiro **`fake_news_detection.ipynb`** contém **todo o trabalho desenvolvido**, organizado em 9 fases:

| Fase | Descrição |
|------|-----------|
| **Fase 1** | Exploração e Análise de Dados (EDA) |
| **Fase 2** | Pré-processamento de Texto |
| **Fase 3** | Feature Engineering (BoW, TF-IDF, Features Adicionais) |
| **Fase 4** | Topic Modeling (LDA e NMF) |
| **Fase 5** | Classificação Supervisionada (Naive Bayes, Logistic Regression) |
| **Fase 6** | Clustering (K-Means) |
| **Fase 7** | Information Retrieval (Similaridade de Cosseno) |
| **Fase 8** | Deep Learning (LSTM Bidirectional) |
| **Fase 9** | Avaliação Final e Comparação de Modelos |

Cada fase inclui:
- ✅ Introdução teórica e justificação das escolhas
- ✅ Código documentado e comentado
- ✅ Visualizações e gráficos
- ✅ Conclusões detalhadas

> **Nota:** O **resumo executivo** do trabalho encontra-se disponível na aplicação Streamlit, na página inicial do dashboard.

---

## 🚀 Aplicação Streamlit

A aplicação Streamlit (`app.py`) oferece uma interface interativa para explorar os resultados do projeto e **testar o modelo em tempo real**.

### Funcionalidades da aplicação:

| Página | Descrição |
|--------|-----------|
| **📊 Dashboard** | Visão geral do projeto, métricas principais e resumo executivo |
| **📈 Métricas dos Modelos** | Comparação detalhada de todos os modelos treinados |
| **📝 Topic Modeling** | Visualização dos tópicos LDA e NMF descobertos |
| **🔬 Demo Interativo** | **Testar o modelo com texto próprio** |

---

## 🔬 Demo Interativo

Para **testar o modelo de deteção de fake news**:

1. Executar a aplicação Streamlit com `streamlit run app.py`
2. Navegar até à aba **"🔬 Demo Interativo"**
3. Introduzir um texto de notícia (em inglês)
4. O modelo LSTM irá classificar o texto como **FAKE** ou **TRUE**
5. Visualize a probabilidade e confiança da previsão

---

## 📊 Dataset

O projeto utiliza o **ISOT Fake News Dataset** da Universidade de Victoria, que contém:
- **~21.000** notícias verdadeiras (fontes: Reuters, etc.)
- **~24.000** notícias falsas (fontes: sites identificados como não confiáveis)
- Período: **2015-2017** (principalmente eleições EUA 2016)

---

## 🏆 Resultados Principais

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| Logistic Regression (BoW) | **99.52%** | **99.56%** | 0.9988 |
| LSTM Bidirectional | 99.45% | 99.51% | **0.9997** |
| Naive Bayes (TF-IDF) | 93.64% | 94.23% | 0.9691 |

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **Pandas, NumPy** - Manipulação de dados
- **NLTK** - Processamento de linguagem natural
- **Scikit-learn** - Machine Learning
- **TensorFlow/Keras** - Deep Learning (LSTM)
- **Streamlit** - Interface web interativa
- **Matplotlib, Seaborn** - Visualizações

---

## 👥 Autores

Projeto desenvolvido para a UC de **Text Mining** - ISCTE

---

## 📄 Licença

Este projeto foi desenvolvido para fins académicos.
