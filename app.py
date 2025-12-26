# -*- coding: utf-8 -*-
"""
Fake News Detection - Dashboard Streamlit
==========================================
Aplicação interativa para visualização dos resultados do projeto de Text Mining.
Inclui Demo Interativo com modelos reais treinados.

Autor: Projeto Text Mining ISCTE
Cadeira: Text Mining
"""

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Imports para modelos (carregados condicionalmente)
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="🔍 Deteção de Fake News - Text Mining ISCTE",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS DOS FICHEIROS EXPORTADOS
# =============================================================================

# Caminho base para os ficheiros exportados
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
EXPORTS_PATH = os.path.join(BASE_PATH, 'exports')

@st.cache_data
def load_json(filepath):
    """Carrega um ficheiro JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Erro ao carregar {filepath}: {e}")
        return None

@st.cache_data
def load_all_metrics():
    """Carrega todas as métricas dos ficheiros exportados"""
    metrics = {}
    
    # Métricas dos modelos
    metrics['model_metrics'] = load_json(os.path.join(EXPORTS_PATH, 'metrics', 'model_metrics.json'))
    metrics['cv_results'] = load_json(os.path.join(EXPORTS_PATH, 'metrics', 'cv_results.json'))
    metrics['dataset_stats'] = load_json(os.path.join(EXPORTS_PATH, 'metrics', 'dataset_stats.json'))
    metrics['confusion_matrices'] = load_json(os.path.join(EXPORTS_PATH, 'metrics', 'confusion_matrices.json'))
    metrics['roc_curves'] = load_json(os.path.join(EXPORTS_PATH, 'metrics', 'roc_curves.json'))
    
    # Estatísticas de texto
    metrics['text_stats'] = load_json(os.path.join(EXPORTS_PATH, 'data', 'text_stats.json'))
    metrics['top_words'] = load_json(os.path.join(EXPORTS_PATH, 'data', 'top_words.json'))
    
    # Clustering
    metrics['clustering'] = load_json(os.path.join(EXPORTS_PATH, 'clustering', 'clustering_metrics.json'))
    
    # IR
    metrics['ir'] = load_json(os.path.join(EXPORTS_PATH, 'ir', 'ir_metrics.json'))
    
    # Topics
    metrics['topic_names'] = load_json(os.path.join(EXPORTS_PATH, 'topics', 'topic_names.json'))
    
    # LSTM training history
    metrics['lstm_history'] = load_json(os.path.join(EXPORTS_PATH, 'models', 'lstm_training_history.json'))
    
    return metrics

@st.cache_data
def load_topics_csv():
    """Carrega os ficheiros CSV de tópicos"""
    topics = {}
    try:
        topics['lda_fake'] = pd.read_csv(os.path.join(EXPORTS_PATH, 'topics', 'topics_lda_fake.csv'))
        topics['lda_true'] = pd.read_csv(os.path.join(EXPORTS_PATH, 'topics', 'topics_lda_true.csv'))
        topics['nmf_fake'] = pd.read_csv(os.path.join(EXPORTS_PATH, 'topics', 'topics_nmf_fake.csv'))
        topics['nmf_true'] = pd.read_csv(os.path.join(EXPORTS_PATH, 'topics', 'topics_nmf_true.csv'))
    except Exception as e:
        st.warning(f"Erro ao carregar tópicos: {e}")
    return topics

@st.cache_data
def load_clustering_csv():
    """Carrega o CSV de seleção de K"""
    try:
        return pd.read_csv(os.path.join(EXPORTS_PATH, 'clustering', 'k_selection_results.csv'))
    except Exception as e:
        st.warning(f"Erro ao carregar k_selection_results.csv: {e}")
        return None

# Carregar todas as métricas no início
ALL_METRICS = load_all_metrics()
TOPICS_DATA = load_topics_csv()
K_SELECTION_DATA = load_clustering_csv()

# =============================================================================
# FUNÇÕES DE PREPROCESSAMENTO
# =============================================================================

def clean_text(text):
    """Limpa o texto removendo caracteres especiais"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Apenas letras
    text = re.sub(r'\s+', ' ', text).strip()  # Remove espaços extras
    return text


def preprocess_text(text):
    """Pipeline completo de preprocessamento"""
    # Limpeza básica
    text = clean_text(text)
    
    # Tokenização simples
    tokens = text.split()
    
    # Remover palavras muito curtas
    tokens = [t for t in tokens if len(t) > 2]
    
    return ' '.join(tokens)


# =============================================================================
# FUNÇÕES DE CARREGAMENTO E TREINO DE MODELOS
# =============================================================================

@st.cache_data
def load_data():
    """Carrega os datasets do projeto"""
    try:
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        fake_path = os.path.join(base_path, 'Fake.csv')
        true_path = os.path.join(base_path, 'True.csv')
        
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        
        # Labels consistentes com o notebook: Fake=0, True=1
        fake_df['label'] = 0  # Fake = 0
        true_df['label'] = 1  # True = 1
        
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Combinar título e texto
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['content_clean'] = df['content'].apply(preprocess_text)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None


@st.cache_resource
def train_models():
    """Treina os modelos de classificação"""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    
    df = load_data()
    if df is None:
        return None
    
    # Preparar dados
    X = df['content_clean']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vetorizadores
    vectorizer_bow = CountVectorizer(max_features=5000)
    vectorizer_tfidf = TfidfVectorizer(max_features=5000)
    
    X_train_bow = vectorizer_bow.fit_transform(X_train)
    X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
    
    # Modelos
    lr_bow = LogisticRegression(max_iter=1000, random_state=42)
    lr_tfidf = LogisticRegression(max_iter=1000, random_state=42)
    nb_bow = MultinomialNB()
    nb_tfidf = MultinomialNB()
    
    # Treinar
    lr_bow.fit(X_train_bow, y_train)
    lr_tfidf.fit(X_train_tfidf, y_train)
    nb_bow.fit(X_train_bow, y_train)
    nb_tfidf.fit(X_train_tfidf, y_train)
    
    return {
        'vectorizer_bow': vectorizer_bow,
        'vectorizer_tfidf': vectorizer_tfidf,
        'lr_bow': lr_bow,
        'lr_tfidf': lr_tfidf,
        'nb_bow': nb_bow,
        'nb_tfidf': nb_tfidf
    }


# =============================================================================
# SIDEBAR - NAVEGAÇÃO
# =============================================================================

st.sidebar.title("📰 Deteção de Fake News")
st.sidebar.markdown("---")

pages = [
    "📋 Resumo",
    "🏠 Visão Geral",
    "📊 Análise Exploratória",
    "🔤 Pré-processamento",
    "📈 Topic Modeling",
    "🤖 Classificação ML",
    "🔍 Information Retrieval",
    "📦 Clustering",
    "🧠 Deep Learning",
    "🎯 Demo Interativo"
]

page = st.sidebar.radio("Navegação", pages)

st.sidebar.markdown("---")
st.sidebar.info("""
**Projeto Text Mining**  
ISCTE 2025/2026  
  
Dataset: Kaggle Fake News  
~44.000 artigos
""")


# =============================================================================
# PÁGINA: VISÃO GERAL
# =============================================================================

if page == "🏠 Visão Geral":
    st.title("🔍 Deteção de Fake News")
    st.markdown("### Projeto de Text Mining - ISCTE 2025/2026")
    
    st.markdown("""
    Este projeto implementa um sistema completo de **deteção de fake news** utilizando 
    técnicas de **Text Mining** e **Machine Learning**. O objetivo é classificar notícias 
    como verdadeiras ou falsas com base no seu conteúdo textual.
    """)
    
    # Obter métricas dos ficheiros carregados
    model_metrics = ALL_METRICS.get('model_metrics', {})
    dataset_stats = ALL_METRICS.get('dataset_stats', {})
    ir_metrics = ALL_METRICS.get('ir', {})
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    total_samples = dataset_stats.get('total_samples', 44445)
    lstm_f1 = model_metrics.get('lstm', {}).get('F1-Score', 0.9982) * 100
    lstm_auc = model_metrics.get('lstm', {}).get('AUC', 0.9999) * 100
    
    with col1:
        st.metric(
            "📰 Total Artigos",
            f"{total_samples:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "🏆 Melhor F1-Score",
            f"{lstm_f1:.2f}%",
            delta="LSTM"
        )
    
    with col3:
        st.metric(
            "📈 Melhor AUC",
            f"{lstm_auc:.2f}%",
            delta="LSTM"
        )
    
    with col4:
        mean_sim = ir_metrics.get('mean_similarity', 0.0445) * 100
        st.metric(
            "🔍 Similaridade Média",
            f"{mean_sim:.2f}%",
            delta="IR System"
        )
    
    st.markdown("---")
    
    # Distribuição do Dataset
    st.subheader("📊 Distribuição do Dataset")
    
    col1, col2 = st.columns([1, 1])
    
    # Calcular distribuição (aproximadamente 50/50 no dataset original)
    fake_count = dataset_stats.get('fake_count', 23481)
    true_count = dataset_stats.get('true_count', 21417)
    
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=['Fake News', 'True News'],
            values=[fake_count, true_count],
            hole=0.4,
            marker_colors=['#FF6B6B', '#4ECDC4']
        )])
        fig.update_layout(
            title="Distribuição de Classes",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Comparação de métricas dos modelos
        models_data = []
        model_names = {
            'lstm': 'LSTM',
            'logistic_regression_bow': 'LR + BoW',
            'logistic_regression_tfidf': 'LR + TF-IDF',
            'naive_bayes_bow': 'NB + BoW',
            'naive_bayes_tfidf': 'NB + TF-IDF'
        }
        colors = ['#28a745', '#667eea', '#764ba2', '#f093fb', '#f5576c']
        
        for key, name in model_names.items():
            if key in model_metrics:
                f1_key = 'F1-Score' if key == 'lstm' else 'f1'
                f1 = model_metrics[key].get(f1_key, 0) * 100
                models_data.append({'name': name, 'f1': f1})
        
        if models_data:
            fig = go.Figure(data=[
                go.Bar(
                    x=[m['name'] for m in models_data],
                    y=[m['f1'] for m in models_data],
                    marker_color=colors[:len(models_data)],
                    text=[f"{m['f1']:.2f}%" for m in models_data],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="F1-Score por Modelo",
                yaxis_title="F1-Score (%)",
                height=350,
                yaxis_range=[80, 102]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Pipeline do Projeto
    st.subheader("🔄 Pipeline do Projeto")
    
    pipeline_steps = [
        ("1️⃣ Carregamento", "Importação e união dos datasets"),
        ("2️⃣ Exploração", "EDA, estatísticas, distribuições"),
        ("3️⃣ Preprocessamento", "Limpeza, tokenização, lematização"),
        ("4️⃣ Vetorização", "BoW, TF-IDF (5000 features)"),
        ("5️⃣ Topic Modeling", "LDA, NMF por classe"),
        ("6️⃣ Classificação", "NB, LR (99.4% F1)"),
        ("7️⃣ Clustering", "K-Means, análise de grupos"),
        ("8️⃣ IR", "Cosine Similarity (88.8% P@5)"),
        ("9️⃣ Deep Learning", "LSTM (99.91% F1) 🏆")
    ]
    
    cols = st.columns(3)
    for i, (step, desc) in enumerate(pipeline_steps):
        with cols[i % 3]:
            # Destacar a fase 9
            if "Deep Learning" in step:
                bg_color = "linear-gradient(135deg, #28a745 0%, #20c997 100%)"
            else:
                bg_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
            st.markdown(f"""
            <div style="background: {bg_color};
                        padding: 15px; border-radius: 10px; margin: 5px 0; color: white;">
                <strong>{step}</strong><br>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PÁGINA: RESUMO
# =============================================================================

elif page == "📋 Resumo":
    st.title("📋 Resumo")
    
    st.markdown("""
    O presente trabalho desenvolve e avalia uma **solução completa de Text Mining** para deteção 
    automática de notícias falsas, utilizando um corpus público constituído por notícias rotuladas 
    como **FAKE** e **TRUE**. O projeto inicia-se com uma **análise exploratória do dataset**, 
    incluindo estatísticas descritivas e verificação de qualidade, seguida de uma fase de preparação 
    do corpus que contempla limpeza textual e deduplicação para reduzir redundância e melhorar a 
    robustez experimental.
    
    Posteriormente, é implementado um **pipeline de pré-processamento** (normalização, tokenização, 
    remoção de stopwords e lematização) e procede-se à **extração de características**, combinando 
    representações vetoriais clássicas — **Bag-of-Words** e **TF-IDF** — com um conjunto de features 
    estilísticas e estruturais (por exemplo, comprimento do texto, pontuação, maiúsculas e diversidade 
    lexical), devidamente normalizadas.
    
    A solução integra **múltiplas técnicas** estudadas na unidade curricular: modelos supervisionados 
    (**Naive Bayes** e **Regressão Logística**), modelos não supervisionados e interpretativos 
    (**LDA** e **NMF** para topic modeling, e **K-Means** para clustering), bem como uma componente 
    de **Recuperação de Informação** baseada em similaridade por cosseno, avaliada com Precision@K. 
    Como extensão, é ainda treinado um modelo de **Deep Learning (LSTM)** para comparação com os 
    métodos clássicos.
    
    A avaliação recorre a **métricas apropriadas** (Accuracy, Precision, Recall, F1 e AUC-ROC), 
    permitindo uma análise comparativa e discussão crítica dos resultados. O trabalho culmina na 
    **exportação de modelos e artefactos experimentais**, promovendo reprodutibilidade e facilitando 
    integração em aplicações de demonstração.
    """)


# =============================================================================
# PÁGINA: ANÁLISE EXPLORATÓRIA
# =============================================================================

elif page == "📊 Análise Exploratória":
    st.title("📊 Análise Exploratória de Dados")
    
    st.markdown("""
    Análise detalhada do dataset de Fake News, incluindo estatísticas descritivas,
    distribuições e características textuais.
    """)
    
    # Obter dados dos ficheiros
    dataset_stats = ALL_METRICS.get('dataset_stats', {})
    text_stats = ALL_METRICS.get('text_stats', {})
    top_words = ALL_METRICS.get('top_words', {})
    
    # Estatísticas do Dataset
    st.subheader("📈 Estatísticas do Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    total = dataset_stats.get('total_samples', 44445)
    fake_count = dataset_stats.get('fake_count', 23481)
    true_count = dataset_stats.get('true_count', 21417)
    
    with col1:
        st.markdown("### Contagem de Artigos")
        fake_pct = (fake_count / total * 100) if total > 0 else 52.3
        true_pct = (true_count / total * 100) if total > 0 else 47.7
        data = {
            'Classe': ['Fake News', 'True News', 'Total'],
            'Quantidade': [fake_count, true_count, total],
            'Percentagem': [f"{fake_pct:.1f}%", f"{true_pct:.1f}%", "100%"]
        }
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Média de Palavras")
        avg_words_fake = text_stats.get('fake', {}).get('mean_words', 424)
        avg_words_true = text_stats.get('true', {}).get('mean_words', 384)
        fig = go.Figure(data=[
            go.Bar(
                x=['Fake News', 'True News'],
                y=[avg_words_fake, avg_words_true],
                marker_color=['#FF6B6B', '#4ECDC4'],
                text=[avg_words_fake, avg_words_true],
                textposition='outside'
            )
        ])
        fig.update_layout(
            yaxis_title="Palavras por Artigo",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### Média de Caracteres")
        avg_chars_fake = text_stats.get('fake', {}).get('mean_length', 2553)
        avg_chars_true = text_stats.get('true', {}).get('mean_length', 2378)
        fig = go.Figure(data=[
            go.Bar(
                x=['Fake News', 'True News'],
                y=[avg_chars_fake, avg_chars_true],
                marker_color=['#FF6B6B', '#4ECDC4'],
                text=[avg_chars_fake, avg_chars_true],
                textposition='outside'
            )
        ])
        fig.update_layout(
            yaxis_title="Caracteres por Artigo",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Palavras mais frequentes
    st.subheader("☁️ Palavras Mais Frequentes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Fake News - Top Words")
        fake_words_data = top_words.get('fake_raw', [])[:15]
        if fake_words_data:
            words = [w[0] for w in fake_words_data]
            counts = [w[1] for w in fake_words_data]
            fig = go.Figure(data=[go.Bar(
                x=words,
                y=counts,
                marker_color='#FF6B6B'
            )])
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🟢 True News - Top Words")
        true_words_data = top_words.get('true_raw', [])[:15]
        if true_words_data:
            words = [w[0] for w in true_words_data]
            counts = [w[1] for w in true_words_data]
            fig = go.Figure(data=[go.Bar(
                x=words,
                y=counts,
                marker_color='#4ECDC4'
            )])
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.subheader("💡 Principais Insights")
    
    text_stats = ALL_METRICS.get('text_stats', {})
    avg_fake = text_stats.get('fake', {}).get('mean_words', 424)
    avg_true = text_stats.get('true', {}).get('mean_words', 384)
    
    insights = [
        f"📌 **Dataset com ~44.000 artigos**: Boa quantidade de dados para treino",
        f"📌 **Fake news mais longas**: Média de {avg_fake} palavras vs {avg_true} em notícias verdadeiras",
        "📌 **'Reuters' como indicador**: Termo comum em notícias verdadeiras (fonte credível)",
        "📌 **Referências políticas**: Ambas classes focam em política americana (2016-2017)",
        "📌 **Fake news mais informais**: Linguagem mais emocional e sensacionalista"
    ]
    
    for insight in insights:
        st.markdown(insight)


# =============================================================================
# PÁGINA: PRÉ-PROCESSAMENTO
# =============================================================================

elif page == "🔤 Pré-processamento":
    st.title("🔤 Pré-processamento de Texto")
    
    st.markdown("""
    Pipeline de preprocessamento aplicado aos textos para preparação do modelo.
    """)
    
    # Pipeline Visual
    st.subheader("🔄 Pipeline de Preprocessamento")
    
    pipeline_data = {
        'Etapa': [
            '1. Lowercase',
            '2. Remoção URLs',
            '3. Remoção HTML',
            '4. Remoção Pontuação',
            '5. Tokenização',
            '6. Stopwords Removal',
            '7. Lematização'
        ],
        'Descrição': [
            'Converter para minúsculas',
            'Remover http://, https://, www.',
            'Remover tags HTML (<p>, <div>, etc.)',
            'Remover caracteres especiais e números',
            'Dividir texto em tokens/palavras',
            'Remover palavras comuns (the, is, at, etc.)',
            'Reduzir palavras à forma base (running → run)'
        ],
        'Técnica': [
            'str.lower()',
            'Regex: http\\S+|www\\S+',
            'Regex: <.*?>',
            'Regex: [^a-zA-Z\\s]',
            'str.split() / NLTK',
            'NLTK stopwords (english)',
            'WordNetLemmatizer'
        ]
    }
    
    st.dataframe(pd.DataFrame(pipeline_data), hide_index=True, use_container_width=True)
    
    # Exemplo Interativo
    st.subheader("🧪 Teste de Preprocessamento")
    
    example_text = st.text_area(
        "Digite um texto para testar o preprocessamento:",
        value="BREAKING: President Trump said on Twitter that the media is FAKE NEWS!!! https://twitter.com/example",
        height=100
    )
    
    if st.button("🔄 Processar Texto"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Texto Original:**")
            st.code(example_text, language=None)
        
        with col2:
            processed = preprocess_text(example_text)
            st.markdown("**✅ Texto Processado:**")
            st.code(processed, language=None)
    
    st.markdown("---")
    
    # Vetorização
    st.subheader("📊 Técnicas de Vetorização")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Bag of Words (BoW)
        
        - Conta frequência de cada palavra
        - Matriz esparsa de contagens
        - Simples mas eficaz
        - `max_features=5000`
        
        ```python
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_features=5000)
        X_bow = vectorizer.fit_transform(texts)
        ```
        """)
    
    with col2:
        st.markdown("""
        ### TF-IDF
        
        - Term Frequency × Inverse Document Frequency
        - Penaliza palavras muito comuns
        - Valoriza palavras discriminativas
        - `max_features=5000`
        
        ```python
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(texts)
        ```
        """)
    
    # Estatísticas de Vocabulário
    st.subheader("📈 Estatísticas do Vocabulário")
    
    vocab_stats = {
        'Métrica': [
            'Vocabulário Total (antes)',
            'Vocabulário após BoW',
            'Vocabulário após TF-IDF',
            'Tokens Únicos (média por doc)',
            'Stopwords Removidas'
        ],
        'Valor': ['~150,000', '5,000', '5,000', '~180', '~179 (NLTK English)']
    }
    
    st.dataframe(pd.DataFrame(vocab_stats), hide_index=True, use_container_width=True)


# =============================================================================
# PÁGINA: TOPIC MODELING
# =============================================================================

elif page == "📈 Topic Modeling":
    st.title("📈 Topic Modeling")
    
    st.markdown("""
    Análise de tópicos utilizando **LDA (Latent Dirichlet Allocation)** e **NMF (Non-negative Matrix Factorization)** 
    para identificar os principais temas presentes nas notícias falsas e verdadeiras.
    """)
    
    # Obter dados de tópicos
    topic_names = ALL_METRICS.get('topic_names', {})
    
    # Seleção do método
    method = st.radio("Selecione o método:", ["LDA", "NMF"], horizontal=True)
    
    # Tópicos por Classe
    st.subheader("🔍 Principais Tópicos por Classe")
    
    tab1, tab2 = st.tabs(["🔴 Fake News", "🟢 True News"])
    
    method_lower = method.lower()
    
    with tab1:
        st.markdown(f"### Top 5 Tópicos em Fake News ({method})")
        
        # Carregar tópicos do CSV
        topics_key = f'{method_lower}_fake'
        topics_df = TOPICS_DATA.get(topics_key)
        names_dict = topic_names.get(method_lower, {}).get('fake', {})
        
        if topics_df is not None and not topics_df.empty:
            for idx, row in topics_df.iterrows():
                topic_num = str(idx)
                topic_name = names_dict.get(topic_num, f"Tópico {idx+1}")
                words = row.get('Top Palavras', '')
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FF6B6B 0%, #ee5a5a 100%);
                            padding: 15px; border-radius: 10px; margin: 10px 0; color: white;">
                    <strong>{topic_name}</strong><br>
                    <small>{words}</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f"### Top 5 Tópicos em True News ({method})")
        
        topics_key = f'{method_lower}_true'
        topics_df = TOPICS_DATA.get(topics_key)
        names_dict = topic_names.get(method_lower, {}).get('true', {})
        
        if topics_df is not None and not topics_df.empty:
            for idx, row in topics_df.iterrows():
                topic_num = str(idx)
                topic_name = names_dict.get(topic_num, f"Tópico {idx+1}")
                words = row.get('Top Palavras', '')
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4ECDC4 0%, #3db9b0 100%);
                            padding: 15px; border-radius: 10px; margin: 10px 0; color: white;">
                    <strong>{topic_name}</strong><br>
                    <small>{words}</small>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparação de Tópicos
    st.subheader("📊 Análise Comparativa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔴 Características Fake News
        
        - **Linguagem mais informal e emocional**
        - Foco em personalidades (Clinton, Hillary, Obama)
        - Termos de conspiração ("deep state", "wire")
        - Menos referências a fontes credíveis
        - Narrativas sensacionalistas
        """)
    
    with col2:
        st.markdown("""
        ### 🟢 Características True News
        
        - **Linguagem formal/jornalística**
        - Referência a "Reuters" (fonte credível)
        - Termos institucionais (Congress, Senate)
        - Foco em política externa e factos
        - Citações e dados verificáveis
        """)
    
    # Variância Explicada
    st.subheader("📈 Variância Explicada pelo LSA")
    
    components = list(range(1, 11))
    variance = [0.15, 0.25, 0.33, 0.39, 0.44, 0.48, 0.52, 0.55, 0.58, 0.60]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=components,
        y=variance,
        mode='lines+markers',
        name='Variância Cumulativa',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="50% Variância")
    fig.update_layout(
        title="Variância Explicada por Número de Componentes (LSA)",
        xaxis_title="Número de Componentes",
        yaxis_title="Variância Explicada Cumulativa",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PÁGINA: CLASSIFICAÇÃO ML
# =============================================================================

elif page == "🤖 Classificação ML":
    st.title("🤖 Modelos de Classificação")
    
    st.markdown("""
    Comparação dos modelos de Machine Learning treinados para classificação de fake news.
    """)
    
    # Obter métricas dos ficheiros
    model_metrics = ALL_METRICS.get('model_metrics', {})
    cv_results = ALL_METRICS.get('cv_results', {})
    confusion_matrices = ALL_METRICS.get('confusion_matrices', {})
    
    # Mapeamento de nomes
    model_display_names = {
        'logistic_regression_bow': 'Logistic Regression + BoW',
        'logistic_regression_tfidf': 'Logistic Regression + TF-IDF',
        'naive_bayes_bow': 'Naive Bayes + BoW',
        'naive_bayes_tfidf': 'Naive Bayes + TF-IDF',
        'lstm': 'LSTM Bidirecional'
    }
    
    # Tabela Comparativa
    st.subheader("📊 Comparação de Modelos")
    
    results_data = []
    for key, display_name in model_display_names.items():
        if key in model_metrics:
            m = model_metrics[key]
            # LSTM tem chaves diferentes
            if key == 'lstm':
                results_data.append({
                    'Modelo': display_name,
                    'Accuracy': f"{m.get('Accuracy', 0)*100:.2f}%",
                    'Precision': f"{m.get('Precision', 0)*100:.2f}%",
                    'Recall': f"{m.get('Recall', 0)*100:.2f}%",
                    'F1-Score': f"{m.get('F1-Score', 0)*100:.2f}%",
                    'AUC': f"{m.get('AUC', 0)*100:.2f}%"
                })
            else:
                results_data.append({
                    'Modelo': display_name,
                    'Accuracy': f"{m.get('accuracy', 0)*100:.2f}%",
                    'Precision': f"{m.get('precision', 0)*100:.2f}%",
                    'Recall': f"{m.get('recall', 0)*100:.2f}%",
                    'F1-Score': f"{m.get('f1', 0)*100:.2f}%",
                    'AUC': f"{m.get('auc', 0)*100:.2f}%"
                })
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Gráficos de Métricas
    st.subheader("📈 Visualização das Métricas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras F1-Score
        models_for_chart = []
        f1_scores = []
        colors = ['#28a745', '#667eea', '#764ba2', '#f093fb', '#f5576c']
        
        for key, display_name in model_display_names.items():
            if key in model_metrics:
                m = model_metrics[key]
                f1_key = 'F1-Score' if key == 'lstm' else 'f1'
                models_for_chart.append(display_name.replace(' + ', '\n'))
                f1_scores.append(m.get(f1_key, 0) * 100)
        
        fig = go.Figure(data=[
            go.Bar(
                x=models_for_chart,
                y=f1_scores,
                marker_color=colors[:len(models_for_chart)],
                text=[f'{s:.2f}%' for s in f1_scores],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="F1-Score por Modelo",
            yaxis_title="F1-Score (%)",
            height=400,
            yaxis_range=[80, 102]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AUC Comparison
        models_for_chart = []
        aucs = []
        
        for key, display_name in model_display_names.items():
            if key in model_metrics:
                m = model_metrics[key]
                auc_key = 'AUC' if key == 'lstm' else 'auc'
                models_for_chart.append(display_name.replace(' + ', '\n'))
                aucs.append(m.get(auc_key, 0) * 100)
        
        fig = go.Figure(data=[
            go.Bar(
                y=models_for_chart,
                x=aucs,
                orientation='h',
                marker_color=colors[:len(models_for_chart)],
                text=[f'{a:.2f}%' for a in aucs],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="AUC-ROC por Modelo",
            xaxis_title="AUC (%)",
            height=400,
            xaxis_range=[90, 102]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cross-Validation
    st.subheader("🔄 Cross-Validation (5-Fold)")
    
    if cv_results:
        col1, col2 = st.columns(2)
        
        cv_display_names = {
            'LR_BoW': 'Logistic Regression + BoW',
            'LR_TF-IDF': 'Logistic Regression + TF-IDF',
            'NB_BoW': 'Naive Bayes + BoW',
            'NB_TF-IDF': 'Naive Bayes + TF-IDF'
        }
        
        with col1:
            cv_data = []
            for key, display_name in cv_display_names.items():
                if key in cv_results:
                    f1_values = cv_results[key].get('f1', [])
                    if f1_values:
                        mean_f1 = np.mean(f1_values)
                        std_f1 = np.std(f1_values)
                        cv_data.append({
                            'Modelo': display_name,
                            'F1 Médio': f"{mean_f1*100:.2f}%",
                            'Desvio Padrão': f"±{std_f1*100:.2f}%"
                        })
            
            if cv_data:
                cv_df = pd.DataFrame(cv_data)
                st.dataframe(cv_df, hide_index=True, use_container_width=True)
        
        with col2:
            models_cv = []
            means = []
            stds = []
            
            for key, display_name in cv_display_names.items():
                if key in cv_results:
                    f1_values = cv_results[key].get('f1', [])
                    if f1_values:
                        models_cv.append(display_name.replace(' + ', '\n'))
                        means.append(np.mean(f1_values) * 100)
                        stds.append(np.std(f1_values) * 100)
            
            if models_cv:
                fig = go.Figure(data=[
                    go.Bar(
                        x=models_cv,
                        y=means,
                        error_y=dict(type='data', array=stds, visible=True),
                        marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c'][:len(models_cv)]
                    )
                ])
                fig.update_layout(
                    title="Cross-Validation Results",
                    yaxis_title="F1-Score (%)",
                    yaxis_range=[85, 102],
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Conclusões
    st.subheader("🏆 Conclusões")
    
    # Determinar melhor modelo
    best_model = "LSTM Bidirecional"
    best_f1 = model_metrics.get('lstm', {}).get('F1-Score', 0.9982) * 100
    best_auc = model_metrics.get('lstm', {}).get('AUC', 0.9999) * 100
    
    st.success(f"""
    **Melhor Modelo: {best_model}**
    
    - ✅ F1-Score: {best_f1:.2f}% (melhor desempenho geral)
    - ✅ AUC: {best_auc:.2f}% (excelente separação de classes)
    - ✅ Deep Learning supera modelos tradicionais
    - ✅ Logistic Regression + BoW é o melhor modelo tradicional (99.40% F1)
    """)


# =============================================================================
# PÁGINA: INFORMATION RETRIEVAL
# =============================================================================

elif page == "🔍 Information Retrieval":
    st.title("🔍 Information Retrieval")
    
    st.markdown("""
    Sistema de recuperação de informação baseado em similaridade de cosseno para encontrar notícias similares.
    """)
    
    # Obter métricas IR dos ficheiros
    ir_metrics = ALL_METRICS.get('ir', {})
    
    # Métricas IR
    st.subheader("📊 Métricas de Similaridade")
    
    col1, col2, col3, col4 = st.columns(4)
    
    mean_sim = ir_metrics.get('mean_similarity', 0.0445)
    fake_fake = ir_metrics.get('fake_fake_mean', 0.0488)
    true_true = ir_metrics.get('true_true_mean', 0.0527)
    fake_true = ir_metrics.get('fake_true_mean', 0.0382)
    
    with col1:
        st.metric("Similaridade Média", f"{mean_sim:.4f}")
    with col2:
        st.metric("FAKE ↔ FAKE", f"{fake_fake:.4f}")
    with col3:
        st.metric("TRUE ↔ TRUE", f"{true_true:.4f}")
    with col4:
        st.metric("FAKE ↔ TRUE", f"{fake_true:.4f}")
    
    st.markdown("---")
    
    # Análise de Similaridade
    st.subheader("📈 Análise de Similaridade entre Classes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sim_data = {
            'Comparação': ['FAKE ↔ FAKE', 'TRUE ↔ TRUE', 'FAKE ↔ TRUE'],
            'Similaridade Média': [fake_fake, true_true, fake_true]
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=sim_data['Comparação'],
                y=sim_data['Similaridade Média'],
                marker_color=['#FF6B6B', '#4ECDC4', '#95a5a6'],
                text=[f'{v:.4f}' for v in sim_data['Similaridade Média']],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Similaridade Média (Cosine)",
            yaxis_title="Cosine Similarity",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        ### 💡 Interpretação
        
        - **FAKE-FAKE ({fake_fake:.4f})**: Fake news têm similaridade entre si
        - **TRUE-TRUE ({true_true:.4f})**: Notícias verdadeiras também são similares
        - **FAKE-TRUE ({fake_true:.4f})**: Menor similaridade entre classes diferentes
        
        ✅ **Conclusão**: A similaridade intra-classe é superior à inter-classe!
        
        Isto indica que o sistema de IR consegue distinguir padrões 
        diferentes entre notícias falsas e verdadeiras com base no 
        conteúdo textual.
        """)
    
    # Como funciona
    st.subheader("⚙️ Como Funciona o Sistema IR")
    
    st.markdown("""
    ```
    1. Texto Query → Preprocessamento → Vetorização (TF-IDF)
    2. Calcular Similaridade Cosseno com todos os documentos
    3. Ranking dos documentos por similaridade
    4. Retornar Top-K documentos mais similares
    ```
    """)
    
    # Exemplo visual
    st.subheader("🔎 Exemplo de Busca")
    
    query_example = st.text_input(
        "Digite uma query de exemplo:",
        value="Trump announces new policy on immigration"
    )
    
    if st.button("🔍 Simular Busca"):
        st.markdown("### Resultados Simulados:")
        
        fake_results = [
            ("Trump DESTROYS opposition with SHOCKING announcement!", 0.82, "FAKE"),
            ("You won't BELIEVE what Trump said about immigrants!", 0.78, "FAKE"),
            ("Media HIDES truth about Trump's immigration plan!", 0.74, "FAKE")
        ]
        
        true_results = [
            ("Trump administration unveils immigration policy changes - Reuters", 0.85, "TRUE"),
            ("White House announces new border security measures", 0.81, "TRUE"),
            ("Congress debates proposed immigration legislation", 0.76, "TRUE")
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🟢 True News")
            for title, sim, label in true_results:
                st.markdown(f"- **{title}**\n  - Similaridade: {sim:.2f}")
        
        with col2:
            st.markdown("#### 🔴 Fake News")
            for title, sim, label in fake_results:
                st.markdown(f"- **{title}**\n  - Similaridade: {sim:.2f}")


# =============================================================================
# PÁGINA: CLUSTERING
# =============================================================================

elif page == "📦 Clustering":
    st.title("📦 Análise de Clustering")
    
    st.markdown("""
    Análise de agrupamento não supervisionado utilizando K-Means.
    """)
    
    # Obter dados de clustering
    clustering_metrics = ALL_METRICS.get('clustering', {})
    k_selection_df = K_SELECTION_DATA
    
    # Métricas de Clustering
    st.subheader("📊 Métricas por Número de Clusters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Silhouette Score
        if k_selection_df is not None and not k_selection_df.empty:
            fig = go.Figure(data=[
                go.Scatter(
                    x=k_selection_df['k'],
                    y=k_selection_df['Silhouette Score'],
                    mode='lines+markers',
                    marker=dict(size=12, color='#667eea'),
                    line=dict(width=3, color='#667eea')
                )
            ])
            fig.update_layout(
                title="Silhouette Score vs K",
                xaxis_title="Número de Clusters (K)",
                yaxis_title="Silhouette Score",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ARI
        if k_selection_df is not None and not k_selection_df.empty:
            fig = go.Figure(data=[
                go.Scatter(
                    x=k_selection_df['k'],
                    y=k_selection_df['Adjusted Rand Index'],
                    mode='lines+markers',
                    marker=dict(size=12, color='#f5576c'),
                    line=dict(width=3, color='#f5576c')
                )
            ])
            fig.update_layout(
                title="Adjusted Rand Index vs K",
                xaxis_title="Número de Clusters (K)",
                yaxis_title="ARI",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Melhor K
    st.subheader("🏆 Análise do Melhor K")
    
    silhouette_k2 = clustering_metrics.get('silhouette_k2', 0.6449)
    ari_k2 = clustering_metrics.get('ari_k2', 0.0028)
    n_clusters = clustering_metrics.get('n_clusters', 2)
    cluster_sizes = clustering_metrics.get('cluster_sizes', [16984, 5791])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if k_selection_df is not None and not k_selection_df.empty:
            display_df = k_selection_df[['k', 'Silhouette Score', 'Adjusted Rand Index']].copy()
            display_df.columns = ['K', 'Silhouette', 'ARI']
            display_df['Silhouette'] = display_df['Silhouette'].apply(lambda x: f"{x:.4f}")
            display_df['ARI'] = display_df['ARI'].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.info(f"""
        **K=2 apresenta melhor Silhouette Score ({silhouette_k2:.4f})**
        
        - ✅ Alta coesão intra-cluster
        - ⚠️ ARI baixo ({ari_k2:.4f}) indica que clusters não correspondem exatamente às classes
        - 📊 Tamanhos dos clusters: {cluster_sizes[0]:,} e {cluster_sizes[1]:,} documentos
        
        **Interpretação**: Os documentos agrupam-se por tópicos (política, 
        internacional, etc.) e não apenas por veracidade.
        """)
    
    # Distribuição dos Clusters
    st.subheader("📊 Distribuição dos Clusters")
    
    if cluster_sizes:
        fig = go.Figure(data=[go.Pie(
            labels=[f'Cluster {i}' for i in range(len(cluster_sizes))],
            values=cluster_sizes,
            hole=0.4,
            marker_colors=['#667eea', '#f5576c', '#4ECDC4', '#f093fb', '#28a745'][:len(cluster_sizes)]
        )])
        fig.update_layout(
            title=f"Distribuição dos {n_clusters} Clusters",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PÁGINA: DEMO INTERATIVO
# =============================================================================

elif page == "🎯 Demo Interativo":
    st.title("🎯 Demo Interativo - Classificador de Fake News")
    
    st.markdown("""
    **Teste o classificador com os modelos reais treinados no projeto!**
    
    Os modelos foram treinados no dataset Kaggle Fake News com ~44.000 artigos.
    """)
    
    # Verificar se modelo LSTM existe na pasta exports/models
    models_path = os.path.join(EXPORTS_PATH, 'models')
    lstm_model_path = os.path.join(models_path, 'lstm_model.keras')
    tokenizer_path = os.path.join(models_path, 'tokenizer.pickle')
    lstm_available = os.path.exists(lstm_model_path) and os.path.exists(tokenizer_path)
    
    # Seleção de modelo PRIMEIRO (antes de carregar)
    st.subheader("1️⃣ Selecione o Modelo")
    
    model_options = [
        "🧠 LSTM Deep Learning (Melhor: F1=99.82%)" if lstm_available else "🧠 LSTM (Treinar no notebook primeiro)",
        "Logistic Regression + BoW (F1=99.4%)",
        "Logistic Regression + TF-IDF (F1=93.4%)",
        "Naive Bayes + BoW (F1=93.7%)",
        "Naive Bayes + TF-IDF (F1=88.2%)"
    ]
    
    # Se LSTM disponível, mostrar mensagem
    if lstm_available:
        st.success("🎉 Modelo LSTM disponível! Treinado com 99.82% F1-Score.")
    else:
        st.info("💡 Para usar o modelo LSTM, execute primeiro a Fase 9 do notebook para treinar e guardar o modelo.")
    
    model_choice = st.selectbox(
        "Modelo de classificação:",
        model_options,
        index=0,
        help="Selecione o modelo que pretende usar para classificar a notícia"
    )
    
    st.markdown("---")
    
    # Área de texto
    st.subheader("2️⃣ Digite ou Cole o Texto")
    
    example_texts = {
        "📝 Texto Personalizado": "",
        "🔴 Exemplo Fake News": """BREAKING: Scientists discover that vaccines cause autism! 
The government has been hiding this information for years. 
Share this before they delete it! The mainstream media won't tell you the truth!""",
        "🟢 Exemplo True News": """WASHINGTON (Reuters) - The Senate passed a bipartisan infrastructure bill 
on Tuesday with a vote of 69-30. The legislation includes $550 billion in new federal investments 
in roads, bridges, and broadband internet. President Biden praised the vote as a historic step."""
    }
    
    selected_example = st.radio(
        "Escolha um exemplo ou escreva o seu próprio texto:",
        list(example_texts.keys()),
        horizontal=True,
        index=0
    )
    
    user_text = st.text_area(
        "Texto da notícia:",
        value=example_texts[selected_example],
        height=200,
        placeholder="Cole aqui o texto de uma notícia para classificar..."
    )
    
    st.markdown("---")
    
    # Botão de classificação
    st.subheader("3️⃣ Classificar")
    
    if st.button("🔍 Classificar Notícia", type="primary", use_container_width=True):
        if len(user_text.strip()) < 10:
            st.warning("⚠️ Por favor, insira um texto mais longo para análise (mínimo 10 caracteres).")
        elif "LSTM" in model_choice and not lstm_available:
            st.error("❌ O modelo LSTM ainda não foi treinado. Execute a Fase 9 do notebook primeiro.")
        else:
            # Verificar se é LSTM ou modelo tradicional
            is_lstm = "LSTM" in model_choice and lstm_available
            
            if is_lstm:
                # Classificação com LSTM
                with st.spinner("🔄 Carregando modelo LSTM e classificando..."):
                    try:
                        # Importar TensorFlow
                        import tensorflow as tf
                        from tensorflow.keras.preprocessing.sequence import pad_sequences
                        
                        # Carregar modelo e tokenizer da pasta exports/models
                        models_path = os.path.join(EXPORTS_PATH, 'models')
                        model_lstm = tf.keras.models.load_model(os.path.join(models_path, 'lstm_model.keras'))
                        with open(os.path.join(models_path, 'tokenizer.pickle'), 'rb') as handle:
                            tokenizer = pickle.load(handle)
                        with open(os.path.join(models_path, 'lstm_params.pickle'), 'rb') as handle:
                            lstm_params = pickle.load(handle)
                        
                        # Preprocessar texto
                        processed_text = preprocess_text(user_text)
                        
                        # Tokenizar e fazer padding
                        sequence = tokenizer.texts_to_sequences([processed_text])
                        padded = pad_sequences(sequence, maxlen=lstm_params['max_sequence_length'], 
                                               padding='post', truncating='post')
                        
                        # Prever
                        # O modelo LSTM foi treinado com: Fake=0, True=1
                        # pred_prob representa a probabilidade de ser TRUE (classe 1)
                        pred_prob = model_lstm.predict(padded, verbose=0)[0][0]
                        is_true = pred_prob > 0.5
                        prob_true = pred_prob
                        prob_fake = 1 - pred_prob
                        
                        # Mostrar resultado
                        st.markdown("---")
                        st.subheader("📊 Resultado da Classificação (LSTM Deep Learning)")
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            if is_true:
                                st.success("## 🟢 NOTÍCIA VERDADEIRA")
                                confidence = prob_true * 100
                            else:
                                st.error("## 🔴 FAKE NEWS DETECTADA!")
                                confidence = prob_fake * 100
                            
                            st.metric("Confiança", f"{confidence:.1f}%")
                        
                        # Gráfico de probabilidades
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['True News', 'Fake News'],
                                y=[prob_true*100, prob_fake*100],
                                marker_color=['#4ECDC4', '#FF6B6B'],
                                text=[f'{prob_true*100:.1f}%', f'{prob_fake*100:.1f}%'],
                                textposition='outside'
                            )
                        ])
                        fig.update_layout(
                            title="Probabilidades por Classe",
                            yaxis_title="Probabilidade (%)",
                            yaxis_range=[0, 105],
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detalhes do processamento
                        with st.expander("🔧 Ver Detalhes do Processamento"):
                            st.markdown("**Texto Original:**")
                            st.code(user_text[:500] + "..." if len(user_text) > 500 else user_text)
                            
                            st.markdown("**Texto Processado:**")
                            st.code(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)
                            
                            st.markdown(f"**Modelo Usado:** LSTM Bidirecional (Deep Learning)")
                            st.markdown(f"**Vocabulário:** {lstm_params['max_vocab_size']} palavras")
                            st.markdown(f"**Comprimento Sequência:** {lstm_params['max_sequence_length']} tokens")
                            st.markdown(f"**Probabilidade Raw:** {pred_prob:.4f}")
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao carregar o modelo LSTM: {e}")
                        st.info("💡 Certifique-se de que executou a Fase 9 do notebook para treinar e guardar o modelo.")
            else:
                # Classificação com modelos tradicionais (ML)
                with st.spinner("🔄 Carregando modelos e classificando..."):
                    models = train_models()
                
                if models is None:
                    st.error("❌ Erro ao carregar os modelos. Verifique se os ficheiros Fake.csv e True.csv existem na pasta do projeto.")
                else:
                    # Preprocessar
                    processed_text = preprocess_text(user_text)
                    
                    # Selecionar modelo e vetorizador
                    if "BoW" in model_choice:
                        vectorizer = models['vectorizer_bow']
                        if "Logistic" in model_choice:
                            clf = models['lr_bow']
                        else:
                            clf = models['nb_bow']
                    else:
                        vectorizer = models['vectorizer_tfidf']
                        if "Logistic" in model_choice:
                            clf = models['lr_tfidf']
                        else:
                            clf = models['nb_tfidf']
                    
                    # Vetorizar e prever
                    X_new = vectorizer.transform([processed_text])
                    prediction = clf.predict(X_new)[0]
                    proba = clf.predict_proba(X_new)[0]
                    # proba[0] = prob de Fake (classe 0), proba[1] = prob de True (classe 1)
                    
                    # Mostrar resultado
                    st.markdown("---")
                    st.subheader("📊 Resultado da Classificação")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        if prediction == 0:  # FAKE (classe 0)
                            st.error("## 🔴 FAKE NEWS DETECTADA!")
                            confidence = proba[0] * 100
                        else:  # TRUE (classe 1)
                            st.success("## 🟢 NOTÍCIA VERDADEIRA")
                            confidence = proba[1] * 100
                        
                        st.metric("Confiança", f"{confidence:.1f}%")
                    
                    # Gráfico de probabilidades
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['True News', 'Fake News'],
                            y=[proba[1]*100, proba[0]*100],
                            marker_color=['#4ECDC4', '#FF6B6B'],
                            text=[f'{proba[1]*100:.1f}%', f'{proba[0]*100:.1f}%'],
                            textposition='outside'
                        )
                    ])
                    fig.update_layout(
                        title="Probabilidades por Classe",
                        yaxis_title="Probabilidade (%)",
                        yaxis_range=[0, 105],
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detalhes do processamento
                    with st.expander("🔧 Ver Detalhes do Processamento"):
                        st.markdown("**Texto Original:**")
                        st.code(user_text[:500] + "..." if len(user_text) > 500 else user_text)
                        
                        st.markdown("**Texto Processado:**")
                        st.code(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)
                        
                        st.markdown(f"**Modelo Usado:** {model_choice}")
                        st.markdown(f"**Número de Features:** 5000")


# =============================================================================
# PÁGINA: DEEP LEARNING
# =============================================================================

elif page == "🧠 Deep Learning":
    st.title("🧠 Deep Learning - Resultados LSTM")
    
    st.markdown("""
    Esta página apresenta os **resultados reais** do modelo LSTM (Long Short-Term Memory) 
    implementado na Fase 9 do projeto.
    """)
    
    # Obter métricas
    model_metrics = ALL_METRICS.get('model_metrics', {})
    lstm_metrics = model_metrics.get('lstm', {})
    lstm_history = ALL_METRICS.get('lstm_history', {})
    confusion_matrices = ALL_METRICS.get('confusion_matrices', {})
    
    lstm_f1 = lstm_metrics.get('F1-Score', 0.9982) * 100
    lstm_auc = lstm_metrics.get('AUC', 0.9999)
    
    # Resultados do LSTM
    st.success(f"""
    🏆 **O modelo LSTM atingiu {lstm_f1:.2f}% F1-Score**, superando todos os modelos tradicionais!
    """)
    
    # Métricas principais
    st.subheader("📊 Métricas do Modelo LSTM")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    acc = lstm_metrics.get('Accuracy', 0.9983) * 100
    prec = lstm_metrics.get('Precision', 0.9986) * 100
    rec = lstm_metrics.get('Recall', 0.9979) * 100
    
    with col1:
        st.metric("Accuracy", f"{acc:.2f}%")
    with col2:
        st.metric("Precision", f"{prec:.2f}%")
    with col3:
        st.metric("Recall", f"{rec:.2f}%")
    with col4:
        st.metric("F1-Score", f"{lstm_f1:.2f}%")
    with col5:
        st.metric("AUC-ROC", f"{lstm_auc:.4f}")
    
    st.caption("*Comparação com o melhor modelo tradicional (LR + BoW)")
    
    st.markdown("---")
    
    # Arquitetura do Modelo
    st.subheader("🏗️ Arquitetura do Modelo LSTM")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Camadas do Modelo:**
        
        ```
        Embedding(10000, 128)
            ↓
        SpatialDropout1D(0.2)
            ↓
        Bidirectional LSTM(64)
            ↓
        Dropout(0.3)
            ↓
        LSTM(32)
            ↓
        Dropout(0.3)
            ↓
        Dense(64, relu)
            ↓
        Dropout(0.3)
            ↓
        Dense(32, relu)
            ↓
        Dense(1, sigmoid)
        ```
        """)
    
    with col2:
        st.markdown("**Parâmetros:**")
        n_epochs = lstm_history.get('epochs', 7)
        params_df = pd.DataFrame({
            'Parâmetro': ['Vocabulário', 'Sequência Máx.', 'Embedding Dim', 'LSTM Units', 'Dropout', 'Épocas', 'Batch Size'],
            'Valor': ['10,000', '300 tokens', '128', '[64, 32]', '0.3', str(n_epochs), '64']
        })
        st.dataframe(params_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("📈 Matriz de Confusão")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cm_data = confusion_matrices.get('lstm', [[4498, 6], [9, 4233]])
        fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Previsto TRUE', 'Previsto FAKE'],
            y=['Real TRUE', 'Real FAKE'],
            text=[[f'{cm_data[i][j]}' for j in range(2)] for i in range(2)],
            texttemplate='%{text}',
            textfont={'size': 20},
            colorscale='Blues',
            showscale=False
        ))
        fig.update_layout(
            title='Matriz de Confusão - LSTM',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        tn, fp = cm_data[0]
        fn, tp = cm_data[1]
        total = tn + fp + fn + tp
        total_errors = fp + fn
        error_rate = (total_errors / total) * 100
        
        st.markdown(f"""
        **Análise dos Erros:**
        
        | Tipo | Quantidade | Descrição |
        |------|------------|-----------|
        | ✅ True Negative | {tn:,} | TRUE corretamente classificado |
        | ✅ True Positive | {tp:,} | FAKE corretamente classificado |
        | ❌ False Positive | {fp} | TRUE classificado como FAKE |
        | ❌ False Negative | {fn} | FAKE classificado como TRUE |
        
        **Total de erros: apenas {total_errors} em {total:,} amostras!**
        
        Taxa de erro: **{error_rate:.2f}%**
        """)
    
    st.markdown("---")
    
    # Curvas de Treino
    st.subheader("📉 Evolução do Treino")
    
    # Dados do histórico de treino real
    train_acc = lstm_history.get('accuracy', [])
    val_acc = lstm_history.get('val_accuracy', [])
    train_loss = lstm_history.get('loss', [])
    val_loss = lstm_history.get('val_loss', [])
    n_epochs = len(train_acc) if train_acc else 7
    epochs = list(range(1, n_epochs + 1))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if train_acc and val_acc:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=[a*100 for a in train_acc], name='Treino', line=dict(color='#667eea')))
            fig.add_trace(go.Scatter(x=epochs, y=[a*100 for a in val_acc], name='Validação', line=dict(color='#f5576c')))
            fig.update_layout(
                title='Accuracy por Época',
                xaxis_title='Época',
                yaxis_title='Accuracy (%)',
                height=350,
                yaxis_range=[65, 102]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if train_loss and val_loss:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Treino', line=dict(color='#667eea')))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validação', line=dict(color='#f5576c')))
            fig.update_layout(
                title='Loss por Época',
                xaxis_title='Época',
                yaxis_title='Loss',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Comparação Final
    st.subheader("🏆 Comparação: LSTM vs ML Tradicional")
    
    # Usar dados reais das métricas
    lr_bow_f1 = model_metrics.get('logistic_regression_bow', {}).get('f1', 0.994) * 100
    lr_tfidf_f1 = model_metrics.get('logistic_regression_tfidf', {}).get('f1', 0.934) * 100
    nb_bow_f1 = model_metrics.get('naive_bayes_bow', {}).get('f1', 0.937) * 100
    
    comparison_data = {
        'Modelo': ['LSTM Bidirecional', 'Logistic Regression + BoW', 'Logistic Regression + TF-IDF', 'Naive Bayes + BoW'],
        'F1-Score': [f'{lstm_f1:.2f}%', f'{lr_bow_f1:.2f}%', f'{lr_tfidf_f1:.2f}%', f'{nb_bow_f1:.2f}%'],
        'AUC-ROC': [f'{lstm_auc:.4f}', 
                    f"{model_metrics.get('logistic_regression_bow', {}).get('auc', 0.998):.4f}",
                    f"{model_metrics.get('logistic_regression_tfidf', {}).get('auc', 0.975):.4f}",
                    f"{model_metrics.get('naive_bayes_bow', {}).get('auc', 0.973):.4f}"],
        'Tipo': ['Deep Learning', 'ML Tradicional', 'ML Tradicional', 'ML Tradicional']
    }
    
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, hide_index=True, use_container_width=True)
    
    # Gráfico de barras
    models = ['LSTM', 'LR + BoW', 'LR + TF-IDF', 'NB + BoW']
    f1_scores = [lstm_f1, lr_bow_f1, lr_tfidf_f1, nb_bow_f1]
    colors = ['#28a745', '#667eea', '#764ba2', '#f093fb']
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=f1_scores,
            marker_color=colors,
            text=[f'{s:.2f}%' for s in f1_scores],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title='F1-Score: LSTM vs Modelos Tradicionais',
        yaxis_title='F1-Score (%)',
        height=400,
        yaxis_range=[88, 102]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Trade-offs
    st.subheader("⚖️ Trade-offs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Vantagens do LSTM
        
        - **Melhor performance** (melhor F1-Score)
        - **Menos erros** de classificação
        - **Aprende features automaticamente**
        - **Captura contexto sequencial**
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Desvantagens do LSTM
        
        - **Treino lento** (~60 min vs 1 seg)
        - **Requer TensorFlow** (~400MB)
        - **Difícil interpretar**
        - **GPU recomendada**
        """)
    
    # Recomendação Final
    st.subheader("💡 Recomendação Final")
    
    st.info("""
    **Para produção:** Usar **Logistic Regression + BoW** (99.40% F1, treino em 1 segundo)
    
    **Para máxima precisão:** Usar **LSTM** (99.91% F1, mas requer ~60 min de treino)
    
    **Para investigação futura:** Explorar **BERT** ou **Transformers** para potencial melhoria adicional
    """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎓 <strong>Projeto de Text Mining</strong> - ISCTE 2025/2026</p>
    <p>Dataset: <a href="https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data">Kaggle Fake and Real News</a> (~44.000 artigos)</p>
    <p>Desenvolvido com ❤️ usando Streamlit, Plotly, Scikit-learn e TensorFlow</p>
</div>
""", unsafe_allow_html=True)
