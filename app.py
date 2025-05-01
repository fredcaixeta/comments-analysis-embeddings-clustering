import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

import json

import os
import sys
import requests

from typing import List, Dict

import numpy as np
from tqdm import tqdm

MISTRAL = os.getenv("MISTRAL_AI")

# Configuração da página
st.set_page_config(page_title="Análise de Comentários", layout="wide")

# Título
st.title("🔍 Análise de Comentários com Similaridade por Embeddings")

# --------------------------- --------------------------- ---------------------------
# --------------------------- START Treating Embedding ---------------------------
# --------------------------- --------------------------- ---------------------------
def carregar_comentarios_de_txt(filename: str = "comentarios.txt") -> List[str]:
    """Carrega os comentários de um arquivo de texto e os retorna em uma lista."""
    comentarios_carregados = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for linha in f:
                comentarios_carregados.append(linha.strip())
        print(f"Comentários carregados com sucesso do arquivo '{filename}'")
    except FileNotFoundError:
        print(f"Arquivo '{filename}' não encontrado.")
    except IOError as e:
        print(f"Erro ao ler o arquivo '{filename}': {e}")
    return comentarios_carregados

def carregar_embeddings_do_json(nome_arquivo) -> List[List[float]]:
    """
    Carrega embeddings de um arquivo JSON no formato especificado.

    Args:
        nome_arquivo (str): O caminho para o arquivo JSON.

    Returns:
        list or None: Uma lista de listas de floats (os embeddings),
                     ou None se ocorrer um erro ao carregar o arquivo.
    """
    try:
        with open(nome_arquivo, 'r') as f:
            data = json.load(f)
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            else:
                print(f"Formato inválido no arquivo '{nome_arquivo}'. Esperava uma lista na chave 'embedding'.")
                return None
    except FileNotFoundError:
        print(f"Arquivo '{nome_arquivo}' não encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON do arquivo '{nome_arquivo}'.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo '{nome_arquivo}': {e}")
        return None


all_comments = carregar_comentarios_de_txt()
embeddings_carregados = carregar_embeddings_do_json(r"app_duckdb\comments_1221_mistral.json")


# 1. Verifica e filtra embeddings inconsistentes
valid_embeddings = []
valid_comments = []
invalid_indices = []

for idx, (comment, emb) in tqdm(enumerate(zip(all_comments, embeddings_carregados)), total=len(all_comments)):
    if isinstance(emb, (list, np.ndarray)) and len(emb) > 0:  # Verifica se é um embedding válido
        valid_embeddings.append(emb)
        valid_comments.append(comment)
    else:
        invalid_indices.append(idx)

# print(f"\nEmbeddings válidos: {len(valid_embeddings)}")
# print(f"Embeddings inválidos: {len(invalid_indices)}")
# print(f"Exemplo de índices inválidos: {invalid_indices[:5]}")

# 2. Padroniza as dimensões (opcional - preenche com zeros se necessário)
max_dim = max(len(emb) for emb in valid_embeddings)
embeddings_padded = []
for emb in valid_embeddings:
    if len(emb) < max_dim:
        # Preenche com zeros se for menor que a dimensão máxima
        padded = np.pad(emb, (0, max_dim - len(emb)), 'constant')
        embeddings_padded.append(padded)
    else:
        embeddings_padded.append(emb)

# 3. Converte para numpy array
embeddings_array = np.array(embeddings_padded)
print(f"\nShape final do array: {embeddings_array.shape}")

# Filtra apenas os embeddings válidos (não vazios e com mesma dimensão)
reference_dim = len(embeddings_array[0])  # Assume primeiro como referência
filtered_data = [
    (comment, emb) 
    for comment, emb in zip(all_comments, embeddings_array) 
    if isinstance(emb, (list, np.ndarray)) and len(emb) == reference_dim
]

filtered_comments, filtered_embeddings = zip(*filtered_data)
embeddings_array = np.array(filtered_embeddings)

# --------------------------- --------------------------- ---------------------------
# ----------------------------- END Treating Embedding ---------------------------
# --------------------------- --------------------------- ---------------------------

def create_embeddings(comments: List[str]) -> List[List[float]]:
    """
    Cria embeddings para uma lista de comentários usando a API da Mistral.

    Args:
        comments: Uma lista de strings (comentários).

    Returns:
        Uma lista de listas de floats, onde cada inner list é o embedding de um comentário.
        Retorna uma lista vazia em caso de erro.
    """
    list_of_embeddings = []
    for comment in comments:
        url = "https://api.mistral.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL}"
        }
        payload = {
            "model": "mistral-embed",
            "input": comment
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Levanta uma exceção para códigos de status de erro

            embeddings_data = response.json()
            if embeddings_data and "data" in embeddings_data and len(embeddings_data["data"]) > 0 and "embedding" in embeddings_data["data"][0]:
                list_of_embeddings.append(embeddings_data["data"][0]["embedding"])
            else:
                print(f"Erro ao processar a resposta para o comentário: '{comment}'. Formato inesperado.")
                print(f"Resposta completa: {json.dumps(embeddings_data, indent=2)}")

        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição para o comentário: '{comment}': {e}")
            if response is not None:
                print(f"Código de status: {response.status_code}")
                print(f"Texto da resposta: {response.text}")
                
    return list_of_embeddings


# ---------------- Interface com o Usuário ----------------

# Visualização dos clusters
st.header("🌌 Visualização dos Clusters")

# Adicionando controle deslizante para número de clusters
n_clusters = st.slider(
    "Selecione o número de clusters:",
    min_value=2,
    max_value=10,
    value=5,
    help="Ajuste a quantidade de grupos para agrupar os comentários",
    key="cluster_slider"  # Adicionando key única para evitar conflitos
)

# 1. Clusterização inicial
@st.cache_data
def perform_clustering(_embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(_embeddings)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(_embeddings)
    return cluster_labels, embeddings_2d, pca

with st.spinner("Agrupando comentários..."):
    cluster_labels, embeddings_2d, pca = perform_clustering(embeddings_array, n_clusters)

# 2.1. Verifique os shapes primeiro
print(f"Shape embeddings_2d: {embeddings_2d.shape}")
print(f"Length cluster_labels: {len(cluster_labels)}")
print(f"Length all_comments: {len(all_comments)}")

# 2.2. Garanta que todos tenham o mesmo tamanho
min_length = min(len(embeddings_2d), len(cluster_labels), len(all_comments))

# 2.3. Crie o DataFrame com os dados truncados
plot_df = pd.DataFrame({
    'x': embeddings_2d[:min_length, 0],
    'y': embeddings_2d[:min_length, 1],
    'cluster': cluster_labels[:min_length],
    'comment': [c[:100] + '...' for c in all_comments[:min_length]]
})

fig = px.scatter(
    plot_df, x='x', y='y', color='cluster',
    hover_data=['comment'], title='Distribuição dos Clusters'
)
st.plotly_chart(fig, use_container_width=True)

# 3. Busca por similaridade
st.header("🔎 Buscar Comentários Similares")
search_term = st.text_input("Digite uma frase para encontrar comentários similares:", 
                           "Gostei muito do conteúdo!")

@st.cache_data(ttl=36000, show_spinner="Calculando similaridades...")  # Cache por 1 hora
def process_search(_search_term, _embeddings_array, _pca, _cluster_labels, _all_comments):
    """Processa a busca com cache para evitar chamadas repetidas à API"""
    # 1. Gera embedding para a busca
    search_embedding = create_embeddings([_search_term])
    
    # 2. Converte para array numpy e ajusta dimensões
    search_embedding = np.array(search_embedding).reshape(1, -1)
    
    # 3. Calcula similaridades
    similarities = cosine_similarity(search_embedding, _embeddings_array)[0]
    
    # 4. Obtém os índices dos mais similares
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    # 5. Prepara dados para visualização
    search_point = _pca.transform(search_embedding)[0]
    
    return {
        'top_indices': top_indices,
        'similarities': similarities,
        'search_point': search_point,
        'search_embedding': search_embedding
    }
    
if st.button("Buscar") and search_term:
    with st.spinner("Procurando comentários similares..."):
        try:
            # Processa a busca (usando cache)
            results = process_search(search_term, embeddings_array, pca, cluster_labels, all_comments)
            
            # 6. Mostra resultados
            st.subheader("Comentários mais similares:")
            for idx in results['top_indices']:
                with st.expander(f"Similaridade: {results['similarities'][idx]:.3f} (Cluster {cluster_labels[idx]})"):
                    st.write(all_comments[idx])
                    st.progress(float(results['similarities'][idx]))
                    
            # 7. Adiciona ponto da busca na visualização
            fig.add_trace(px.scatter(
                x=[results['search_point'][0]],
                y=[results['search_point'][1]], 
                color_discrete_sequence=['red'],
                symbol=['Busca'],
                size=[10]
            ).data[0])
            st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Erro durante a busca: {str(e)}")