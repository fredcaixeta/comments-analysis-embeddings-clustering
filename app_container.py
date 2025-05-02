import streamlit as st
import numpy as np

import plotly.express as px
import plotly.figure_factory as ff

from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

import pandas as pd
import json
import os
import requests

from tqdm import tqdm

from typing import List, Dict

MISTRAL = os.getenv("MISTRAL_AI")

# Configuração da página
st.set_page_config(page_title="Análise de Comentários", layout="wide")

# Título
st.title("🔍 Análise Hierárquica de Comentários")

# --------------------------- Funções ---------------------------
def similar_request_container(embedding_referencia: list, api_url: str = "http://localhost:5000/similar"):
    """Faz uma requisição POST para a API de similaridade de embeddings.

    Args:
        embedding_referencia: A lista de floats representando o embedding de referência.
        api_url: A URL da API para buscar embeddings similares (padrão: http://localhost:5000/similar).

    Returns:
        Um dicionário JSON contendo a resposta da API, ou None em caso de erro.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {'embedding': embedding_referencia}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Levanta uma exceção para códigos de status de erro (4xx ou 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição para a API: {e}")
        if response is not None:
            print(f"Código de status da resposta: {response.status_code}")
            try:
                print(f"Conteúdo da resposta: {response.json()}")
            except json.JSONDecodeError:
                print(f"Conteúdo da resposta (não JSON): {response.text}")
        return None

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

# --------------------------- Controle de Clusters ---------------------------

def compute_clusters(embeddings, n_clusters=5):
    """Função de clusterização com verificação de dimensões"""
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim == 1:
        embeddings_array = embeddings_array.reshape(-1, 1)
        
    Z = linkage(embeddings_array, method='complete', metric='cosine')
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(idx)
    
    return clusters, cluster_labels

def compute_balanced_clusters(embeddings, n_clusters=5):
    """
    Clusterização usando K-Means para clusters mais balanceados
    """
    # Converter para array numpy se necessário
    embeddings_array = np.array(embeddings)
    
    # Clusterização com K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Organizar os resultados
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)  # Armazena índices dos comentários
    
    return clusters, cluster_labels

# def cosine_similarity(a: List[float], b: List[List[float]]) -> List[float]:
#     """Calcula a similaridade de cossenos entre um vetor e uma matriz de vetores."""
#     a_np = np.array(a).reshape(1, -1)  # Garante que 'a' seja um vetor linha (1, n)
#     b_np = np.array(b)
#     if b_np.ndim == 1:
#         b_np = b_np.reshape(1, -1) # Garante que b_np seja uma matriz
#     dot_product = np.dot(a_np, b_np.T) # Produto escalar com a transposta de b
#     norm_a = np.linalg.norm(a_np)
#     norm_b = np.linalg.norm(b_np, axis=1) # Calcula as normas de cada vetor em b
#     if norm_a == 0:
#         return [0.0] * b_np.shape[0]
#     result =  dot_product / (norm_a * norm_b)
#     return result[0].tolist() # Retorna uma lista de similaridades


# --------------------------- Busca por Similaridade ---------------------------
st.header("🔎 Busca por Similaridade")

search_term = st.text_input("Digite uma frase para buscar comentários similares:", 
                          "Awkward")

num_results = st.number_input(
    "Número de comentários similares a mostrar:",
    min_value=1,
    max_value=20,
    value=3
)

if st.button("Buscar"):
    if not search_term:
        st.warning("Por favor, digite um termo para buscar")
    else:
        with st.spinner("Procurando comentários similares..."):
            
            # 1. Gera embedding para a busca
            search_embedding = create_embeddings([search_term])
            if not search_embedding:
                st.error("Falha ao gerar embedding para a busca")
                st.stop()
            
            result_container = similar_request_container(embedding_referencia=search_embedding[0])
            
            st.subheader(f"Top {len(result_container)} comentários mais similares:") # 3 Results for now
            
            for result in result_container:
                print(f"Text: {result['text']} - Similaridade {result['similarity']:.3f} ")
            
                with st.container():
                    st.markdown(f"""
                    <div style="border-left: 3px solid #4CAF50; padding-left: 10px; margin: 10px 0;">
                        <p style="font-weight: bold;">Similaridade: {result['similarity']:.3f} </p>
                        <p>{result['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(result['similarity']))
                        
# -----------------------------------------------
# --------------- Clusterização -----------------
# -----------------------------------------------

        # st.header("Clusters")
        # fig = ff.create_dendrogram(embeddings_array)
        # fig.update_layout(margin=dict(l=0, r=20, t=20, b=20))
        # st.plotly_chart(fig)

        # number_of_clusters = st.slider('How many clusters?', 2, 20, 5)

        # with st.status("Computing clusters.....", expanded=True):
        #     clusters = compute_clusters(embeddings=embeddings_array, n_clusters=number_of_clusters)[0].items()
        #     sorted_clusters = sorted(clusters, key=lambda x: len(x[1]) * -1)
        #     for cluster, labels in sorted_clusters:
        #         st.write(f"Cluster: {cluster} ({len(labels)})")
        #         st.write(labels)

