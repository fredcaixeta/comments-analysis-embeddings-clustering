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

# --------------------------- Carregamento de dados ---------------------------
all_comments = carregar_comentarios_de_txt()
embeddings_carregados = carregar_embeddings_do_json(r"app_duckdb\comments_1221_mistral.json")

# Processamento dos embeddings (igual ao anterior)
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

print(f"\nEmbeddings válidos: {len(valid_embeddings)}")
print(f"Embeddings inválidos: {len(invalid_indices)}")
print(f"Exemplo de índices inválidos: {invalid_indices[:5]}")

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

# -----------------------------------------------
# 1. BUSCA DE COMENTÁRIOS (igual ao original)
# -----------------------------------------------
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
            
            search_embedding = np.array(search_embedding).reshape(1, -1)
            
            # 2. Calcula similaridades
            similarities = cosine_similarity(search_embedding, embeddings_array)[0]
            
            # 3. Pega os índices dos mais similares
            top_indices = np.argsort(similarities)[-num_results:][::-1]
            
            # 4. Verifica se os índices são válidos
            valid_indices = [idx for idx in top_indices if idx < len(valid_comments)]
            if not valid_indices:
                st.error("Nenhum resultado válido encontrado")
                st.stop()
            
            # 5. Mostra resultados
            st.subheader(f"Top {len(valid_indices)} comentários mais similares:")
            for idx in valid_indices:
                with st.container():
                    st.markdown(f"""
                    <div style="border-left: 3px solid #4CAF50; padding-left: 10px; margin: 10px 0;">
                        <p style="font-weight: bold;">Similaridade: {similarities[idx]:.3f} </p>
                        <p>{valid_comments[idx]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(similarities[idx]))
                        
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

