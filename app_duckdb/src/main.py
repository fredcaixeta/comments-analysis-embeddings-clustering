from flask import Flask, jsonify, request, Response
import duckdb
import numpy as np
import json
from typing import List

app = Flask(__name__)
DATABASE_FILE = "/app/data/embeddings.duckdb"  # Caminho dentro do container
#DATABASE_FILE = "/app/data/novo_banco.duckdb"


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calcula a similaridade de cossenos entre dois vetores."""
    a_np = np.array(a)
    b_np = np.array(b)
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def listar_tudo():
    """Lista todos os dados da tabela embeddings."""
    try:
        with duckdb.connect(DATABASE_FILE) as conn:
            data = conn.execute("SELECT * FROM embeddings").fetchall()
            columns = [desc[0] for desc in conn.description]
            resultados = []
            for row in data:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8', errors='ignore')
                    elif isinstance(value, list):
                        value = [float(x) if isinstance(x, (int, float)) else x for x in value]
                    else:
                        value = str(value) if not isinstance(value, (str, int, float, bool, list, type(None))) else value
                    row_dict[col] = value
                resultados.append(row_dict)
            return Response(json.dumps({"data": resultados}), mimetype='application/json')
    except duckdb.Error as e:
        return jsonify({"error": str(e)}), 500


def buscar_similares(reference_embedding: list[float], top_n: int = 3):
    """Busca os top N embeddings mais similares."""
    resultados_similaridade = []
    try:
        with duckdb.connect(DATABASE_FILE) as conn:
            data = conn.execute("SELECT id, text, embedding FROM embeddings").fetchall()
            for id, text, embedding in data:
                print(f"ID: {id}, Texto: '{text}', Tipo da embedding lida do DB: {type(embedding)}, Tamanho: {len(embedding) if embedding else 0}")
                if not embedding:
                    print(f"AVISO: Embedding vazia encontrada para ID {id}")
                    continue  # Saltar este registro se a embedding estiver vazia

                similarity = cosine_similarity(reference_embedding, embedding)
                resultados_similaridade.append({"id": id, "text": text, "similarity": similarity, "embedding": list(embedding)})
            resultados_ordenados = sorted(resultados_similaridade, key=lambda item: item['similarity'], reverse=True)
            return jsonify(resultados_ordenados[:top_n])
    except duckdb.Error as e:
        return jsonify({"error": str(e)}), 500

@app.route('/listar', methods=['GET'])
def listar_tudo_endpoint():
    return listar_tudo()

# @app.route('/listar', methods=['GET'])
# def listar_tudo_endpoint():
#     return jsonify({"message": "API está funcionando"})

@app.route('/similar', methods=['POST'])
def buscar_similares_endpoint():
    data = request.get_json()
    reference_embedding = data.get('embedding')
    if not isinstance(reference_embedding, list) or not all(isinstance(x, float) for x in reference_embedding):
        return jsonify({"error": "Embedding de referência inválido"}), 400
    return buscar_similares(reference_embedding)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')