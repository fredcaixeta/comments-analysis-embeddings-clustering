import requests
import json

def carregar_embedding(filename: str = "embedding.json") -> list:
    """Carrega um embedding de um arquivo JSON.

    Args:
        filename: O nome do arquivo JSON do qual carregar o embedding.

    Returns:
        Uma lista de floats representando o embedding, ou None se ocorrer um erro.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if "embedding" in data and isinstance(data["embedding"], list) and all(isinstance(item, float) for item in data["embedding"]):
                return data["embedding"]
            else:
                print(f"Formato inválido no arquivo '{filename}'. Esperava uma lista de floats na chave 'embedding'.")
                return None
    except FileNotFoundError:
        print(f"Arquivo '{filename}' não encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON do arquivo '{filename}'.")
        return None
    except IOError as e:
        print(f"Erro ao ler o arquivo '{filename}': {e}")
        return None

def fazer_request_similar(embedding_referencia: list, api_url: str = "http://localhost:5000/similar"):
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
    
def listar():
    import requests
    import json

    API_URL = "http://localhost:5000/listar"

    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Levanta uma exceção para códigos de status de erro (4xx ou 5xx)
        data = response.json()
        print("Resposta da API (Listar Tudo):")
        print(json.dumps(data, indent=4))
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição para a API: {e}")
        if response is not None:
            print(f"Código de status da resposta: {response.status_code}")
            try:
                print(f"Conteúdo da resposta: {response.json()}")
            except json.JSONDecodeError:
                print(f"Conteúdo da resposta (não JSON): {response.text}")

if __name__ == "__main__":
    # Carrega o embedding de referência do arquivo embedding.json
    embedding_de_referencia = carregar_embedding()

    if embedding_de_referencia:
        # Faz a requisição para a API de similaridade
        resultados_similares = fazer_request_similar(embedding_de_referencia)

        if resultados_similares:
            print("\nResultados da busca de embeddings similares:")
            print(json.dumps(resultados_similares, indent=4))
        else:
            print("\nFalha ao obter resultados de embeddings similares.")
    else:
        print("\nFalha ao carregar o embedding de referência.")
    
    # Query to List
    #listar()