import json
import random

DIRETORIO_PRINCIPAL = r'C:\Users\nmgra\PycharmProjects\stl_nicolas'
ARQUIVO_CONFIGURACOES = 'configuracoes.json'

def carregar_configuracoes(arquivo=None):
    global ARQUIVO_CONFIGURACOES
    ARQUIVO_CONFIGURACOES = arquivo
    return True

def get_padrao(variavel=None):
    """
    Função que recebe uma variavel e retorna seu valor padrào do projeto
    :param variavel: Variável que deseja retornar o valor padrão
    :return: Valor padrão da variável
    """
    with open(f'{DIRETORIO_PRINCIPAL}/{ARQUIVO_CONFIGURACOES}') as f:
        data = json.load(f)

    return data.get(variavel)


def get_valor_aleatorio(lista=None):
    """
    Função que recebe lista e retorna um valor aleatório dessa lista
    :param lista: Lista com dados aleatórios
    :return: Valor aleatório da lista
    """

    if lista is None:
        lista = []

    if len(lista) == 0:
        return None
    numero_aleatorio = random.randint(0, len(lista) - 1)
    return lista[numero_aleatorio]
