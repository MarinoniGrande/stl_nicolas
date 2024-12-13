import json
import random

DIRETORIO_PRINCIPAL = '/content/drive/MyDrive/Mestrado'


def get_padrao(variavel=None):
    """
    Função que recebe uma variavel e retorna seu valor padrào do projeto
    :param variavel: Variável que deseja retornar o valor padrão
    :return: Valor padrão da variável
    """
    with open(f'{DIRETORIO_PRINCIPAL}/configuracoes.json') as f:
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
