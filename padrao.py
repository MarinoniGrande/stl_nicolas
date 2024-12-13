import json

def get_padrao(variavel=None):
    """

    :param variavel:
    :return:
    """
    with open('configuracoes.json') as f:
        data = json.load(f)

    return data.get(variavel)