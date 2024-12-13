from util import get_padrao, get_valor_aleatorio
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import model_from_json


class AutoencoderConfiguracao:
    def __init__(self, modelagem=None, input_shape=None):
        """
        Classe de configurações de autoencoders padrão
        :param modelagem: Modelagem do Autoencoder
        :param input_shape: Tamanho da entrada
        """
        self.modelagem = modelagem if modelagem is not None else get_padrao('POOL_MODELAGEM')

        self.input_shape = input_shape
        self.filtros = None
        self.kernel_size = None
        self.activation = None
        self.strides = None
        self.padding = None
        self.kernel_initializer = None
        self.nr_layers = None
        self.output_activation = None
        self.qtd_epocas = None

        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.latente = None
        self.seed = None

    def atualizar_modelagem(self):
        """
        Função para atualizar os dados do autoencoder baseado na sua modelagem (SLA)
        :return: Status de atualização
        """
        # MODELAGEM 'S'
        self.gerar_seed()

        # MODELAGEM 'L'
        self.gerar_latente()

        # MODELAGEM 'A'
        self.gerar_arquitetura()

        return True

    def gerar_seed(self):
        """
        Função que gera a seed do autoencoder baseado na modelagem
        :return: Seed gerada
        """
        if 'S' in self.modelagem.upper():
            self.seed = random.randint(get_padrao('AEC_SEED_RANDOM_INI'), get_padrao('AEC_SEED_RANDOM_FIM'))
        else:
            self.seed = get_padrao('AEC_SEED_PADRAO')

        tf.random.set_seed(self.seed)

        if get_padrao('DEBUG'):
            print(f'Seed: {self.seed}')

        return self.seed

    def gerar_latente(self):
        """
        Função que gera o vetor latente do autoencoder baseado na modelagem
        :return: Vetor latente gerada
        """
        if 'L' in self.modelagem.upper():
            self.latente = random.randint(get_padrao('AEC_VETOR_LATENTE_RANDOM_INI'),
                                          get_padrao('AEC_VETOR_LATENTE_RANDOM_FIM'))
        else:
            self.latente = get_padrao('AEC_VETOR_LATENTE_PADRAO')

        if get_padrao('DEBUG'):
            print(f'Latente: {self.latente}')

        return self.latente

    def gerar_arquitetura(self):
        """
        Função que gera a arquitetura do autoencoder baseado na modelagem
        :return: Classe atualizada
        """
        if 'A' in self.modelagem.upper():
            self.nr_layers, self.filtros, self.strides = self.get_layers_aleatorio()
            self.kernel_size = self.get_kernel_size_aleatorio()
            self.activation = self.get_activation_aleatorio()
            self.padding = self.get_padding_aleatorio()
            self.kernel_initializer = self.get_kernel_initializer_aleatorio()
            self.output_activation = self.get_output_activation_aleatorio()
            self.qtd_epocas = self.get_qtd_epocas_aleatorio()
        else:
            self.nr_layers = get_padrao('AEC_NR_LAYERS_PADRAO')
            self.filtros = get_padrao('AEC_FILTROS_PADRAO')
            self.kernel_size = tuple(get_padrao('AEC_KERNEL_SIZE_PADRAO'))
            self.activation = get_padrao('AEC_ACTIVATION_PADRAO')
            self.strides = get_padrao('AEC_STRIDES_PADRAO')
            self.padding = get_padrao('AEC_PADDING_PADRAO')
            self.kernel_initializer = get_padrao('AEC_KERNEL_INITIALIZER_PADRAO')
            self.output_activation = get_padrao('AEC_OUTPUT_ACTIVATION_PADRAO')
            self.qtd_epocas = get_padrao('AEC_QTD_EPOCAS_PADRAO')

        if get_padrao('DEBUG'):
            print(f'Nr. Layers: {self.nr_layers}')
            print(f'Filtros: {self.filtros}')
            print(f'Kernel Size: {self.nr_layers}')
            print(f'Activation: {self.activation}')
            print(f'strides: {self.strides}')
            print(f'Padding: {self.padding}')
            print(f'Kernel Initializer: {self.kernel_initializer}')
            print(f'Output Activation: {self.output_activation}')
            print(f'Qtd. Epocas: {self.qtd_epocas}')

        return self

    def get_layers_aleatorio(self):
        """
        Função que retorna a quantidade de layers e os valores de cada layer
        :return: Número de layeres e valores de layers aleatório
        """

        nr_layers = random.randint(get_padrao('AEC_QTD_LAYERS_RANDOM_INI'), get_padrao('AEC_QTD_LAYERS_RANDOM_FIM'))
        valores_layers = []
        for valor in range(0, nr_layers):
            valores_layers.append(get_valor_aleatorio(get_padrao('AEC_VALORES_LAYERS_RANDOM')))

        controle, qtd_layers, valores_strides = self.input_shape[0], 0, []

        for v in valores_layers:
            if controle == 1:
                break
            if controle % 2 == 0:
                controle = controle / 2
                valores_strides.append(2)
            elif controle % 3 == 0:
                controle = controle / 3
                valores_strides.append(3)
            else:
                break
            qtd_layers += 1

        nr_layers = qtd_layers
        valores_layers = valores_layers[:nr_layers]
        return nr_layers, valores_layers, valores_strides

    def get_kernel_size_aleatorio(self):
        """
        Função que retorna um valor de kernel size aleatorio
        :return: Kernel size aleatório
        """
        valor = get_valor_aleatorio([2, 3])
        return (valor, valor)

    def get_activation_aleatorio(self):
        """
        Função que retorna um valor de activation aleatorio
        :return: Activation aleatório
        """
        return get_valor_aleatorio(['relu'])

    def get_padding_aleatorio(self):
        """
        Função que retorna um valor de padding aleatorio
        :return: Padding aleatório
        """
        return get_valor_aleatorio(['same'])

    def get_kernel_initializer_aleatorio(self):
        """
        Função que retorna um valor de kernel initializer aleatorio
        :return: Kernel initializer aleatório
        """
        return get_valor_aleatorio(['he_uniform'])

    def get_strides_aleatorio(self):
        """
        Função que retorna um valor de strides aleatorio
        :return: Striders aleatório
        """
        return get_valor_aleatorio([2, 3])

    def get_output_activation_aleatorio(self):
        """
        Função que retorna um valor de output activation aleatorio
        :return: Output activation aleatório
        """
        return get_valor_aleatorio(['linear'])

    def get_qtd_epocas_aleatorio(self):
        """
        Função que retorna a quantidade de épocas aleatorio
        :return: Quantidade de épocas aleatório
        """
        return random.randint(get_padrao('AEC_QTD_EPOCAS_RANDOM_INI'), get_padrao('AEC_QTD_EPOCAS_RANDOM_FIM'))


class Autoencoder(AutoencoderConfiguracao):
    def __init__(self, modelagem=None, base=None, id=None, input_shape=None):
        """
        Classe padrão de criação de autoencoder
        :param modelagem: Modelagem do Autoencoder
        :param base: Base utilizada pelo autoencoder
        :param id: Código de identificação do autoencoder
        :param input_shape: Tamanho das imagens de entrada
        """
        super().__init__(modelagem=modelagem, input_shape=input_shape)
        self.base = base
        self.id = id

    def salvar(self):
        """
        Função utilizada para salvar o autoencoder, encoder e informações do encoder em arquivos
        :return: Status de Salvamento (Sempre True)
        """
        if get_padrao('DEBUG'):
            print(f'Salvando autoencoder {self.id}')

        nm_arquivo = f"{get_padrao('AEC_DIRETORIO')}/autoencoder_{str(self.id).zfill(3)}"
        with open(f"{nm_arquivo}.json", "w") as json_file:
            json_file.write(self.autoencoder.to_json())

        self.autoencoder.save_weights(f"{nm_arquivo}.weights.h5")

        nm_arquivo = f"{get_padrao('AEC_DIRETORIO')}/encoder_{str(self.id).zfill(3)}"
        with open(f"{nm_arquivo}.json", "w") as json_file:
            json_file.write(self.encoder.to_json())

        self.encoder.save_weights(f"{nm_arquivo}.weights.h5")

        with open(f"{nm_arquivo}.txt", 'w') as f:
            f.write(f'AUTOENCODER {self.id}\n')
            f.write(f'SEED: {str(self.seed)}\n')
            f.write(f'LATENTE: {str(self.latente)}\n')
            f.write(f'NR LAYERS: {str(self.nr_layers)}\n')
            f.write(f'FILTROS: {str(self.filtros)}\n')
            f.write(f'KERNEL SIZE: {str(self.kernel_size)}\n')
            f.write(f'ACTIVATION: {str(self.activation)}\n')
            f.write(f'STRIDES: {str(self.strides)}\n')
            f.write(f'PADDING: {str(self.padding)}\n')
            f.write(f'KERNEL INITIALIZER: {str(self.kernel_initializer)}\n')
            f.write(f'OUTPUT ACTIVATION: {str(self.output_activation)}\n')
            f.write(f'QTD EPOCAS: {str(self.qtd_epocas)}\n')

        return True

    def criar(self):
        """
        Função que cria um autoencoder, onde cada etapa é uma função separada
        :return: Autoencoder criado
        """
        self.atualizar_modelagem()

        self.criar_encoder()

        self.criar_decoder()

        self.autoencoder = models.Sequential([self.encoder, self.decoder])

        if get_padrao('DEBUG'):
            self.autoencoder.summary()

        self.treinar()

        self.salvar()

        return self

    def treinar(self):
        """
        Função responsável pelo treinamento do autoencoder
        :return: Status de treinamento (Sempre True)
        """
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(self.base.x_train, self.base.x_train, epochs=self.qtd_epocas, batch_size=64, shuffle=True,
                             validation_data=(self.base.x_test, self.base.x_test))

        return True

    def calcular_saida_encoder(self):
        """
        Função que calcula a saída do encoder para reconstrução exata no decoder
        :return: Altura e largura da saída do encoder
        """
        controle, qtd_layers, valores_strides = self.input_shape[0], 0, []
        for stride in self.strides:
            controle = controle / stride

        return int(controle), int(controle)

    def criar_encoder(self):
        """
        Função responsável em criar a primeira parte do autoencoder, o Encoder.
        Ele começa criando um sequencial, onde o Input é o shape de entrada do Autneocder.
        Após isso são inseridas as camadas de convolução, baseado na arquitetura do autoencoder.
        Por fim, é adicionado a camada do vetor latente, adicionando uma camada dense, do tamanho
        de vetor latente.
        :return: Encoder criado
        """
        self.encoder = models.Sequential()
        self.encoder.add(layers.Input(shape=self.input_shape))
        for camada in range(0, self.nr_layers):
            self.encoder.add(
                layers.Conv2D(filters=self.filtros[camada], kernel_size=self.kernel_size, activation=self.activation,
                              strides=self.strides[camada], padding=self.padding,
                              kernel_initializer=self.kernel_initializer))

        self.encoder.add(layers.Flatten())
        self.encoder.add(layers.Dense(self.latente, activation=self.activation))

        if get_padrao('DEBUG'):
            self.encoder.summary()

        return self.encoder

    def criar_decoder(self):
        """
        Função responsável em criar a segunda parte do autoencoder, o Decoder.
        Ele começa calculando qual é a altura e largura da primeira etapa, para ser possível reconstruir fielmente o decoder,
        baseado no encoder.
        É adicionado uma camada inicial com a altura e largura calculada, e depois adicionado as camadas de convolução transpose,
        com as mesmas configurações do encoder.

        Por fim, é criado a cama de saída, com filtro de 1
        :return: Decoder criado
        """
        altura, largura = self.calcular_saida_encoder()
        self.decoder = models.Sequential()
        self.decoder.add(layers.Input(shape=(self.latente,)))
        self.decoder.add(
            layers.Dense(units=self.filtros[self.nr_layers - 1] * altura * largura, activation=self.activation))
        self.decoder.add(layers.Reshape((altura, largura, self.filtros[self.nr_layers - 1])))

        for camada in range(self.nr_layers - 1, -1, -1):
            self.decoder.add(
                layers.Conv2DTranspose(filters=self.filtros[camada], kernel_size=self.kernel_size,
                                       strides=self.strides[::-1][camada],
                                       activation=self.activation, padding=self.padding,
                                       kernel_initializer=self.kernel_initializer))

        self.decoder.add(layers.Conv2D(filters=1, kernel_size=self.kernel_size, activation=self.output_activation,
                                       padding=self.padding))

        if get_padrao('DEBUG'):
            self.decoder.summary()

        return self.decoder

    def carregar_model(self, json_path=None, weights_path=None, tipo='autoencoder'):
        """
        Função de carregar o autoencoder por meio de um json
        :param json_path: Path do json do modelo
        :param weights_path: Path do arquivo de pesos do modelo
        :param tipo: Tipo do arquivo (Autoencoder ou encoder)
        :return: Modelo carregado
        """
        arquivo_json = open(json_path, 'r')
        model_json_carregado = arquivo_json.read()
        arquivo_json.close()
        model_carregado = model_from_json(model_json_carregado)
        model_carregado.load_weights(weights_path)
        if tipo == 'encoder':
            self.encoder = model_carregado
        else:
            self.autoencoder = model_carregado

        return self
