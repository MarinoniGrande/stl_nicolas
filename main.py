import sys
import BO.base.base
import BO.autoencoder.pool
import tensorflow as tf
import numpy as np

from util import get_padrao, carregar_configuracoes

carregar_configuracoes(sys.argv[1])

base = BO.base.base.Base(tipo='unlabeled')
_, _ = base.carregar()
base_labeled = BO.base.base.Base(tipo='labeled', is_base_separada=True, diretorio=get_padrao('BASE_ALVO_DIRETORIO'))
_, _ = base_labeled.carregar()
x_target = tf.reshape(base_labeled.x_test, (-1,) + base_labeled.x_test[0].shape)

for d in [
    "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/s",
    "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/l",
    "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/a",
    "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/sl",
    "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/la",
    "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/sa",
    "/content/drive/MyDrive/Mestrado/autoencoders/kyoto/sla",
]:
    print(d)
    pool = BO.autoencoder.pool.Pool(base=base, diretorio=d)
    _ = pool.carregar_pool(tipo='encoder')
    pool.aplicar_funcao_custo_offline()
    # _ = pool.aplicar_finetuning(x_target=base_labeled.x_train)

    # Example usage (assuming 'pool' and 'base_labeled' are defined as in the original code):
    Lpb = evaluate_autoencoders(pool, base_labeled, x_target, d)
    cc, ee = 0, 0
    soma = np.sum([Lpb], axis=1)
    soma = soma.reshape(soma.shape[1], soma.shape[2])
    predicted_ensemble = np.argmax(soma, axis=1)
    for m in range(0, len(predicted_ensemble)):
        b = predicted_ensemble[m]
        c = base_labeled.y_test[m]
        if (b == c):
            cc = cc + 1
        else:
            ee = ee + 1
    res_soma = cc / (cc + ee)
    print(f'Soma: {res_soma}')

    cc, ee = 0, 0
    prod = np.product([Lpb], axis=1)
    prod = prod.reshape(prod.shape[1], prod.shape[2])
    predicted_ensemble = np.argmax(prod, axis=1)
    for m in range(0, len(predicted_ensemble)):
        b = predicted_ensemble[m]
        c = base_labeled.y_test[m]
        if (b == c):
            cc = cc + 1
        else:
            ee = ee + 1
    res_prod = cc / (cc + ee)
    print(f'Prod: {res_prod}')

# prompt: Having the pool of autoencoders (pool.pool) Make a loop in each autoencoder and apply a randomforest classifier with the base_labeled.x_test and y_test, and after that get the general accuracy of all the classifiers with the product method and the sum method

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def evaluate_autoencoders(pool, base_labeled, x_target, diretorio):
    """
    Evaluates a pool of autoencoders using a RandomForestClassifier.

    Args:
        pool: The pool of autoencoders.
        base_labeled: The labeled dataset for evaluation.

    Returns:
        A tuple containing the product and sum of accuracies.
    """
    lista_predicoes = []
    all_accuracies = []
    for autoencoder in pool.pool:
        # Load the encoded features
        print(autoencoder.id)
        try:
            # finetuning nao fiz, o que fiz é extracao de caracteristicas.
            # finetuning seria reajustar os pesos do autoencoder
            encoded_features = np.load(f"{diretorio}/encoder_{str(autoencoder.id).zfill(3)}.npy")
        except FileNotFoundError:
            print(f"Warning: Could not find encoded features for autoencoder {autoencoder.id}. Skipping.")
            continue

        # Split data (assuming base_labeled has x_test and y_test)
        # x_train, x_test, y_train, y_test = train_test_split(encoded_features, base_labeled.y_test, test_size=0.3, random_state=42)

        # Train and evaluate the RandomForestClassifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Example parameters
        # rf_classifier = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
        rf_classifier.fit(encoded_features, base_labeled.y_train)
        resultado = autoencoder.encoder.predict(x_target)
        predictions = rf_classifier.predict_proba(resultado)
        lista_predicoes.append(predictions)
        # accuracy = accuracy_score(base_labeled.y_test, predictions)
        # all_accuracies.append(accuracy)
    # product_accuracy = np.prod(all_accuracies)
    # sum_accuracy = np.sum(all_accuracies)

    return lista_predicoes
