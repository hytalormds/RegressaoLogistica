# Modelo de Regressão Logística para Classificação de Dígitos MNIST

Este repositório contém um exemplo simples de implementação de um modelo de regressão logística usando TensorFlow para classificar dígitos manuscritos do conjunto de dados MNIST. O código está bem comentado para facilitar a compreensão e a adaptação para outros projetos.

## Pré-requisitos

Antes de executar o código, certifique-se de ter as seguintes bibliotecas instaladas:

```bash
pip install tensorflow tqdm
```

## Como Usar

1. Baixe ou clone este repositório em seu ambiente local.
2. Abra um terminal e navegue até o diretório do projeto.
3. Execute o script Python `logistic_regression_mnist.py`.

```bash
python logistic_regression_mnist.py
```

O modelo será treinado no conjunto de dados MNIST, e a precisão do teste será exibida no final do treinamento.

## Detalhes do Código

O código está dividido em seções claras para facilitar a compreensão:

- **Importação de Bibliotecas:** Importação das bibliotecas necessárias, incluindo TensorFlow e tqdm.

- **Carregamento dos Dados:** Utilização do módulo `input_data` do TensorFlow para carregar o conjunto de dados MNIST.

- **Definição do Modelo:** Criação do grafo computacional para a regressão logística usando TensorFlow.

- **Definição da Função de Perda e Otimizador:** Utilização da entropia cruzada como função de perda e otimizador de descida de gradiente.

- **Treinamento do Modelo:** Loop de treinamento para ajustar os pesos do modelo usando mini-lotes de dados.

- **Teste do Modelo:** Avaliação da precisão do modelo no conjunto de teste.

- **Fechamento da Sessão:** Encerramento da sessão do TensorFlow após a conclusão do treinamento.

## Nota

A precisão de teste pode variar devido à aleatoriedade no embaralhamento dos dados de treinamento.
