# Smoke-Fire-Detection

Uma solução de Deep Learning para classificação de imagens contendo fumaça, fogo ou nenhum dos dois. Este repositório contém um notebook Jupyter (`model.ipynb`) que treina, avalia e compara diferentes arquiteturas de Redes Neurais Convolucionais (CNNs) para esta tarefa.

## Pré-requisitos

*   Python 3.11 ou superior
*   `pip` ou `conda` para gerenciamento de pacotes

## Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/LucasdfRocha/Smoke-Fire-Detection
    cd Smoke-Fire-Detection
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3.  **Instale as dependências:**
    As dependências estão listadas no arquivo `pyproject.toml`. Você pode instalá-las usando `pip`:
    ```bash
    pip install -e .
    # Ou instale manualmente as bibliotecas principais:
    # pip install tensorflow pandas numpy matplotlib seaborn scikit-learn scikeras opencv-python tqdm pillow ultralytics pyyaml
    ```

## Preparação dos Dados

1.  **Baixe o Dataset:**
    *   O dataset original utilizado neste projeto é muito grande para ser incluído diretamente no repositório GitHub.
    *   Faça o download do dataset no seguinte link do Kaggle: [Smoke Fire Detection YOLO Dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)
    *   Após o download, descompacte o arquivo e **certifique-se de que a pasta principal contendo os dados seja nomeada como `data`** e colocada na raiz do projeto clonado.

2.  **Estrutura de Dados Esperada (Após Download):** Após baixar e nomear a pasta corretamente, a estrutura dentro do diretório `data/` deve ser semelhante a esta (formato YOLO):
    ```
    data/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    ```
    *   Onde `images/` contém os arquivos de imagem (`.jpg`, `.png`, etc.).
    *   E `labels/` contém os arquivos de anotação `.txt` correspondentes (um por imagem).

3.  **Execute o Script de Limpeza:** O notebook `model.ipynb` espera que os dados estejam organizados por classe (`fire`, `smoke`, `none`) em um diretório `processed_data/`. O script `data-cleaning.py` realiza essa organização:
    *   Ele lê os dados da estrutura YOLO em `data/`.
    *   Determina a classe principal de cada imagem (fire, smoke, none) com base nas anotações.
    *   Copia as imagens para a estrutura `processed_data/` necessária para o notebook.
    ```bash
    python data-cleaning.py
    ```
    Este script irá criar o diretório `processed_data` e organizar as imagens nas subpastas de classe apropriadas (`fire`, `smoke`, `none`) para cada conjunto (`train`, `val`, `test`). Ele também gera um arquivo `processed_data/metadata.csv`.

## Executando o Notebook de Treinamento

1.  **Inicie o Jupyter:** Certifique-se de que seu ambiente virtual está ativado e execute:
    ```bash
    jupyter notebook
    ```
    Ou use sua IDE preferida com suporte a notebooks Jupyter (como VS Code).

2.  **Abra e Execute `model.ipynb`:**
    *   Navegue até o arquivo `model.ipynb` na interface do Jupyter.
    *   Execute as células do notebook sequencialmente.

3.  **O que o Notebook Faz:**
    *   **Carrega os Dados:** Usa `ImageDataGenerator` do Keras para carregar imagens dos diretórios `processed_data/train`, `processed_data/val`, e `processed_data/test`.
    *   **Define Modelos:** Define três arquiteturas de CNN: `base_model`, `improved_model` (com Batch Normalization, Dropout, Regularização L2), e `create_model_for_search` (usada para busca de hiperparâmetros).
    *   **Treina e Avalia:**
        *   Treina e avalia o `base_model`.
        *   Treina e avalia o `improved_model`.
        *   Realiza uma busca manual simples por hiperparâmetros (`learning_rate`, `dropout_rate`, número de filtros) usando um conjunto pré-definido de combinações.
        *   Treina um `final_model` com os melhores hiperparâmetros encontrados na busca.
    *   **Visualiza Resultados:**
        *   Plota curvas de aprendizado (acurácia e perda) para treino e validação de cada modelo.
        *   Compara o desempenho dos três modelos no conjunto de teste.
        *   Gera e plota matrizes de confusão para cada modelo no conjunto de teste.
        *   Mostra exemplos de previsões corretas e incorretas para cada classe e modelo.
        *   Visualiza os mapas de ativação da primeira camada convolucional do `final_model` para uma imagem de exemplo.

## Saídas

*   **Gráficos:** Vários gráficos (`.png`) comparando o desempenho dos modelos, matrizes de confusão e exemplos de previsão são salvos no diretório `results/`.
*   **Modelos Treinados:** Os modelos treinados (`base_model_trained`, `improved_model_trained`, `final_model`) ficam disponíveis na memória do notebook para inspeção ou salvamento manual, se necessário. O notebook atual não salva explicitamente os pesos dos modelos em arquivos.
*   **Logs:** A saída das células do notebook mostra o progresso do treinamento, as métricas de avaliação e os relatórios de classificação.

## Script Adicional (`pretrained_model.py`)

Este repositório também contém `pretrained_model.py`, que utiliza a biblioteca `ultralytics` para treinar um modelo YOLOv8 para *detecção* de objetos (fogo e fumaça), em vez de classificação de imagem inteira. Ele usa o arquivo `data.yaml` para configuração do dataset. Para executá-lo (após preparar os dados no formato YOLO e configurar `data.yaml`):
