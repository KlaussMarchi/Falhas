# ESTRUTURA DE ARQUIVOS
- Datasets: pasta onde ficam os arquivos .dat das imagens e mascaras do problema. Atualmente estamos usando o dataset 2

- Seguir ordem dos arquivos .ipynb (1, 2, 3...)

### Separar Imagens
    A pasta Dataset é como se fosse o backup, para rodar o código temos que pegar o dataset2 (pastas images e masks) e jogar em "database/target", para ser lida no arquivo 1 - Prossing.ipynb 

### Pasta Model
Arquivos do modelo, desde o pré-processamento até a comparação final, obtendo:
1. Processing.ipynb -> pré processamento dos dados, lê os arquivos .dat e os processa, o output são arquivos .npy (imagens e máscara) e os joga na pasta "processed", que será lido pelo arquivo Model.ipynb

2. Model.ipynb -> Pega os arquivos da pasta "processed", e roda o modelo. Após rodado um backup com o modelo, as imagens de predição e informações vão para a pasta backup.

3. PostProcessing.ipynb -> Só compara os modelos treinados e armazenados na pasta backup

### Objetos auxiliares
Todos são encontrados na pasta "utils"

- Files.py -> funções auxiliares para ler arquivos e extrair imagens
- Losses -> Classe com todas as funções de perda, lidando com binário ou multiclasse
- Network -> classe do seletor da rede neural que será escolhida, é chamada no arquivo "Model.ipynb"


