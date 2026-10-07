# dataset_zu × dataset_wu

## Resumo

- O `200-20.zip` que os autores da ResACEUnet (Zu et al. 2024) publicaram no Zenodo 20339874 **é o FaultSeg3D** (Wu et al. 2019), o mesmo dado do `dataset_wu`: os 220 volumes e os 220 rótulos são iguais bit a bit (máx |dif| = 0), na mesma orientação.
- Eles não geraram dado novo nem mexeram nos valores: o que muda é o formato do arquivo e a numeração.
- Com a normalização deste `Format` (a mesma do `dataset_wu` e do Marlim), os tiles normalizados também saem iguais aos do `dataset_wu`; só a ordem dos arquivos é outra.

## Como foi verificado

- O `Compare.ipynb` desta pasta refaz tudo abaixo do zero.

- Desvio, média e fração de falha de cada volume: as listas ordenadas dos dois conjuntos são idênticas (máx |dif| = 0), e cada volume do zenodo tem um par único no `dataset_wu`.
- Cada par foi comparado voxel a voxel nas 48 orientações possíveis (6 transposições × 8 espelhamentos): os 220 batem na orientação original, sem transposição nem espelho, e os 220 rótulos também.
- Na primeira comparação eu tinha olhado volumes de mesmo número (o 0 contra o 0, correlação 0,02) e concluído errado que eram dados diferentes; as diferenças de período e de escala que medi ali eram entre volumes diferentes, não entre os conjuntos.

## O que os autores fizeram com o dado do Wu

| | FaultSeg3D (`dataset_wu/original`) | Zenodo `200-20.zip` |
|---|---|---|
| Arquivo | `.dat` float32 cru, 128³ | `.npy` float32, 128³, o mesmo array em `(inline, xline, tempo)` |
| Treino | 0–199 | 0–199: os mesmos 200 volumes, embaralhados (só 1 manteve o número) |
| Validação | 200–219 | 200–219: os mesmos 20 volumes, também renumerados |
| Amplitude | já padronizada (média ~0, desvio ~1) | idêntica |
| Rótulo | 0/1 | idêntico |

- O pré-processamento deles acontece no código, na hora de carregar (min-max por volume, augmentação, recorte 96³), não nos arquivos.
- O artigo (seção 3.1) diz que eles geraram 220 volumes pelo método de Wu et al. (2020), 200 para treino e 20 para validação. O arquivo publicado é o conjunto do FaultSeg3D com a mesma divisão 200/20; não há sinal de dado gerado por eles.

## O que isso muda aqui

- Treinar no `dataset_zu` é treinar no mesmo dado do `dataset_wu`. A diferença é o split: o `1 - Model` sorteia validação e teste pela ordem do `DataBase.csv`, então os 10 + 10 volumes separados aqui são outros que os do `dataset_wu`. Comparar os dois mede o efeito do sorteio, não do dado.
- A divisão dos autores (treino = 0–199, validação = 200–219, os mesmos 20 em que eles medem o IoU de 0,764) não é usada: o `1 - Model` não tem split fixo.
- Os p01/p99 do conjunto são os mesmos nos dois (−2,648 e 2,693), e os dois `DataBase.csv` têm as mesmas estatísticas, só em outra ordem.

## Correspondência

Número do arquivo no zenodo → número do mesmo volume no `dataset_wu`.

### Treino

| zenodo → wu | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| 0 → 69 | 1 → 143 | 2 → 29 | 3 → 58 | 4 → 122 | 5 → 193 | 6 → 119 | 7 → 96 | 8 → 65 | 9 → 13 |
| 10 → 129 | 11 → 35 | 12 → 11 | 13 → 150 | 14 → 15 | 15 → 98 | 16 → 127 | 17 → 37 | 18 → 28 | 19 → 186 |
| 20 → 115 | 21 → 145 | 22 → 91 | 23 → 24 | 24 → 8 | 25 → 83 | 26 → 7 | 27 → 105 | 28 → 112 | 29 → 88 |
| 30 → 52 | 31 → 116 | 32 → 155 | 33 → 120 | 34 → 0 | 35 → 47 | 36 → 163 | 37 → 159 | 38 → 151 | 39 → 64 |
| 40 → 60 | 41 → 100 | 42 → 167 | 43 → 174 | 44 → 26 | 45 → 41 | 46 → 27 | 47 → 40 | 48 → 176 | 49 → 14 |
| 50 → 23 | 51 → 153 | 52 → 25 | 53 → 89 | 54 → 164 | 55 → 147 | 56 → 79 | 57 → 139 | 58 → 123 | 59 → 130 |
| 60 → 95 | 61 → 45 | 62 → 1 | 63 → 114 | 64 → 158 | 65 → 76 | 66 → 177 | 67 → 146 | 68 → 92 | 69 → 160 |
| 70 → 38 | 71 → 94 | 72 → 185 | 73 → 192 | 74 → 18 | 75 → 48 | 76 → 17 | 77 → 5 | 78 → 106 | 79 → 53 |
| 80 → 173 | 81 → 118 | 82 → 170 | 83 → 44 | 84 → 171 | 85 → 32 | 86 → 20 | 87 → 55 | 88 → 101 | 89 → 42 |
| 90 → 131 | 91 → 195 | 92 → 68 | 93 → 6 | 94 → 99 | 95 → 85 | 96 → 73 | 97 → 86 | 98 → 157 | 99 → 30 |
| 100 → 141 | 101 → 156 | 102 → 12 | 103 → 97 | 104 → 191 | 105 → 80 | 106 → 182 | 107 → 180 | 108 → 10 | 109 → 82 |
| 110 → 56 | 111 → 154 | 112 → 168 | 113 → 107 | 114 → 149 | 115 → 84 | 116 → 71 | 117 → 188 | 118 → 113 | 119 → 126 |
| 120 → 81 | 121 → 121 | 122 → 197 | 123 → 187 | 124 → 189 | 125 → 138 | 126 → 183 | 127 → 108 | 128 → 194 | 129 → 2 |
| 130 → 57 | 131 → 135 | 132 → 3 | 133 → 4 | 134 → 125 | 135 → 117 | 136 → 21 | 137 → 63 | 138 → 110 | 139 → 152 |
| 140 → 78 | 141 → 104 | 142 → 59 | 143 → 140 | 144 → 133 | 145 → 102 | 146 → 124 | 147 → 137 | 148 → 36 | 149 → 178 |
| 150 → 134 | 151 → 67 | 152 → 166 | 153 → 34 | 154 → 31 | 155 → 50 | 156 → 9 | 157 → 16 | 158 → 72 | 159 → 190 |
| 160 → 77 | 161 → 148 | 162 → 196 | 163 → 175 | 164 → 132 | 165 → 181 | 166 → 93 | 167 → 70 | 168 → 74 | 169 → 111 |
| 170 → 199 | 171 → 19 | 172 → 136 | 173 → 109 | 174 → 54 | 175 → 162 | 176 → 128 | 177 → 66 | 178 → 87 | 179 → 103 |
| 180 → 43 | 181 → 75 | 182 → 172 | 183 → 165 | 184 → 198 | 185 → 90 | 186 → 61 | 187 → 169 | 188 → 51 | 189 → 22 |
| 190 → 39 | 191 → 142 | 192 → 144 | 193 → 62 | 194 → 46 | 195 → 33 | 196 → 49 | 197 → 161 | 198 → 184 | 199 → 179 |

### Validação

| zenodo → wu | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| 200 → 209 | 201 → 216 | 202 → 204 | 203 → 207 | 204 → 213 | 205 → 219 | 206 → 212 | 207 → 210 | 208 → 208 | 209 → 201 |
| 210 → 215 | 211 → 205 | 212 → 200 | 213 → 217 | 214 → 202 | 215 → 211 | 216 → 214 | 217 → 206 | 218 → 203 | 219 → 218 |
