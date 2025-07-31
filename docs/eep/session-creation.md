# **Configurando Sessões**

Esta documentação aborda as configurações necessárias para estabelecer uma sessão no projeto Arrasta. As últimas atualizações resultaram em mudanças significativas na configuração da sessão, movendo parte das configurações para arquivos ***.csv***, além de um novo painel de configuração.

## **1. Localização dos Arquivos de Configuração**

Os parâmetros de configuração da sessão estão distribuídos em diferentes arquivos .csv, localizados, geralmente, em ***arrasta-v1.0.2\design\drag-n-drop***. Dentro deste diretório, encontramos três subdiretórios relevantes:

- blocks: Configuração dos blocos da sessão.
- instructions: Instruções da sessão.
- trials: Configuração das tentativas da sessão.

## **2. Arquivos e Parâmetros**

### **2.1. Diretório "blocks"**

O arquivo "exemplo.csv" contém os seguintes parâmetros para configurar os blocos da sessão:

- **ID**: Identificador do bloco.
- **Name**: Nome do bloco.
- **EndSessionOnCriterion**: Define se a sessão é finalizada após o desempenho atingir o critério do bloco.
- **EndSessionOnNotCriterionAfterBlockRepetitions**: Determina se a sessão será encerrada se o desempenho **NÃO** atingir o critério após um certo número de repetições de bloco.
  - Se **T** (True) a sessão é finalizada após o desempenho NÃO atingir o critério do bloco.
  - Se **F** (False), desabilitado.
- **RepeatStyle**: Controla como um mesmo bloco é repetido caso o desempenho não atinja o critério do bloco.
  - Se **None**, desabilitado, nenhuma repetição é realizada.
  - Se **Consecutive**, o bloco é repetido ***MaxBlockRepetitionConsecutives*** vezes.
  - Se **Global**, o bloco é repetido ***MaxBlockRepetitionInSession*** vezes, porém não bloqueia o fluxo quando ***NextBlockOnNotCriterion > -1***.
  - Se **ConsecutiveAndGlobal**, o bloco utiliza ambos ***Consecutive*** e ***Global***.
- **MaxBlockRepetitionConsecutives**: Número de repetições permitidas no bloco. Ver RepeatStyle.
- **MaxBlockRepetitionInSession**: Número de repetições permitidas no bloco. Ver RepeatStyle.
- **EndCriterionStyle**: Define o critério de desempenho utilizado para determinar o encerramento da sessão.
  - Se **HitCount**, o número de acertos é usado como critério de desempenho.
  - Se **MissCount**, o número de erros é usado como critério de desempenho.
  - Se **ConsecutiveHits**, o número de acertos consecutivos é usado como critério de desempenho.
  - Se **ConsecutiveMisses**, o número de erros consecutivos é usado como critério de desempenho.
  - Se **HitPorcentage**, a porcentagem de acertos é usada como critério de desempenho.
  - Se **MissPorcentage**, a porcentagem de erros é usada como critério de desempenho.
- **EndCriterionEvaluationTime**: Define quando o critério de desempenho de um bloco é avaliado durante uma sessão.
  - Se **OnTrialEnd**, o critério de desempenho do bloco é avaliado tentativa à tentativa.
  - Se **OnBlocEnd**, o critério de desempenho do bloco é avaliado ao final da última tentativa do bloco.
- **NextBlockOnCriterion**: ID do bloco seguinte após o critério de desempenho ser atingido.
- **NextBlockOnNotCriterion**: ID do bloco seguinte após o critério de desempenho **NÃO** ser atingido.
- **EndCriterionValue**: valor do critério de desempenho. Ver EndCriterionStyle.
- **Reinforcement**: Porcentagem de tentativas do bloco com reforçamento diferencial.


### **2.2. Diretório "instructions"**
O arquivo "exemplo.csv" neste diretório inclui parâmetros para as instruções da sessão, como:

- **Trial**: Identificador da tentativa.
- **Block**: Identificador do bloco associado.
- **Instruction**: Representa as instruções associadas a uma tentativa ou bloco na sessão.
- **DoCalibration**: Determina se a calibração deve ser realizada durante a sessão.
  - Se **T** (True), a calibração é realizada.
  - Se **F** (Falso), a calibração é ignorada.

### **2.3. Diretório "trials"**
O arquivo "multi-sample.csv" contém parâmetros para configurar as tentativas da sessão, incluindo:

- **ID**: Identificador da tentativa.
- **Trials**: Quantidade de tentativas.
- **Relation**: Tipo de relação entre estímulos (A-A, A-B, A-C, C-B, B-A, ...).
- **Samples**: Quantidade de estímulos modelos (móveis).
- **Comparisons**: Quantidades de estímulos alvos (fixos).
- **DragableAnimation**:
- **StimuliFolder**: Nome da pasta  de imagens usada para carregar os arquivos png: media/png/NOME_DA_PASTA (escrever apenas NOME_DA_PASTA e não o endereço completo).
- **GridSize**: Tamanho da grade de estímulos em formato NxN.
  - Por exemplo, se = 4, teremos uma matriz 4x4. Logo, é possível posicionar 16 estímulos na grade.
- **Orientation**: Tipo de orientação do arraste.
    - Se **None**, sem orientação.
    - Se **TopToBottom**, orientação é do tipo "cima para baixo".
    - Se **BottomToTop**, ortientação é do tipo "baixo para cima".
    - Se **LeftToRight**, orientação é do tipo "esquerda para direita".
    - Se **RightToLeft**, orientação é do tipo "direita para esquerda".
    - Se **Custom**, orientação é do tipo customizada.
- **AutoAnimatedOnStart**: Controla a ativação da animação da borda do estímulo modelo.
  - Se **T** (True), a animação é ativada assim que a tentativa é iniciada, automaticamente  em um dos estímulos modelos.
  - Se **F** (False), a animação é ativada somente depois que um estímulo modelo for selecionado.
- **FoodDispensingRule**: define a regra para a liberação de comida (ativação do dispensador).
  - Se **FoodDisabled**, não há liberação.
  - Se **FoodOnFirstTryOnly**, há liberação apenas para acertos de primeira.
  - Se **FoodOnRetryAllowedOnCompletionOnly**, há liberação apenas para acertos na última relação.
  - Se **FoodOnFirstTryOnlyOnCompletionOnly**, há liberação apenas para acertos na última relação e apenas para acertos de primeira.
  - Se **FoodOnRetryAllowed**, há liberação para acertos.

### **2.4. Arquivo "exemplo.csv"**

Este arquivo é crucial para a criação da sessão, pois relaciona os parâmetros definidos nos arquivos anteriores. Os parâmetros incluem:

- **BlockID**: Identificador do bloco associado.
- **TrialID**: Identificador da tentativa associada.
- **TrialIDSource**: fonte da tentativa.

Basicamente, o objetivo deste arquivo é pesquisar pelos ID's de cada um dos parâmetros nos diretórios previamente mencionados e relacioná-los com aqueles encontrados nos arquivos específicos dentro de cada pasta correspondente. Através desses IDs, é possível acessar todos os outros parâmetros configurados em cada linha das planilhas presentes nos arquivos. Portanto, é crucial garantir a precisão da configuração desses arquivos, evitando linhas ou colunas incompletas, bem como o uso de tipos de parâmetros inexistentes.

## **3. Recomendações de Edição**

Para facilitar a visualização e edição dos arquivos .csv, recomendamos o uso do LibreOffice. Disponível em [Site do LibreOffice](https://pt-br.libreoffice.org/baixe-ja/libreoffice-novo/).