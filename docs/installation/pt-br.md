# Instalação

O ambiente de desenvolvimento no Windows (x86_64-win64) contém:

- Lazarus IDE versão trunk (3.4+), a interface de desenvolvimento.
- Free Pascal Compiler versão trunk (3.3.1+), o compilador.
- O repositório do projeto com suas dependências no GitHub.

Os passos a seguir devem demorar aproximadamente 45min e assumem que você possui uma boa conexão banda larga.

# Passo-a-passo

## Passo 1 (20 min) - fpcupdeluxe, Lazarus IDE, FPC
1. Crie a pasta `C:\fpcupdeluxe\`
2. Crie a pasta `C:\lazarus-trunk\`
3. Baixe o arquivo [fpcupdeluxe-x86_64-win64.exe](https://github.com/LongDirtyAnimAlf/fpcupdeluxe/releases/download/v2.4.0e/fpcupdeluxe-x86_64-win64.exe) do gerenciador de versões [fpcupdeluxe](https://github.com/LongDirtyAnimAlf/fpcupdeluxe/releases). O `fpcupdeluxe` será usado para instalar a IDE e o compilador.
4. Copie o arquivo [fpcupdeluxe-x86_64-win64.exe] para a pasta `fpcupdeluxe`
5. Abra o arquivo [fpcupdeluxe-x86_64-win64.exe]
6. Clique sobre o botão `Set install path` e escolha a pasta `C:\lazarus-trunk\`
7. Na lista `FPC version`, selecione a opção `trunk`
8. Na lista `Lazarus version`, selecione a opção `trunk`
9. Clique no botão `Setup+` e marque a opção `Include help` na lista `Miscellaneous settings`
10. Clique sobre o botão `Install/update FPC+Lazarus` e aguarde a conclusão.

## Passo 2 (4 min) - Tema Escuro Monokai no Lazarus (opcional)
1. Baixe o arquivo `Monokai.xml` com o tema [Monokai](https://wiki.freepascal.org/UserSuppliedSchemeSettings)
2. Copie o arquivo `Monokai.xml` para a pasta `C:\lazarus-trunk\config_lazarus\userschemes`
3. Abra o Lazarus
4. Navegue para Tools>Options>Editor>Display>Colors e escolha `Monokai` no drop down `Color Schemes`
5. Navegue para Lazarus>Tools>Options>Editor>Display fonte para `Consolas`, tamanho 9
6. Instale o pacote [metadarkstyle](https://github.com/zamtmn/metadarkstyle) (dica: você pode usar o Online Package Manager para isso)

## Passo 3 (1 min) - Baixar dependências
O projeto depende de três bibliotecas

- zmq 4.x (https://zeromq.org/)
- Eye Link (https://www.sr-research.com/support/forum-9.html, exige registro)
- SDL2 (https://github.com/libsdl-org/SDL/releases)

1. Você pode baixar as dlls [aqui](https://drive.google.com/drive/folders/1DVSJrth2xP6rerUs1RnUYDRQWJoM7YhA?usp=sharing)


## Passo 4 (20 min) - Git Bash, Stimulus Control
1. Baixe e instale o Git para windows https://git-scm.com/download/win
2. Abra o `Git Bash` (prompt de comando)
3. Clone o repositório recursivamente executando o comando (dica: use apenas `SHIFT+Insert` para colar texto no prompt):
    ```
    mkdir sc
    git clone --recursive https://github.com/cpicanco/stimulus-control-sdl2.git sc
    ```
4. Copie as `dlls` do Passo 4 para a pasta `sc`
5. Abra o Lazarus (sempre usando o atalho criado pelo `fpcupdeluxe` em sua área de trabalho)
6. Instale o pacote `Online Package Manager`
   1. Naveque para Package>Install/Uninstall Packages
   2. Escreva `online` na caixa de busca à direita
   3. Clique sobre `OnlinePackageManager 1.x.x.x`
   4. Clique sobre `Install selection`
   5. Clique sobre `Rebuil IDE`
7. Navegue para Project>Open Project
8. Abra o projeto `experiment.lpi` localizado na pasta `sc`
9. Clique em `Yes` para instalar os pacotes que estiverem faltando (rgbabitmap and synaser)
10. Clique no botão (drop down) Change Build Mode, o engrenagem, para escolher a build desejada.
11. Pressione F9 ou clique no botão `Run`, a seta verde, para compilar.