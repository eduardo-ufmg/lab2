# Relatórios, Códigos e Dados para as atividades na disciplina ELT-015: Laboratório de Controle e Automação II

## Autores

- Eduardo Henrique Basilio de Carvalho
- João Vitor Braga da Silva Alves
- Leticia Vitoria Martins do Carmo
- Renan Neves da Silva

## Recursos e diretrizes

- Todos os relatórios devem ter, como base, o modelo disponível no diretório [modelo](./modelo/). Qualquer mudança pode ser feita, em cópia local a cada atividade, conforme necessário.
- A estrutura atual garante suporte à extensão [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) para Visual Studio Code no Ubuntu 24.
- Todas as atividades têm duas partes: experimento e processamento.
    - No experimento, o sistema da bancada correspondente é usado. Este pode ser um computador do qual temos acesso limitado aos recursos, um CLP, um sistema distribuído, ou qualquer outro ambiente que não podemos configurar livremente. Por isso, este repositório não contém necessariamente os códigos, configurações e dados de entrada dos experimentos, nem tem a responsabilidade de gerenciar sua execução, manutenção e versionamento. São responsáveis pelo experimento as próprias bancadas.
    - No processamento, os dados coletados no experimento são processados, analisados e apresentados. Esta parte é feita localmente, e é a única parte que deve ser gerenciada por este repositório. Assim, os códigos, configurações e dados do processamento devem ser armazenados neste repositório.
- O processamento é feito, a princípio, em Python.

## Configuração do Ambiente

Veja o [repositório pai](https://github.com/eduardo-ufmg/s20261/).
