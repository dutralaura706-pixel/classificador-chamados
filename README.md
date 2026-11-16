# Classificador de chamados de suporte

Projeto que fiz para praticar Python e aprendizado de máquina. A ideia é simples: o programa lê o texto de um chamado de suporte técnico e diz se ele é de *rede, **acesso, **hardware* ou *software*.

## Os dados

Usei 76 chamados fictícios, 19 de cada categoria, criados para este estudo. Estão no arquivo chamados.csv, com duas colunas: texto e categoria.

## O que eu fiz

1. Li e limpei os dados com pandas. Descobri que espaços escondidos no fim do texto criavam categorias duplicadas, então passei a padronizar tudo.
2. Separei 80% dos chamados para treino e 20% para teste, mantendo a mesma proporção de cada categoria.
3. Transformei os textos em números com TF-IDF.
4. Treinei uma regressão logística com scikit-learn.
5. Avaliei no teste e olhei os chamados que o modelo errou.

## Resultado

O modelo acertou *14 de 16* chamados de teste (87,5%). Também testei 4 frases novas, que não estavam nos dados, e ele acertou as 4.

Os dois erros foram um chamado de software e um de hardware classificados como acesso. Pelo que vi, o chamado de software tinha palavras como "sistema" e "chamados", que aparecem bastante nos exemplos de acesso. O de hardware falava de teclado molhado, e eu tinha poucos exemplos assim.

## Limitações

Os dados são poucos e fictícios, e o teste tem só 16 chamados. Um erro a mais ou a menos muda bastante a acurácia, então não dá para tratar esse número como definitivo.

Quero continuar com mais exemplos, validação cruzada e uma interface simples para digitar um chamado e ver a categoria.

## Como rodar

bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python treinar.py


Feito com Python, pandas e scikit-learn.