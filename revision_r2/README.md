# revision_r2 — reexecução com réplicas genuínas

Correção iniciada pelos autores após a submissão da R1.

## Problema

As fábricas de modelo em `run_experiment_v2.py` fixavam `seed=42`. As cinco
sementes variavam só a amostragem do subconjunto de treino. Em n=1159 o
tamanho pedido iguala o pool, então o amostrador devolve sempre o mesmo
subconjunto — as cinco réplicas eram execuções idênticas (verificável em
`results_v2/pred_results.csv`, valores iguais até o 6º decimal).

## Conteúdo

- `pgsg_1_R2.tex` — manuscrito revisado
- `carta_resposta_correcoes.tex` — correções para os revisores
- `scripts/rerun_pgsg1_seeds.py` — reexecução com seed do modelo = seed do cenário
- `scripts/reconcile_prior_ablation.py` — diagnóstico que identificou o problema
- `results/rerun_curva_completa.csv` — 70 cenários x 5 modelos, 10 sementes
- `results/rerun_pgsg1_seeds.csv` — ponto n=1159 isolado

## Resultado principal

Δ_prior não é significativo em nenhum tamanho de treino (Wilcoxon pareado,
10 sementes). O prior de literatura não melhora acurácia; seu valor é a
reprodutibilidade do gate.

## Pendências

- Figuras 1 e 2 ainda não regeradas a partir de `rerun_curva_completa.csv`
- Colunas ρ e Jaccard da Tabela 2 ainda com 5 sementes (a reexecução salvou
  apenas R² e RMSE; os vetores de gate precisam ser extraídos)
