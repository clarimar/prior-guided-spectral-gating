#!/usr/bin/env python
"""
Reexecução do pgsg_1 extraindo os vetores de gate — colunas ρ e Jaccard.

MOTIVO
------
A reexecução anterior (rerun_pgsg1_seeds.py) salvou apenas R² e RMSE, de modo
que as colunas ρ(g,s) e Jaccard da Tabela 2 continuavam com as 5 sementes
originais enquanto todas as demais passaram a 10. Numa carta-resposta cujo
tema é replicação inadequada, essa heterogeneidade é constrangedora.

Este script repete o protocolo salvando os gates, recomputa as duas métricas
com 10 sementes e grava os vetores em .npz — que servem também ao pgsg_3,
onde a distância entre gate e prior é o objeto de estudo.

MÉTRICAS (definições do manuscrito, §4.4)
-----------------------------------------
ρ(g,s)  : correlação de Pearson entre o gate aprendido e o prior, média
          sobre as sementes.
Jaccard : sobreposição média do decil superior de bandas (q = ceil(0.1·p)
          = 28 para p = 281) entre TODOS os pares de sementes no mesmo n.

USO
---
    python scripts/rerun_gates.py \
        --pgsg1-root ~/Dropbox/pgsg/pgsg_1 \
        --n-grid 30,60,100,200,400,700,1159 --seeds 10 \
        --out results/rerun_gates

Rodar só o ponto publicado, para conferir rápido:
    python scripts/rerun_gates.py --n-grid 1159 --seeds 10
"""

from __future__ import annotations

import argparse
import csv as csvmod
import importlib.util
import itertools
import os
import sys
from pathlib import Path

import numpy as np

SEP = "=" * 74

# valores publicados (5 sementes), para comparação
PUB_5 = {
    30: (1.00, 0.97), 60: (1.00, 0.96), 100: (1.00, 0.97), 200: (1.00, 1.00),
    400: (1.00, 0.97), 700: (1.00, 0.95), 1159: (1.00, 1.00),
}


def head(t):
    print(f"\n{SEP}\n{t}\n{SEP}")


def carregar_modulo(path: Path, nome: str):
    spec = importlib.util.spec_from_file_location(nome, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


def achar_csv(root: Path):
    achados = []
    for pat in ("*angoDMC*.csv", "*mango*.csv", "*Mango*.csv", "*dmc*.csv", "*DMC*.csv"):
        achados.extend(root.rglob(pat))
    achados = [a for a in achados if ".git" not in a.parts]
    return max(achados, key=lambda p: p.stat().st_size) if achados else None


# ------------------------------------------------------------------ gate
def extrair_gate(modelo, p_esperado: int) -> np.ndarray:
    """Recupera o vetor de gate g ∈ (0,1)^p do modelo treinado.

    O nome do atributo não é conhecido a priori; tentamos os candidatos
    plausíveis e, se nenhum servir, falhamos listando o que existe — em vez
    de devolver silenciosamente algo errado.
    """
    # 0) property (acesso direto, sem chamada) — é o caso da PGSGv2Model
    for nome in ("gates", "gate", "gate_"):
        try:
            v = getattr(modelo, nome)
        except Exception:
            continue
        if v is None or callable(v):
            continue
        try:
            g = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v).ravel()
        except Exception:
            continue
        if g.shape[0] == p_esperado:
            return g

    # 1) método dedicado
    for nome in ("get_gate", "gate_vector"):
        fn = getattr(modelo, nome, None)
        if callable(fn):
            try:
                g = np.asarray(fn()).ravel()
                if g.shape[0] == p_esperado:
                    return g
            except Exception:
                pass

    # 2) atributo já em espaço (0,1)
    for nome in ("gate_", "gate", "gates_", "g_", "g"):
        v = getattr(modelo, nome, None)
        if v is None:
            continue
        try:
            g = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v).ravel()
        except Exception:
            continue
        if g.shape[0] == p_esperado and g.min() >= 0.0 and g.max() <= 1.0:
            return g

    # 3) logits θ -> sigmoide
    for nome in ("theta_", "theta", "logits_", "logits"):
        v = getattr(modelo, nome, None)
        if v is None:
            continue
        try:
            t = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v).ravel()
        except Exception:
            continue
        if t.shape[0] == p_esperado:
            return 1.0 / (1.0 + np.exp(-t))

    # 4) dentro de um módulo torch aninhado
    for attr in ("model_", "model", "net_", "net", "module_"):
        sub = getattr(modelo, attr, None)
        if sub is None:
            continue
        for nome, val in list(vars(sub).items()) if hasattr(sub, "__dict__") else []:
            if not hasattr(val, "detach"):
                continue
            t = np.asarray(val.detach().cpu().numpy()).ravel()
            if t.shape[0] == p_esperado:
                if t.min() >= 0.0 and t.max() <= 1.0:
                    return t
                return 1.0 / (1.0 + np.exp(-t))
        if hasattr(sub, "named_parameters"):
            for nome, par in sub.named_parameters():
                t = np.asarray(par.detach().cpu().numpy()).ravel()
                if t.shape[0] == p_esperado:
                    return 1.0 / (1.0 + np.exp(-t))

    disponiveis = sorted(a for a in dir(modelo) if not a.startswith("__"))
    raise SystemExit(
        "não foi possível extrair o gate do modelo treinado.\n"
        f"esperado vetor de tamanho {p_esperado}. atributos disponíveis:\n  "
        + ", ".join(disponiveis)
    )


# --------------------------------------------------------------- métricas
def rho_pearson(g: np.ndarray, s: np.ndarray) -> float:
    return float(np.corrcoef(g, s)[0, 1])


def topo_q(g: np.ndarray, q: int) -> set[int]:
    return set(np.argsort(g)[::-1][:q].tolist())


def jaccard_par(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def jaccard_medio(gates: list[np.ndarray], q: int) -> float:
    """Média do Jaccard do decil superior sobre TODOS os pares de sementes."""
    topos = [topo_q(g, q) for g in gates]
    pares = [jaccard_par(a, b) for a, b in itertools.combinations(topos, 2)]
    return float(np.mean(pares)) if pares else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgsg1-root", default="~/Dropbox/pgsg/pgsg_1")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model-file", default=None)
    ap.add_argument("--season", type=int, default=4)
    ap.add_argument("--n-grid", default="30,60,100,200,400,700,1159")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--decil", type=float, default=0.10,
                    help="fração superior de bandas para o Jaccard (0.10 = decil)")
    ap.add_argument("--out", default="results/rerun_gates")
    a = ap.parse_args()

    root = Path(os.path.expanduser(a.pgsg1_root))
    model_file = (Path(os.path.expanduser(a.model_file)) if a.model_file
                  else root / "pgsg_v2.py")
    n_grid = [int(x) for x in a.n_grid.split(",")]
    seeds = list(range(a.seeds))

    head("Setup")
    mod_model = carregar_modulo(model_file, "pgsg_v2_gates")
    sys.path.insert(0, str(root))
    mod = carregar_modulo(root / "run_experiment_v2.py", "run_exp_gates")
    print(f"modelo : {model_file}")

    csv_path = Path(os.path.expanduser(a.csv)) if a.csv else achar_csv(root)
    if csv_path is None:
        raise SystemExit(f"CSV não encontrado sob {root}; use --csv")
    print(f"csv    : {csv_path}")
    print(f"n_grid : {n_grid}   sementes: {seeds}")

    SpectralDataset = mod_model.SpectralDataset
    PGSGv2Model = mod_model.PGSGv2Model

    ds_full = mod.load_mango_dmc_v3(csv_path)
    mask = np.asarray(ds_full.group_ids) == a.season
    ds4 = SpectralDataset(
        X=np.asarray(ds_full.X)[mask], y=np.asarray(ds_full.y)[mask],
        wavelengths=np.asarray(ds_full.wavelengths), metadata=dict(ds_full.metadata),
        group_ids=np.asarray(ds_full.group_ids)[mask],
    )

    gen = mod.ScenarioGenerator(
        test_strategy=mod.FixedFractionTest(fraction=0.2, seed=42),
        train_strategy=mod.StratifiedTrainSampler(),
        n_grid=n_grid, seeds=seeds,
    )
    gen.fit(ds4)
    cenarios = list(gen.iter_scenarios())
    print(f"cenários: {len(cenarios)}")

    head("Treinando e extraindo gates")
    # coleta[(n, condicao)] = {"gates": [...], "rhos": [...]}
    coleta: dict[tuple[int, str], dict] = {}
    prior_ref: dict[int, np.ndarray] = {}
    wl_ref: dict[int, np.ndarray] = {}

    for i, sc in enumerate(cenarios, 1):
        prep = mod.Preprocessor(drop_zero_bands=True, apply_snv=True, normalize_target=False)
        X_tr, y_tr = prep.fit_transform(sc.train_dataset)
        X_te, y_te = prep.transform(sc.test_dataset)
        wl = np.asarray(prep.params.kept_wavelengths)
        tr = SpectralDataset(X=X_tr, y=y_tr, wavelengths=wl,
                             metadata=dict(sc.train_dataset.metadata))
        prior = np.asarray(mod.make_literature_prior(wl))
        p = X_tr.shape[1]
        prior_ref.setdefault(sc.n_actual, prior)
        wl_ref.setdefault(sc.n_actual, wl)

        for cond, pri in (("lit", prior), ("rand", None)):
            m = PGSGv2Model(hidden=32, max_epochs=500, patience=30, seed=sc.seed)
            m.fit(tr, prior=pri)
            g = extrair_gate(m, p)
            ch = (sc.n_actual, cond)
            coleta.setdefault(ch, {"gates": [], "rhos": [], "seeds": []})
            coleta[ch]["gates"].append(g)
            coleta[ch]["rhos"].append(rho_pearson(g, prior))
            coleta[ch]["seeds"].append(sc.seed)

        r_lit = coleta[(sc.n_actual, "lit")]["rhos"][-1]
        print(f"  [{i:>3}/{len(cenarios)}] n={sc.n_actual:<5} seed={sc.seed}  "
              f"p={p}  ρ_lit={r_lit:.4f}")

    # ------------------------------------------------------------- saída
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    linhas = []
    for n in sorted({k[0] for k in coleta}):
        p = len(prior_ref[n])
        q = int(np.ceil(a.decil * p))
        reg = {"n_actual": n, "p": p, "q": q}
        for cond in ("lit", "rand"):
            d = coleta[(n, cond)]
            rhos = np.asarray(d["rhos"])
            reg[f"rho_{cond}"] = rhos.mean()
            reg[f"rho_{cond}_sd"] = rhos.std(ddof=1) if len(rhos) > 1 else 0.0
            reg[f"jaccard_{cond}"] = jaccard_medio(d["gates"], q)
            reg[f"k_{cond}"] = len(rhos)
        linhas.append(reg)

        np.savez_compressed(
            out / f"gates_n{n}.npz",
            gates_lit=np.stack(coleta[(n, "lit")]["gates"]),
            gates_rand=np.stack(coleta[(n, "rand")]["gates"]),
            prior=prior_ref[n], wavelengths=wl_ref[n],
            seeds=np.asarray(coleta[(n, "lit")]["seeds"]),
        )

    campos = list(linhas[0].keys())
    with (out / "gate_metrics.csv").open("w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=campos)
        w.writeheader(); w.writerows(linhas)

    head("ρ e Jaccard com 10 sementes")
    print(f"{'n':>6} {'ρ_lit':>16} {'Jacc_lit':>9} {'ρ_rand':>9} {'Jacc_rand':>10} "
          f"{'publicado (5)':>16}")
    for r in linhas:
        pub = PUB_5.get(r["n_actual"]) or PUB_5.get(1159)
        print(f"{r['n_actual']:>6} {r['rho_lit']:>9.4f}±{r['rho_lit_sd']:.4f} "
              f"{r['jaccard_lit']:>9.4f} {r['rho_rand']:>9.4f} {r['jaccard_rand']:>10.4f} "
              f"{f'ρ={pub[0]:.2f} J={pub[1]:.2f}':>16}")

    print(f"\ngravado: {out}/gate_metrics.csv")
    print(f"         {out}/gates_n*.npz  (vetores de gate, reutilizáveis no pgsg_3)")

    head("Leitura")
    print("  Se ρ_lit ≈ 1.00 e Jaccard_lit ≈ 0.95–1.00, os valores publicados se")
    print("  confirmam com 10 sementes e a Tabela 2 fica homogênea.")
    print("  A coluna rand é o contraste: espera-se Jaccard bem menor, o mesmo")
    print("  efeito que o H4 do pgsg_2 documenta (0.60 vs 0.94 em NIR).")


if __name__ == "__main__":
    main()
