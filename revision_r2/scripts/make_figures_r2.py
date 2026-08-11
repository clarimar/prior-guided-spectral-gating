#!/usr/bin/env python
"""
Figuras 1 e 2 revisadas — pgsg_1 (R2).

Regera as duas figuras a partir de rerun_curva_completa.csv (10 sementes
genuínas), substituindo as versões baseadas em 5 réplicas degeneradas.

MUDANÇAS EM RELAÇÃO ÀS FIGURAS ANTERIORES
-----------------------------------------
Fig. 1: faixa sombreada passa a ser ±1 DP sobre as 10 sementes, no lugar do
        IC bootstrap (que na última linha reamostrava 5 cópias do mesmo
        número). O painel (b) não marca mais um cruzamento único em n*≈95:
        a curva não é monotônica nessa região.
Fig. 2: painel (a) com os sinais corretos de Δ_PLS; painel (b) com a
        diferença PAREADA por semente e barras de erro, marcando os pontos
        com p<0.05 no teste de Wilcoxon. Painéis (c) e (d) mantêm ρ e
        Jaccard das 5 sementes originais — a reexecução salvou apenas R² e
        RMSE, não os vetores de gate. Isso é sinalizado no próprio eixo.

USO
---
    python scripts/make_figures_r2.py \
        --csv results/rerun_curva_completa.csv \
        --outdir revision_r2/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ρ e Jaccard não foram remedidos na reexecução (só R² e RMSE foram salvos).
# Valores das 5 sementes originais, mantidos e sinalizados como tal.
RHO_5SEEDS = {30: 1.00, 60: 1.00, 100: 1.00, 200: 1.00, 400: 1.00, 700: 1.00, 1159: 1.00}
JACC_5SEEDS = {30: 0.97, 60: 0.96, 100: 0.97, 200: 1.00, 400: 0.97, 700: 0.95, 1159: 1.00}

CORES = {
    "PGSGv2": "#1f4e79",
    "PGSGv2-random": "#5b9bd5",
    "PLS": "#c0392b",
    "MLP": "#27ae60",
    "CNN1D-shallow": "#e67e22",
}
ROTULOS = {
    "PGSGv2": "PGSGv2 (lit. prior)",
    "PGSGv2-random": "PGSGv2 (uninformed)",
    "PLS": "PLS",
    "MLP": "MLP",
    "CNN1D-shallow": "CNN1D-shallow",
}
MARCAS = {"PGSGv2": "o", "PGSGv2-random": "s", "PLS": "^", "MLP": "D", "CNN1D-shallow": "v"}


def carregar(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    faltando = {"n_actual", "seed", "model_name", "r2"} - set(df.columns)
    if faltando:
        raise SystemExit(f"colunas ausentes no CSV: {sorted(faltando)}")
    return df


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model_name", "n_actual"])["r2"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def delta_prior_pareado(df: pd.DataFrame) -> pd.DataFrame:
    """Diferença lit - rand por semente, com Wilcoxon."""
    piv = df[df.model_name.isin(["PGSGv2", "PGSGv2-random"])].pivot_table(
        index=["n_actual", "seed"], columns="model_name", values="r2"
    ).dropna()
    linhas = []
    for n, g in piv.groupby(level=0):
        d = (g["PGSGv2"] - g["PGSGv2-random"]).values
        try:
            _, p = wilcoxon(d)
        except ValueError:
            p = np.nan
        linhas.append({"n_actual": n, "delta": d.mean(),
                       "sd": d.std(ddof=1) if len(d) > 1 else 0.0,
                       "p": p, "k": len(d)})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- Figura 1
def figura1(res: pd.DataFrame, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ns = sorted(res.n_actual.unique())
    for modelo in ["PGSGv2", "PGSGv2-random", "PLS", "MLP", "CNN1D-shallow"]:
        sub = res[res.model_name == modelo].sort_values("n_actual")
        if sub.empty:
            continue
        x, y, sd = sub.n_actual.values, sub["mean"].values, sub["std"].values
        for ax in (ax1, ax2):
            ax.plot(x, y, marker=MARCAS[modelo], ms=5, lw=1.6,
                    color=CORES[modelo], label=ROTULOS[modelo])
            ax.fill_between(x, y - sd, y + sd, color=CORES[modelo], alpha=0.15, lw=0)

    # (a) escala completa
    ax1.axhline(0, color="0.4", ls="--", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns], fontsize=8)
    ax1.set_xlabel("Training set size $n$")
    ax1.set_ylabel("$R^2$ (test set, mean $\\pm$ 1 SD, 10 seeds)")
    ax1.set_title("(a) Full range", fontsize=10)
    ax1.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax1.grid(alpha=0.25, lw=0.5)

    # (b) zoom: sem marcação de cruzamento único
    ax2.set_xscale("log")
    grandes = [n for n in ns if n >= 200]
    ax2.set_xlim(min(grandes) * 0.92, max(grandes) * 1.08)
    ax2.set_ylim(0.10, 0.88)
    ax2.set_xticks(grandes)
    ax2.set_xticklabels([str(n) for n in grandes], fontsize=8)
    ax2.set_xlabel("Training set size $n$")
    ax2.set_ylabel("$R^2$")
    ax2.set_title("(b) Zoom: $n \\geq 200$", fontsize=10)
    ax2.grid(alpha=0.25, lw=0.5)
    # sombreia a faixa onde PLS é competitivo, em vez de um n* interpolado
    ax2.axvspan(200, 400, color="0.85", alpha=0.5, zorder=0)
    ax2.text(275, 0.14, "PLS competitive", fontsize=7.5, color="0.35", ha="center")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_scale_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1_scale_curves.pdf/.png")


# ---------------------------------------------------------------- Figura 2
def figura2(res: pd.DataFrame, dp: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    (axa, axb), (axc, axd) = axes

    ns = sorted(res.n_actual.unique())
    xpos = np.arange(len(ns))
    rot = [str(n) for n in ns]

    # (a) Delta_PLS — sinais corretos
    lit = res[res.model_name == "PGSGv2"].set_index("n_actual")["mean"]
    pls = res[res.model_name == "PLS"].set_index("n_actual")["mean"]
    d_pls = np.array([lit[n] - pls[n] for n in ns])
    cores = ["#1f4e79" if v > 0 else "#c0392b" for v in d_pls]
    axa.bar(xpos, d_pls, color=cores, width=0.65)
    axa.axhline(0, color="0.2", lw=0.9)
    axa.set_xticks(xpos); axa.set_xticklabels(rot, fontsize=8)
    axa.set_xlabel("$n$"); axa.set_ylabel("$\\Delta_\\mathrm{PLS}$")
    axa.set_title("(a) H1: $R^2$(PGSGv2-lit) $-$ $R^2$(PLS)", fontsize=10)
    axa.grid(axis="y", alpha=0.25, lw=0.5)
    for i, v in enumerate(d_pls):
        axa.text(i, v + (0.03 if v > 0 else -0.06), f"{v:+.2f}",
                 ha="center", fontsize=7, color="0.25")

    # (b) Delta_prior pareado, com barras de erro
    dp = dp.set_index("n_actual")
    delta = np.array([dp.loc[n, "delta"] for n in ns])
    sd = np.array([dp.loc[n, "sd"] for n in ns])
    pv = np.array([dp.loc[n, "p"] for n in ns])
    cores = ["#1f4e79" if v > 0 else "#e67e22" for v in delta]
    axb.bar(xpos, delta, yerr=sd, color=cores, width=0.65,
            error_kw=dict(ecolor="0.3", lw=1.0, capsize=3))
    axb.axhline(0, color="0.2", lw=0.9)
    axb.set_xticks(xpos); axb.set_xticklabels(rot, fontsize=8)
    axb.set_xlabel("$n$"); axb.set_ylabel("$\\Delta_\\mathrm{prior}$ (paired)")
    axb.set_title("(b) H2: literature prior $-$ uninformed", fontsize=10)
    axb.grid(axis="y", alpha=0.25, lw=0.5)
    for i, (v, s, p) in enumerate(zip(delta, sd, pv)):
        if np.isfinite(p) and p < 0.05:
            axb.text(i, v + s + 0.012, "*", ha="center", fontsize=11, color="0.2")
    axb.text(0.98, 0.04, "* $p<0.05$ (Wilcoxon)", transform=axb.transAxes,
             ha="right", fontsize=7, color="0.35")

    # (c) rho — 5 sementes originais, sinalizado
    rho = [RHO_5SEEDS[n if n in RHO_5SEEDS else 1159] for n in ns]
    axc.plot(xpos, rho, marker="o", ms=5, lw=1.6, color="#1f4e79")
    axc.set_xticks(xpos); axc.set_xticklabels(rot, fontsize=8)
    axc.set_ylim(0.990, 1.002)
    axc.set_xlabel("$n$"); axc.set_ylabel("$\\rho(\\mathbf{g},\\mathbf{s})$  (5 seeds)")
    axc.set_title("(c) H3: gate--prior correlation", fontsize=10)
    axc.grid(alpha=0.25, lw=0.5)

    # (d) Jaccard — 5 sementes originais, sinalizado
    jac = [JACC_5SEEDS[n if n in JACC_5SEEDS else 1159] for n in ns]
    axd.plot(xpos, jac, marker="s", ms=5, lw=1.6, color="#27ae60")
    axd.set_xticks(xpos); axd.set_xticklabels(rot, fontsize=8)
    axd.set_ylim(0.90, 1.02)
    axd.set_xlabel("$n$"); axd.set_ylabel("Jaccard (top 10\\%)  (5 seeds)")
    axd.set_title("(d) H4: gate stability across seeds", fontsize=10)
    axd.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_hypotheses.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2_hypotheses.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/rerun_curva_completa.csv")
    ap.add_argument("--outdir", default="revision_r2/figures")
    a = ap.parse_args()

    df = carregar(Path(a.csv))
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    res = resumo(df)
    dp = delta_prior_pareado(df)

    print(f"CSV: {a.csv}  ({len(df)} linhas, "
          f"{df.seed.nunique()} sementes, {df.n_actual.nunique()} tamanhos)")
    print("\nΔ_prior pareado:")
    for _, r in dp.iterrows():
        marca = " *" if np.isfinite(r.p) and r.p < 0.05 else ""
        print(f"  n={int(r.n_actual):>5}  {r.delta:+.4f} ± {r.sd:.4f}  "
              f"p={r.p:.4f}  (k={int(r.k)}){marca}")

    print("\nFiguras:")
    figura1(res, out)
    figura2(res, dp, out)
    print(f"\nem: {out.resolve()}")
    print("\nNota: painéis (c) e (d) da Fig. 2 ainda vêm das 5 sementes originais.")
    print("A reexecução salvou apenas R² e RMSE, não os vetores de gate.")


if __name__ == "__main__":
    main()
