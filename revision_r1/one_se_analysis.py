"""One-standard-error rule para seleção de componentes do PLS.

Análise para revisão do manuscrito CHEMOLAB-D-26-00658R1 (Reviewer #2, Comment 1).

Reaproveita o pipeline exato do artigo fundador:
- Pickles gerados por src/data/preprocessing_multi.py
- Split 80/20 com random_state=42
- Z-score em X e y (StandardScaler)
- baseline_pls.py-compatível (y_train[:, 0], PLSRegression default)

Este pipeline foi validado por reproduzir:
  Tecator PLS test R² = 0.9191 (paper reporta 0.919)
  Gasoline PLS test R² = 0.9414 (paper reporta 0.941)

Uso:
    cd ~/Downloads/pgsg_0_original
    python one_se_analysis.py 2>&1 | tee one_se_output.txt
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score


PROCESSED_DIR = Path("data/processed")
H_RANGE = list(range(1, 16))


# ============================================================================


def load_pickle(dataset_name: str) -> dict:
    path = PROCESSED_DIR / f"{dataset_name}_processed.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def loo_cv_curve(
    X: np.ndarray, y: np.ndarray, H_range=H_RANGE
) -> dict[int, tuple[float, float]]:
    """Curva LOO-CV para PLS com H em H_range.

    Retorna dict[H] = (RMSE, SE_do_RMSE).

    SE(RMSE) via delta method: SE(RMSE) = SE(MSE) / (2 * RMSE)
    onde SE(MSE) = std_amostra(e^2) / sqrt(n).
    """
    loo = LeaveOneOut()
    n = len(y)
    results: dict[int, tuple[float, float]] = {}

    for H in H_range:
        errs_sq = np.zeros(n)
        for i, (tr, va) in enumerate(loo.split(X)):
            m = PLSRegression(n_components=H)
            m.fit(X[tr], y[tr])
            y_hat = m.predict(X[va]).ravel()[0]
            errs_sq[i] = (y[va][0] - y_hat) ** 2

        mse = float(errs_sq.mean())
        rmse = float(np.sqrt(mse))
        se_mse = float(errs_sq.std(ddof=1) / np.sqrt(n))
        se_rmse = se_mse / (2 * rmse) if rmse > 0 else 0.0
        results[H] = (rmse, se_rmse)

    return results


def one_se_pick(results: dict[int, tuple[float, float]]) -> tuple[int, int, float]:
    H_min = min(results, key=lambda h: results[h][0])
    rmse_min, se_min = results[H_min]
    threshold = rmse_min + se_min
    H_star = min(h for h in results if results[h][0] <= threshold)
    return H_min, H_star, threshold


def kfold_r2_with_saved_splits(
    data: dict, H: int
) -> tuple[float, float, list[float]]:
    """5-fold CV usando OS MESMOS splits salvos no pickle."""
    X_train = data["X_train"]
    y_train = data["y_train"][:, 0]
    cv_splits = data["cv_splits"]

    r2s: list[float] = []
    for fold in cv_splits:
        tr = fold["train_idx"]
        va = fold["val_idx"]
        m = PLSRegression(n_components=H).fit(X_train[tr], y_train[tr])
        y_hat = m.predict(X_train[va]).ravel()
        r2s.append(r2_score(y_train[va], y_hat))

    return float(np.mean(r2s)), float(np.std(r2s)), r2s


# ============================================================================


def analyze(dataset_name: str, run_kfold: bool = True) -> None:
    print("=" * 72)
    print(f" {dataset_name.upper()}  (mesma pipeline do artigo fundador)")
    print("=" * 72)

    data = load_pickle(dataset_name)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"][:, 0]
    y_test = data["y_test"][:, 0]

    print(f"\n  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  y_train range (normalizado): "
          f"[{y_train.min():.3f}, {y_train.max():.3f}]")

    # --- LOO-CV no treino, one-SE rule --------------------------------------
    print(f"\n  LOO-CV curve on training set (H = 1..15):")
    results = loo_cv_curve(X_train, y_train)
    H_min, H_star, thresh = one_se_pick(results)

    for H in H_RANGE:
        r, s = results[H]
        marks = []
        if H == H_min:
            marks.append("H_min")
        if H == H_star:
            marks.append("H*")
        mark = f"  <-- {' & '.join(marks)}" if marks else ""
        print(f"    H = {H:2d}:  RMSECV = {r:.4f}  +/-  {s:.4f}{mark}")

    rmse_min, se_min = results[H_min]
    print(f"\n  RESULTS:")
    print(f"    H_min (min RMSECV)       = {H_min}")
    print(f"    RMSECV(H_min)            = {rmse_min:.4f}")
    print(f"    SE(H_min)                = {se_min:.4f}")
    print(f"    Threshold = RMSE + SE    = {thresh:.4f}")
    print(f"    H*  (one-SE rule)        = {H_star}")

    # --- test-set performance com H_min, H*, e H=10 (do paper) --------------
    def score_H(H: int) -> tuple[float, float]:
        m = PLSRegression(n_components=H).fit(X_train, y_train)
        y_hat = m.predict(X_test).ravel()
        rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
        r2 = float(r2_score(y_test, y_hat))
        return r2, rmse

    r2_10, rmse_10 = score_H(10)
    r2_H_min, rmse_H_min = score_H(H_min)
    r2_H_star, rmse_H_star = score_H(H_star)

    print(f"\n  Single-split test performance (normalized scale):")
    print(f"    H = 10 (paper):      R2 = {r2_10:.4f}   RMSE = {rmse_10:.4f}")
    print(f"    H = {H_min} (argmin):     R2 = {r2_H_min:.4f}   RMSE = {rmse_H_min:.4f}")
    print(f"    H = {H_star} (one-SE):     R2 = {r2_H_star:.4f}   RMSE = {rmse_H_star:.4f}")

    # --- 5-fold CV com os folds salvos no pickle ----------------------------
    if run_kfold:
        print(f"\n  5-fold CV using saved cv_splits (protocol of section 5.2):")
        for H in sorted({10, H_min, H_star}):
            r2_cv, r2_std, r2s = kfold_r2_with_saved_splits(data, H)
            r2s_str = "  ".join(f"{r:.4f}" for r in r2s)
            print(f"    H = {H:2d}:  R2 = {r2_cv:.4f}  +/-  {r2_std:.4f}"
                  f"   (per-fold: {r2s_str})")

    print()


# ============================================================================


def main() -> None:
    print("\nOne-standard-error rule para selecao de H no PLS")
    print("Manuscrito CHEMOLAB-D-26-00658R1, Reviewer #2, Comment 1\n")

    if not (PROCESSED_DIR / "tecator_processed.pkl").exists():
        print("ERRO: rode primeiro:")
        print("  python src/data/preprocessing_multi.py --dataset tecator")
        print("  python src/data/preprocessing_multi.py --dataset gasoline")
        return

    analyze("tecator", run_kfold=True)
    analyze("gasoline", run_kfold=True)


if __name__ == "__main__":
    main()
