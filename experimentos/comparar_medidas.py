"""
Experimento fase 2 (revisión P1): compara la similitud híbrida con otras medidas que
también miden el perfil, y comprueba si el problema es lo bastante difícil.

- Medidas: hybrid estandarizada (la propuesta), hybrid sin normalizar (como en producción),
  Gower clásica, euclídea (numéricas estandarizadas + one-hot), coseno solo numéricas
  y volumen (solo como referencia, mide tamaño).
- Dificultad: se regeneran los clientes acercando los parámetros de cada perfil a la media
  global (lambda = 0, 0.5, 0.75). Con lambda alto los perfiles se parecen más entre sí.

Precision@10 sobre 300 clientes al azar, comparando cada uno contra los 10.000.

Uso: python experimentos/comparar_medidas.py
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import generar_datos as gd  # noqa: E402
from simsys import DataPreprocessor  # noqa: E402

OUT = config.OUTPUT_DIR / "fase2"
K, N_QUERIES, SEED = 10, 300, 42


def preparar(df, id_col, excluir, norm):
    p = DataPreprocessor(norm, ["hybrid"], id_col, excluir)
    p.fit_transform(df)
    num, cat = p.columns_of_type("numeric"), p.columns_of_type("categorical", "boolean")
    X = p.processed_data
    Xn = X[num].to_numpy(np.float64)
    Xc = np.column_stack([pd.factorize(X[c].astype(str))[0] for c in cat]) if cat else np.zeros((len(X), 0))
    return X.index.to_numpy(), Xn, Xc


def sim_hybrid(Xn, Xc, q):
    a = len(Xn[0]) / (Xn.shape[1] + Xc.shape[1])
    nq, na = np.linalg.norm(Xn[q], axis=1), np.linalg.norm(Xn, axis=1)
    den = np.outer(nq, na)
    cos = np.divide(Xn[q] @ Xn.T, den, out=np.zeros_like(den), where=den != 0)
    gow = (Xc[q][:, None, :] != Xc[None, :, :]).mean(axis=2)
    return 1 - (a * (1 - cos) + (1 - a) * gow)


def sim_gower(Xraw, Xc, q):
    rng_ = Xraw.max(0) - Xraw.min(0)
    rng_[rng_ == 0] = 1
    s_num = np.zeros((len(q), len(Xraw)))
    for j in range(Xraw.shape[1]):
        s_num += 1 - np.abs(Xraw[q, j][:, None] - Xraw[None, :, j]) / rng_[j]
    s_cat = (Xc[q][:, None, :] == Xc[None, :, :]).sum(2)
    return (s_num + s_cat) / (Xraw.shape[1] + Xc.shape[1])


def sim_euclid(Xn, Xc, q):
    oh = pd.get_dummies(pd.DataFrame(Xc).astype(str)).to_numpy(float) / np.sqrt(2)
    Z = np.hstack([Xn, oh])
    sq = (Z ** 2).sum(1)
    return -(sq[q][:, None] + sq[None, :] - 2 * Z[q] @ Z.T)


def sim_coseno(Xn, Xc, q):
    nq, na = np.linalg.norm(Xn[q], axis=1), np.linalg.norm(Xn, axis=1)
    den = np.outer(nq, na)
    return np.divide(Xn[q] @ Xn.T, den, out=np.zeros_like(den), where=den != 0)


def sim_volume(Xmm, Xc, q):
    n = np.linalg.norm(Xmm, axis=1)
    a, b = n[q][:, None], n[None, :]
    mx = np.maximum(a, b)
    return np.divide(np.minimum(a, b), mx, out=np.ones((len(q), len(n))), where=mx != 0)


def precision_at_k(S, q, y):
    S = S.copy()
    S[np.arange(len(q)), q] = -np.inf
    top = np.argsort(-S, axis=1, kind="stable")[:, :K]
    prec = (y[top] == y[q][:, None]).mean()
    sizes = pd.Series(y).value_counts()
    azar = np.mean([(sizes[y[i]] - 1) / (len(y) - 1) for i in q])
    return prec, azar


def evaluar(df, id_col, seg_col, excluir):
    ids, Xs, Xc = preparar(df, id_col, excluir, "StandardScaler")
    _, Xraw, _ = preparar(df, id_col, excluir, None)
    _, Xmm, _ = preparar(df, id_col, excluir, "MinMaxScaler")
    y = df.set_index(id_col).loc[ids, seg_col].to_numpy()
    q = np.random.default_rng(SEED).choice(len(ids), N_QUERIES, replace=False)
    filas = []
    medidas = [
        ("Híbrida estandarizada (propuesta)", lambda: sim_hybrid(Xs, Xc, q)),
        ("Híbrida sin normalizar (producción)", lambda: sim_hybrid(Xraw, Xc, q)),
        ("Gower clásica", lambda: sim_gower(Xraw, Xc, q)),
        ("Euclídea (estandarizada + one-hot)", lambda: sim_euclid(Xs, Xc, q)),
        ("Coseno solo numéricas", lambda: sim_coseno(Xs, Xc, q)),
        ("Volumen (referencia)", lambda: sim_volume(Xmm, Xc, q)),
    ]
    for nombre, f in medidas:
        p, a = precision_at_k(f(), q, y)
        filas.append({"MEDIDA": nombre, "PRECISION@10": round(p, 3), "AZAR": round(a, 3), "MEJORA": round(p / a, 1)})
    return pd.DataFrame(filas)


def clientes_con_dificultad(lam, n=10_000, seed=42):
    """Acerca los parámetros de cada perfil a la media ponderada de todos (lam=0: original)."""
    orig = copy.deepcopy(gd.PERFILES_CLIENTES)
    P = orig
    w = np.array([p["peso"] for p in P.values()])
    w = w / w.sum()
    media = {
        "edad": float(np.sum(w * [p["edad"][0] for p in P.values()])),
        "live": float(np.sum(w * [p["live"] for p in P.values()])),
        "prod": np.sum(w[:, None] * np.array([p["prod"] for p in P.values()]), axis=0),
    }
    for k in ("acc", "dep", "rec", "ant"):  # magnitudes: media geométrica
        media[k] = float(np.exp(np.sum(w * np.log([p[k] for p in P.values()]))))
    nuevos = {}
    for nombre, p in P.items():
        q = dict(p)
        q["edad"] = ((1 - lam) * p["edad"][0] + lam * media["edad"], p["edad"][1])
        q["live"] = (1 - lam) * p["live"] + lam * media["live"]
        q["prod"] = tuple((1 - lam) * np.array(p["prod"]) + lam * media["prod"])
        for k in ("acc", "dep", "rec", "ant"):
            q[k] = float(np.exp((1 - lam) * np.log(p[k]) + lam * np.log(media[k])))
        nuevos[nombre] = q
    gd.PERFILES_CLIENTES = nuevos
    try:
        return gd.generar_clientes(n, seed)
    finally:
        gd.PERFILES_CLIENTES = orig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for proy in ("CLIENTES", "LOCALES"):
        c = config.PROYECTOS[proy]
        df = pd.read_excel(c["fichero"])
        r = evaluar(df, c["id"], c["columna_validacion"], c["excluir"])
        r.insert(0, "PROYECTO", proy)
        res.append(r)
        print(r.to_string(index=False), "\n")
    pd.concat(res).to_csv(OUT / "comparacion_medidas.csv", index=False)

    c = config.PROYECTOS["CLIENTES"]
    dif = []
    for lam in (0.0, 0.5, 0.75):
        df = clientes_con_dificultad(lam)
        r = evaluar(df, c["id"], c["columna_validacion"], c["excluir"])
        r.insert(0, "LAMBDA", lam)
        dif.append(r)
        print(f"lambda={lam}\n", r.to_string(index=False), "\n")
    pd.concat(dif).to_csv(OUT / "dificultad.csv", index=False)


if __name__ == "__main__":
    main()
