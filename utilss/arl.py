"""ARL, ATS y simulación Monte Carlo (compartido por los módulos 2 y 6)."""
import numpy as np
from scipy import optimize, stats


def arl_shewhart(delta, n, L=3.0):
    """ARL de un gráfico X̄ con límites ±Lσ/√n y un desplazamiento delta (en σ)."""
    d = np.asarray(delta, float) * np.sqrt(n)
    beta = stats.norm.cdf(L - d) - stats.norm.cdf(-L - d)
    return 1 / (1 - beta)


def n_minimo(delta, L, arl1_max, n_max=500):
    """Menor n tal que ARL1 ≤ arl1_max. Devuelve None si no se logra con n ≤ n_max."""
    ns = np.arange(1, n_max + 1)
    ok = arl_shewhart(delta, ns, L) <= arl1_max
    return int(ns[ok][0]) if ok.any() else None


def arl_ewma(delta, n, lam, L, m=201):
    """ARL de un EWMA bilateral por cadena de Markov (Brook y Evans, 1972)."""
    H = L * np.sqrt(lam / (2 - lam))
    w = 2 * H / m
    centros = -H + (np.arange(m) + 0.5) * w
    sup = -H + (np.arange(m) + 1) * w
    media = (1 - lam) * centros + lam * delta * np.sqrt(n)
    P = (stats.norm.cdf((sup[None, :] - media[:, None]) / lam)
         - stats.norm.cdf((sup[None, :] - w - media[:, None]) / lam))
    arl = np.linalg.solve(np.eye(m) - P, np.ones(m))
    return float(arl[m // 2])


def L_para_arl0_ewma(arl0, lam):
    """Busca el L del EWMA que produce el ARL0 deseado."""
    return optimize.brentq(lambda L: arl_ewma(0.0, 1, lam, L) - arl0, 1.0, 5.0)


def largo_corrida(metodo, delta_eff, p, reps, tmax, rng, sd=1.0):
    """Largo de corrida simulado (Shewhart, EWMA o CUSUM). Devuelve (arreglo, truncadas).

    delta_eff: desplazamiento de la media del estadístico, en unidades de σ/√n.
    sd: desviación estándar del estadístico (1 = proceso sin cambio de variabilidad).
    """
    rl = np.full(reps, float(tmax))
    vivo = np.ones(reps, bool)
    z, cp, cn = np.zeros(reps), np.zeros(reps), np.zeros(reps)
    for t in range(1, tmax + 1):
        idx = np.flatnonzero(vivo)
        if idx.size == 0:
            break
        x = rng.normal(delta_eff, sd, idx.size)
        if metodo == "Shewhart":
            sen = np.abs(x) > p["L"]
        elif metodo == "EWMA":
            z[idx] = p["lam"] * x + (1 - p["lam"]) * z[idx]
            sen = np.abs(z[idx]) > p["H"]
        else:  # CUSUM bilateral
            cp[idx] = np.maximum(0, cp[idx] + x - p["k"])
            cn[idx] = np.maximum(0, cn[idx] - x - p["k"])
            sen = (cp[idx] > p["Hc"]) | (cn[idx] > p["Hc"])
        rl[idx[sen]] = t
        vivo[idx[sen]] = False
    return rl, int(vivo.sum())


def simular_arl(metodo, delta_eff, p, reps, tmax, rng):
    """ARL simulado. Devuelve (ARL, error estándar, corridas truncadas)."""
    rl, trunc = largo_corrida(metodo, delta_eff, p, reps, tmax, rng)
    return float(rl.mean()), float(rl.std(ddof=1) / np.sqrt(reps)), trunc