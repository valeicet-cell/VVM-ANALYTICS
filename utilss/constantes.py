"""Constantes para gráficos de control: c4 y d2 se calculan; d3 viene de tabla."""
from functools import lru_cache

import numpy as np
from scipy import integrate, stats
from scipy.special import gammaln

# d3 (desviación estándar del rango relativo) para n = 2 a 25
_D3 = {
    2: 0.8525, 3: 0.8884, 4: 0.8798, 5: 0.8641, 6: 0.8480, 7: 0.8332,
    8: 0.8198, 9: 0.8078, 10: 0.7971, 11: 0.7873, 12: 0.7785, 13: 0.7704,
    14: 0.7630, 15: 0.7562, 16: 0.7499, 17: 0.7441, 18: 0.7386, 19: 0.7335,
    20: 0.7287, 21: 0.7242, 22: 0.7199, 23: 0.7159, 24: 0.7121, 25: 0.7084,
}
N_MAX = 25


@lru_cache(maxsize=None)
def c4(n):
    """Factor c4 = E[S]/sigma, con la función gamma."""
    return float(np.sqrt(2 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)))


@lru_cache(maxsize=None)
def d2(n):
    """Factor d2 = E[R]/sigma, por integración numérica de la normal."""
    def f(x):
        return 1 - stats.norm.cdf(x) ** n - (1 - stats.norm.cdf(x)) ** n
    valor, _ = integrate.quad(f, -np.inf, np.inf)
    return float(valor)


def d3(n):
    return _D3[n]


def A2(n):
    return 3 / (d2(n) * np.sqrt(n))


def D3(n):
    return max(0.0, 1 - 3 * d3(n) / d2(n))


def D4(n):
    return 1 + 3 * d3(n) / d2(n)