"""
Algoritmo Genetico (representacion real) para maximizar
    f(x) = x * sin(10*pi*x) + 1   en [-1, 2]
Algoritmos Evolutivos - TESI - 2do parcial
Uso:  python ga.py [config.json]
"""
import json, math, random, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def aptitud(x):
    """Funcion objetivo (a maximizar): f(x) = x*sin(10*pi*x) + 1."""
    return x * math.sin(10 * math.pi * x) + 1


def seleccion_torneo(pob, fit, k, rng):
    """Torneo de tamano k: gana el individuo con mayor aptitud."""
    idx = rng.sample(range(len(pob)), k)
    return pob[max(idx, key=lambda i: fit[i])]


def cruce_aritmetico(p1, p2, pc, rng):
    """Cruce aritmetico (real): h = a*p1 + (1-a)*p2, a ~ U(0,1)."""
    if rng.random() < pc:
        a = rng.random()
        return a * p1 + (1 - a) * p2, a * p2 + (1 - a) * p1
    return p1, p2


def mutacion_gaussiana(x, pm, sigma, lo, hi, rng):
    """Mutacion gaussiana: x' = x + N(0, sigma), recortada a [lo, hi]."""
    if rng.random() < pm:
        x += rng.gauss(0, sigma)
    return min(max(x, lo), hi)


def ejecutar(cfg, verbose=False):
    rng = random.Random(cfg["semilla"])
    lo, hi = cfg["limite_inferior"], cfg["limite_superior"]
    n = cfg["tam_poblacion"]
    pob = [rng.uniform(lo, hi) for _ in range(n)]
    historial_mejor, historial_prom, diversidad = [], [], []
    mejor_x, mejor_f = None, -1e9
    gen_estab = 0
    t0 = time.perf_counter()
    for g in range(cfg["generaciones"]):
        fit = [aptitud(x) for x in pob]
        i_best = max(range(n), key=lambda i: fit[i])
        if fit[i_best] > mejor_f + 1e-9:
            mejor_f, mejor_x, gen_estab = fit[i_best], pob[i_best], g
        historial_mejor.append(mejor_f)
        historial_prom.append(sum(fit) / n)
        m = sum(pob) / n
        diversidad.append(math.sqrt(sum((x - m) ** 2 for x in pob) / n))
        # elitismo
        elite = sorted(range(n), key=lambda i: -fit[i])[: cfg["elitismo"]]
        nueva = [pob[i] for i in elite]
        while len(nueva) < n:
            p1 = seleccion_torneo(pob, fit, cfg["k_torneo"], rng)
            p2 = seleccion_torneo(pob, fit, cfg["k_torneo"], rng)
            h1, h2 = cruce_aritmetico(p1, p2, cfg["prob_cruce"], rng)
            for h in (h1, h2):
                nueva.append(mutacion_gaussiana(h, cfg["prob_mutacion"],
                             cfg["sigma_mutacion"], lo, hi, rng))
        pob = nueva[:n]
    t = time.perf_counter() - t0
    return dict(mejor_x=mejor_x, mejor_f=mejor_f, gen_estab=gen_estab,
                tiempo=t, hist=historial_mejor, prom=historial_prom, div=diversidad)


def fuerza_bruta(lo, hi, pasos=3_000_000):
    """Validacion: barrido de malla fina del dominio."""
    bx, bf = lo, aptitud(lo)
    for i in range(pasos + 1):
        x = lo + (hi - lo) * i / pasos
        f = aptitud(x)
        if f > bf:
            bx, bf = x, f
    return bx, bf


def graficar(res, ruta="convergencia.png"):
    plt.figure(figsize=(7, 4))
    plt.plot(res["hist"], label="Mejor aptitud", lw=2)
    plt.plot(res["prom"], label="Aptitud promedio", ls="--")
    plt.xlabel("Generación"); plt.ylabel("f(x)")
    plt.title("Convergencia del AG"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(ruta, dpi=150); plt.close()


if __name__ == "__main__":
    ruta_cfg = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    with open(ruta_cfg, encoding="utf-8") as f:
        cfg = json.load(f)
    r = ejecutar(cfg)
    bx, bf = fuerza_bruta(cfg["limite_inferior"], cfg["limite_superior"])
    graficar(r)
    print(f"Mejor individuo AG : x = {r['mejor_x']:.6f}  f(x) = {r['mejor_f']:.6f}")
    print(f"Fuerza bruta       : x = {bx:.6f}  f(x) = {bf:.6f}")
    print(f"Error absoluto     : {abs(bf - r['mejor_f']):.2e}")
    print(f"Generación de estabilización: {r['gen_estab']}  |  tiempo: {r['tiempo']:.4f} s")
