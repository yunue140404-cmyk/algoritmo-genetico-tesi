"""Producto 4: 4 configuraciones x 10 semillas."""
import json, statistics as st, copy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ga import ejecutar, fuerza_bruta, graficar

base = json.load(open("config.json"))
CONFIGS = {
 "C1": dict(tam_poblacion=10,  prob_cruce=0.6, prob_mutacion=0.01),
 "C2": dict(tam_poblacion=30,  prob_cruce=0.7, prob_mutacion=0.05),
 "C3": dict(tam_poblacion=50,  prob_cruce=0.8, prob_mutacion=0.10),
 "C4": dict(tam_poblacion=100, prob_cruce=0.9, prob_mutacion=0.30),
}
SEMILLAS = list(range(1, 11))
bx, bf = fuerza_bruta(-1, 2)
out = {"opt": [bx, bf], "cfg": {}}
curvas, divs = {}, {}
for nombre, c in CONFIGS.items():
    runs = []
    for s in SEMILLAS:
        cfg = copy.deepcopy(base); cfg.update(c); cfg["semilla"] = s
        runs.append(ejecutar(cfg))
    mejores = [r["mejor_f"] for r in runs]
    out["cfg"][nombre] = dict(**c,
        prom=st.mean(mejores), std=st.stdev(mejores), mejor=max(mejores), peor=min(mejores),
        gen=st.mean(r["gen_estab"] for r in runs), t=st.mean(r["tiempo"] for r in runs)*1000,
        aciertos=sum(1 for m in mejores if abs(m-bf) < 1e-3),
        div_final=st.mean(r["div"][-1] for r in runs))
    G = len(runs[0]["hist"])
    curvas[nombre] = [st.mean(r["hist"][g] for r in runs) for g in range(G)]
    divs[nombre] = [st.mean(r["div"][g] for r in runs) for g in range(G)]
json.dump(out, open("resultados.json", "w"), indent=1)
for k, v in out["cfg"].items(): print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
print("opt", bx, bf)
for fn, data, yl, tt in (("conv_configs.png", curvas, "Mejor aptitud (promedio de 10 corridas)", "Convergencia por configuración"),
                         ("diversidad.png", divs, "Desviación estándar de la población", "Diversidad por configuración")):
    plt.figure(figsize=(7, 4))
    for k, v in data.items(): plt.plot(v, label=k, lw=2)
    if fn.startswith("conv"): plt.axhline(bf, color="k", ls=":", label="Óptimo")
    plt.xlabel("Generación"); plt.ylabel(yl); plt.title(tt); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
# función objetivo
import numpy as np
x = np.linspace(-1, 2, 2000); y = x*np.sin(10*np.pi*x)+1
plt.figure(figsize=(7,3.6)); plt.plot(x, y); plt.scatter([bx],[bf],color="r",zorder=3,label=f"Máximo global ({bx:.4f}, {bf:.4f})")
plt.xlabel("x"); plt.ylabel("f(x)"); plt.title("f(x) = x·sin(10πx) + 1"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig("funcion.png", dpi=150); plt.close()
