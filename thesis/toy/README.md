# Zabawka 2D i kontrola inicjalizacji

Kod wyników wstępnych z `thesis/SPEC.md` (sekcje 3 i 10). Uruchamiane na CPU, poza głównym kodem repo.

| Plik | Co robi | Zależności |
|---|---|---|
| `toy_dynamic_2d.py` | monokularna zabawka 2D: kodowania PE, sprzężenie F Σ Fᵀ, kary `el` i `curv`, aktywacje, baseline z głowami jak w repo | `jax[cpu]`, `optax`, `numpy` |
| `aggregate.py` | średnie po ziarnach i różnice sparowane wobec wariantu odniesienia | standardowa biblioteka |
| `check_deform_init.py` | `DeformNetwork` z repo odtworzona w numpy: szok w iteracji 3000 (\|δs\|, σ_max(F), det F < 0) | `numpy` |
| `results.jsonl` | 94 przebiegi zabawki z 26–27.10.2026 | — |

Przykłady:

```bash
pip install "jax[cpu]" optax numpy
# baseline, kamera skacząca, ziarno 0
python toy_dynamic_2d.py pe10 0 none 0 teleport 0
# kara krzywizny, kamera płynna
python toy_dynamic_2d.py pe10 0 curv 1e-3 smooth 0
# sprzężenie F Sigma F^T
python toy_dynamic_2d.py pe10 1 none 0 smooth 0
# baseline z inicjalizacją głów jak w repo, aktywacja SiLU
REPO_HEADS=1 python toy_dynamic_2d.py pe10 0 none 0 teleport 0
ACT=silu python toy_dynamic_2d.py pe10 0 none 0 teleport 0
# tabela różnic wobec pe10
python aggregate.py results.jsonl pe10
```

Jeden przebieg trwa ok. 2 min (warianty z 1 zapytaniem do sieci) albo ok. 9 min (warianty z punktami sigma) na jednym rdzeniu. Wyniki to przesłanki, nie dowody: 2D, bez zasłaniania i densyfikacji, 1–3 ziarna.
