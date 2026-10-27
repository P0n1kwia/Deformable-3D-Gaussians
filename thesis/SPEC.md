# Specyfikacja projektu: regularyzacja pola deformacji w skali Gaussianów

Wersja robocza z 27.10.2026. Dokument referencyjny: zebrane ustalenia, wzory, zmiany w kodzie, plan eksperymentów i punkty decyzji. Własny plan dla promotora warto napisać od nowa, korzystając z tego jako ściągi.

## 1. Tytuł

- **Praca magisterska (propozycja):** Regularyzacja pola deformacji w skali Gaussianów w monokularnej rekonstrukcji scen dynamicznych metodą 3D Gaussian Splatting.
- **Artykuł (roboczo):** *Locally Affine Deformation Fields for Monocular Dynamic Gaussian Splatting*.

Tytuł pracy celowo nie obiecuje wyniku: pasuje zarówno do wariantu „metoda działa”, jak i do wariantu analitycznego (sekcja 9).

## 2. Streszczenie

Deformable 3DGS modeluje ruch siecią MLP, która przesuwa Gaussiany z przestrzeni kanonicznej do sceny w chwili t i osobnymi głowami zmienia ich obrót i skalę. Przy nagraniu z jednej kamery pole łatwo się przeucza: jest chropowate poniżej skali pojedynczego Gaussiana, a kształt Gaussianów nie wynika z ruchu otoczenia. Projekt proponuje:

1. **karę krzywizny** pola liczoną wzdłuż osi każdego Gaussiana, zerową dla ruchu afinicznego (rdzeń pracy);
2. **diagnostykę** pola: chropowatość, szok przy włączeniu deformacji, zależność od ruchu kamery;
3. **sprzężenie kształtu z ruchem**, Σₜ = F Σ Fᵀ, jako hipotezę do sprawdzenia z limitem czasu;
4. opcjonalnie **podział Gaussianów tam, gdzie ruch na ich nośniku nie jest afiniczny**.

Ewaluacja: D-NeRF (różnica train–test) i NeRF-DS (PSNR, LPIPS), wobec uczciwego baseline'u i istniejących regularyzacji (Nerfies, EER).

## 3. Problem: co jest w kodzie Deformable 3DGS

Gałąź `main` (wersja Lightweight, rasteryzator `diff-gaussian-rasterization-extentions`, commit `d595eac`).

| Obserwacja | Gdzie | Znaczenie |
|---|---|---|
| Trzy niezależne głowy: δx (3), δr (4), δs (3) | `utils/time_utils.py`, `DeformNetwork` | kształt nie wynika z ruchu |
| `scales = get_scaling + d_scaling`, `rotations = get_rotation + d_rotation` | `gaussian_renderer/__init__.py:88-89` | skala dodawana liniowo, kwaternion po dodaniu nieznormalizowany |
| Normalizacja kwaternionu zakomentowana | `cuda_rasterizer/forward.cu:129` (submoduł) | R(λq̂) = (1−λ²)I + λ²R(q̂): obrót + skalowanie w płaszczyźnie ⊥ osi; rodzina 4-wymiarowa, **nie** dowolna macierz liniowa |
| Wejście sieci odłączone: `deform.step(gaussians.get_xyz.detach(), …)` | `train.py:97` | gradient z MLP nie trafia do pozycji kanonicznych |
| Deformacja włączana po `warm_up = 3000`, głowy z domyślną inicjalizacją PyTorcha | `arguments/__init__.py`, `train.py:90` | **szok w iteracji 3000** (niżej) |
| `compute_cov3D_python` ignoruje deformację | `gaussian_renderer/__init__.py:85-86` | ścieżki `cov3D_precomp` nie da się użyć bez zmian |
| Densyfikacja patrzy na skalę kanoniczną | `scene/gaussian_model.py:321-353` | rozciągnięcie w czasie nie wywołuje podziału |
| `train.py` wypisuje najlepszy test PSNR ze wszystkich checkpointów | `train.py:133-136` | do pracy liczyć metryki z ostatniej iteracji (`render.py`, `metrics.py`) |

**Szok w iteracji 3000** (zmierzone na nietrenowanej `DeformNetwork` z repo, odtworzonej w numpy, 3 ziarna):
- mediana |δs| = 0,03–0,05 na oś, przy kanonicznych skalach rzędu 0,005–0,02;
- F = I + ∂δx/∂x: mediana największej wartości osobliwej 3,2–3,9, a det F < 0 dla 43–47% punktów.

**Przeuczenie** (Droby, EER, arXiv 2604.16747): na D-NeRF średnia różnica PSNR train–test to ok. 6 dB; ponad 80% pochodzi z dzielenia Gaussianów; kara kNN za niespójną deformację zmniejsza różnicę o ok. 40% (4DGS).

## 4. Hipotezy

- **H1.** Kara krzywizny w skali Gaussiana zmniejsza różnicę train–test na D-NeRF i poprawia PSNR/LPIPS na NeRF-DS wobec uczciwego baseline'u, co najmniej na poziomie EER, ale bez szukania sąsiadów.
- **H2.** Wyprowadzenie kształtu z ruchu (F Σ Fᵀ) poprawia uogólnianie, szczególnie przy płynnej kamerze. **Uwaga:** w mojej zabawce 2D tego nie odtworzyłem (sekcja 10); traktować jako hipotezę, nie założenie.
- **H3.** Efekty zależą od ruchu kamery: D-NeRF (kamera „skacząca”) i nagrania rzeczywiste (płynna kamera) zachowują się inaczej (DyCheck, arXiv 2210.13445).

## 5. Metoda

### 5.1. Uczciwy baseline (obowiązkowy, przed czymkolwiek innym)

1. **Zerowa inicjalizacja głów** δx, δr, δs (`nn.init.zeros_` na wagach i biasie). W zabawce inicjalizacja jak w repo kosztowała 1,4 dB przy kamerze skaczącej (3/3 ziaren).
2. **Coarse-to-fine dla PE pozycji** (okno Nerfies: wagi częstotliwości j rosną od 0 do 1 według ½(1 − cos(π·clip(α − j, 0, 1))), α od 0 do L w pierwszych ok. 60% iteracji z deformacją). W zabawce +2,8 dB przy kamerze skaczącej (3/3).
3. Opcjonalnie: znormalizowany kwaternion i skala w logarytmie (q̂ ⊗ δq, s·exp(δs)) jako wariant baseline'u.

Każdą nową metodę porównujemy z baseline'em po tych poprawkach. Inaczej zysk może wynikać z naprawy inicjalizacji, a nie z metody.

### 5.2. Kara krzywizny w skali Gaussiana (wkład 1)

Dla Gaussiana o środku μ, obrocie R = [r₁ r₂ r₃] i skalach s₁, s₂, s₃ definiujemy osie aₖ = h·sₖ·rₖ (h = √3, krok ograniczony z dołu, np. 10⁻³ rozmiaru sceny). Dla pola φₜ(x) = x + Δ(x, t):

```
cₖ = φₜ(μ + aₖ) + φₜ(μ − aₖ) − 2·φₜ(μ)
L_curv = mean_i Σₖ ‖cₖ‖² / ‖aₖ‖²
```

Własności:
- z rozwinięcia Taylora cₖ = ∂²φ[aₖ, aₖ] + O(‖aₖ‖⁴), więc kara jest **zerowa dla każdego ruchu afinicznego**: obrót, rozciąganie i ścinanie są darmowe (w przeciwieństwie do kary sprężystej, która ciągnie F do obrotu: w Nerfies to ‖log Σ‖²_F z wartości osobliwych Jacobianu, z odporną funkcją Geman–McClure i wagą równą udziałowi punktu w renderze; w zabawce użyto prostszego ‖FᵀF − I‖²);
- to dokładnie człon pomijany przez przybliżenie pierwszego rzędu F Σ Fᵀ; transformacja unscented używa go do poprawki średniej;
- nie wymaga sąsiadów (w przeciwieństwie do EER), skala dopasowuje się do każdego Gaussiana.

Implementacja: losowa próbka n Gaussianów (np. 8192) na iterację, 6 dodatkowych zapytań do MLP na Gaussiana, ten sam czas t (z szumem AST) co w renderze, osie z odłączonych (`detach`) parametrów kanonicznych. Strata: `loss += λ_curv · L_curv` po warm-upie. λ dobrać przeglądem na jednej scenie (zakres startowy 10⁻⁴…10⁻²; strata repo to L1 + D-SSIM, więc skala inna niż w zabawce). Precyzja: fp32, bez TF32.

Szkic (do dopisania w `train.py` lub osobnym module):

```python
def curvature_loss(deform, gaussians, t, h=3 ** 0.5, n=8192, min_step=1e-3):
    idx = torch.randint(0, gaussians.get_xyz.shape[0], (n,), device='cuda')
    mu = gaussians.get_xyz[idx].detach()
    R = build_rotation(gaussians._rotation[idx]).detach()           # kolumny = osie
    step = torch.clamp(h * gaussians.get_scaling[idx].detach(), min=min_step)
    A = (R * step[:, None, :]).transpose(1, 2)                      # wiersze a_k
    pts = torch.cat([mu[:, None], mu[:, None] + A, mu[:, None] - A], 1).reshape(-1, 3)
    d = deform.step(pts, t[idx].repeat_interleave(7, 0))[0].view(n, 7, 3)
    phi = pts.view(n, 7, 3) + d
    c = phi[:, 1:4] + phi[:, 4:7] - 2 * phi[:, :1]                  # (n, 3, 3)
    return ((c ** 2).sum(-1) / step ** 2).mean()
```

W trybie `--is_6dof` `deform.step` zwraca macierz 4×4, więc φ trzeba liczyć jako T(p)·p.

### 5.3. Sprzężenie Σₜ = F Σ Fᵀ (wkład 3, limit 2 tygodnie)

- F z punktów sigma: dₖ = (φ(μ + aₖ) − φ(μ − aₖ)) / 2 ≈ F aₖ, F̂ = D A⁻¹, Σₜ = F̂ L Lᵀ F̂ᵀ z żywym L = R S (gradient do kształtu kanonicznego przez dokładny wzór).
- Ten sam zestaw 7 punktów co w karze krzywizny, więc w trybie sprzężonym kara jest za darmo.
- Renderer: nowy argument `cov3D`; gdy podany, idzie jako `cov3D_precomp` (format `strip_symmetric`: xx, xy, xz, yy, yz, zz), a `scales = rotations = None`. Rasteryzator zwraca gradient do `cov3D_precomp` (sprawdzone w `d595eac`). Zachować tensor `means2D_densify`.
- Jedna funkcja `deform_gaussians(...)` zamiast `deform.step` w 11 miejscach: `train.py:97, 217`, `render.py:46, 65, 94, 140, 184, 231, 287`, `train_gui.py:565, 672`.
- Koszt: 7 zapytań zamiast 1. Szacunek pamięci aktywacji ok. 18 KiB na zapytanie, czyli ok. 10 GB dla T-Rex (79 tys. Gaussianów); pomaga gradient checkpointing albo punkty sigma tylko dla widocznych Gaussianów. FPS renderowania spadnie (baseline: średnio 179 FPS według README).
- Ablacja obowiązkowa: Σₜ = R_F Σ R_Fᵀ (sam obrót z rozkładu polarnego F). Jeśli daje to samo, teza o „pełnym F” upada.

### 5.4. Podział przy nieafinicznym ruchu (wkład 4, opcjonalny)

Kryterium: max po kilku chwilach t z ‖cₖ‖ / ‖aₖ‖ powyżej progu, wtedy podział wzdłuż osi k o największej krzywiźnie. Kara mówi „bądź afiniczny w mojej skali”, podział „jeśli się nie da, zmniejsz mnie”. Test **przy tej samej liczbie Gaussianów** co baseline, bo więcej Gaussianów to więcej przeuczenia (EER). Pokrewne: EvoGS (Dong i in., arXiv 2609.00994), GradRig (2609.05127).

### 5.5. Diagnostyka (wkład 2, zbierana przy okazji)

- **Chropowatość:** iloraz mediany ‖J‖ w punkcie (autograd) i ‖F‖ z różnic w skali Gaussiana. Zabawka: 7–8 dla PE z 10 częstotliwościami.
- **Szok:** σ_max(F), odsetek det F < 0, |δs| / s w iteracji 3000, z domyślną i zerową inicjalizacją.
- **Różnica train–test:** train PSNR na wszystkich widokach treningowych, test na pełnym zbiorze testowym, ostatnia iteracja.
- **Zastrzeżenie z zabawki:** gładkość pola nie przewiduje jakości (gładsze pole przy kodowaniu z rozmiarem Gaussiana i przy SiLU dawało gorszy test). Nie twierdzić „gładsze = lepsze”.

## 6. Dane i protokół

- **D-NeRF:** 8 scen, `--is_blender`, 40 tys. iteracji. Lego: test na zbiorze walidacyjnym (zalecenie autora repo, test Lego jest niespójny z treningiem). Główna metryka: różnica train–test. Odniesienie z README: średnio 41,14 dB (7 scen bez Lego), 43 tys. Gaussianów.
- **NeRF-DS:** 7 scen, 20 tys. iteracji (README), opcjonalnie `--is_6dof`. Test z drugiej kamery. Metryki: PSNR, SSIM, LPIPS (VGG). Wartość 24,11 dB z artykułu pochodzi z gałęzi `paper`, więc podajemy własny baseline.
- **Opcjonalnie:** HyperNeRF (vrig) lub iPhone (DyCheck) do eksperymentu z ruchem kamery.
- **Losowość:** wszystkie warianty na wszystkich scenach; 3 ziarna na 2 scenach do oszacowania rozrzutu.
- **Dziennik eksperymentów:** commit, konfiguracja, ziarno, metryki z ostatniej iteracji, czas iteracji, liczba Gaussianów.

## 7. Baseline'y i ablacje

| Grupa | Warianty |
|---|---|
| Baseline | Deformable 3DGS z repo; po poprawkach z 5.1 |
| Konkurencyjne regularyzacje | kara sprężysta Nerfies, wiernie: ‖log Σ‖²_F + Geman–McClure (w punkcie i w skali Gaussiana); EER (kNN, własna implementacja, przegląd λ) |
| Metoda | + L_curv; + sprzężenie; + sprzężenie + L_curv; + podział (opcja) |
| Ablacje | krok h (0,5; 1; √3; 2,5); λ_curv; liczba częstotliwości PE (6, 10, coarse-to-fine); aktywacja (ReLU, SiLU); obrót polarny wobec pełnego F; kamera skacząca wobec płynnej |

## 8. Punkty decyzji

Kryteria spisane przed eksperymentami i uzgodnione z promotorem.

| Kiedy | Kryterium | Jeśli tak | Jeśli nie |
|---|---|---|---|
| koniec tyg. 2 | baseline odtwarza README; L_curv poprawia test PSNR/LPIPS lub różnicę train–test na ≥ 2 z 4 scen bez spadku test PSNR o > 0,2 dB | pełne tabele | ≤ 1 tydzień na λ i h; dalej nic: plan B1 albo zmiana tematu |
| koniec tyg. 6 | L_curv ≥ EER i kara sprężysta na większości scen, przy niższym koszcie | L_curv jako rdzeń artykułu | plan B1 |
| koniec tyg. 8 | sprzężenie wygrywa z baseline'em + L_curv na większości scen NeRF-DS | drugi wkład | wynik negatywny z wyjaśnieniem |
| tyg. 9–10 | zostaje czas | wkład 4 lub drugi backbone (4DGS) | tylko ablacje i pisanie |

## 9. Plany awaryjne

- **B1, artykuł analityczny:** co decyduje o uogólnianiu w monokularnym 3DGS (inicjalizacja, coarse-to-fine, ruch kamery, skala regularyzacji, brak związku gładkości z jakością).
- **B2, magisterka bez artykułu:** rzetelny wynik negatywny to pełnoprawna praca (wymagania obrony potwierdzić z promotorem).
- **B3, inne miejsce:** druga runda ICIAP, warsztaty przy CVPR/ICCV, preprint arXiv.

## 10. Wyniki wstępne (zabawka 2D)

Monokular 2D: fantom Shepp–Logana odkształcany wirem, zoomem i przesunięciem; jeden rzut 1D (128 px) na klatkę; 300 Gaussianów 2D; MLP 4×128 jak w Deformable 3DGS; 4000 kroków Adama; kamera skacząca (losowy kąt na klatkę) albo płynna (obrót o 180°). Zmiana PSNR nowych widoków wobec baseline'u z zerową inicjalizacją głów:

| Wariant | Kamera skacząca | Kamera płynna (widoki ±10–20°) |
|---|---|---|
| coarse-to-fine | +2,8 dB (3/3) | +0,3 dB |
| L_curv (λ = 10⁻³) | +0,5 dB (3/3) | +0,1 dB (rozbieżnie) |
| coarse-to-fine + L_curv wobec coarse-to-fine | +0,4 dB | 0 |
| kara sprężysta ‖FᵀF − I‖² (najlepsze λ, 1 ziarno) | +0,3 dB | +1,1 dB |
| sprzężenie F Σ Fᵀ | −0,4 dB (3/3) | ≈ 0 |
| kodowanie z rozmiarem Gaussiana (jak mip-NeRF) | −0,8 dB (3/3) | −0,8 dB |
| SiLU zamiast ReLU (2 ziarna) | +0,35 dB | −1,5 dB |
| inicjalizacja głów jak w repo | −1,4 dB (3/3) | +0,4 dB (rozbieżnie) |

Zastrzeżenia: 2D, rzuty addytywne bez zasłaniania, bez densyfikacji, jeden typ ruchu, 1–3 ziarna. To przesłanki do sprawdzenia na prawdziwych danych, nie wyniki do artykułu. Kod i surowe wyniki: `thesis/toy/`.

## 11. Nowość (stan na 26–27.10.2026, przegląd arXiv)

Nie znalazłem pracy, która w rekonstrukcji z wideo bierze pełną kowariancję z gradientu wyuczonego, swobodnego pola, ani kary drugiego rzędu liczonej w skali każdego Gaussiana. Mechanizm F Σ Fᵀ jest jednak znany w innych zastosowaniach:

| Praca | Co robi | Różnica |
|---|---|---|
| PhysGaussian (2311.12198), GASP | F z symulatora MPM | fizyka zamiast pola z obrazów |
| D3GA (2311.08581) | J Σ Jᵀ z czworościanów klatki | szablon, wiele kamer |
| GauHuman (2312.02973) i podobne awatary z LBS | kowariancja przez zmieszaną macierz skinningu | szablon SMPL |
| CAGE-GS, GSDeformer, View-Dependent Deformation Fields | Jacobian deformacji przy edycji | edycja, nie rekonstrukcja |
| **GradRig (2609.05127)** | pełne F skinningu (z gradientem wag) jako F Σ Fᵀ | riggowanie gotowych splatów |
| **Make It Flow (2610.13860)** | 7 punktów sigma przez pole prędkości | animacja wody |
| **Kinematics-GS (2605.08635)** | kowariancja wzdłuż kierunku prędkości | prędkość, nie jej gradient; rozmycie ruchu |
| PIDG (2511.06299), VeloGauss (2605.10567) | ∇v tylko w karze fizycznej | kształt z osobnych głów (sprawdzone w kodzie) |
| EER (2604.16747) | kara kNN za niespójną deformację | sąsiedzi, kara także za obrót |
| SaRO-GS (2412.06299) | pole uwzględniające rozmiar Gaussiana | bliskie odrzuconemu pomysłowi z kodowaniem |
| EvoGS (2609.00994, Dong i in.) | densyfikacja świadoma deformacji | inne kryterium podziału |

Przed wysłaniem artykułu powtórzyć przegląd z ostatnich 3 miesięcy (hasła: „deformation gradient” + „dynamic Gaussian”, „Jacobian” + „covariance”, „curvature” / „second-order” + „deformation field”).

## 12. Harmonogram (13 tygodni)

| Tydzień | Zadania |
|---|---|
| 1 | środowisko, trening baseline'u na 2 + 2 scenach, poprawki 5.1, skrypty metryk z ostatniej iteracji |
| 2 | L_curv, pierwsze porównanie, diagnostyka szoku i chropowatości; **punkt decyzji 1** |
| 3–4 | pełne D-NeRF; przegląd λ i h; kara sprężysta |
| 5–6 | NeRF-DS; EER; **punkt decyzji 2** |
| 7–8 | sprzężenie F Σ Fᵀ i ablacja obrotu polarnego; **punkt decyzji 3** |
| 9–10 | ablacje, ewentualnie wkład 4 lub drugi backbone, rysunki |
| 11–13 | pisanie artykułu, poprawki, zgłoszenie |

Termin ICIAP 2027 sprawdzić w aktualnym call for papers (w 2025 r. rundy zamykały się 14.02 i 15.04; Springer LNCS).

## 13. Literatura

**Obowiązkowo:** Zwicker i in., EWA Splatting (2002); Nerfies (2011.12948); EER (2604.16747); DyCheck (2210.13445); PhysGaussian (2311.12198); GradRig (2609.05127); D-MiSo (2405.14276) i GaMeS (2402.01459).

**Warto:** 3DGUT (CVPR 2025) i Make It Flow (2610.13860); MonoDyGauBench (2412.04457); SC-GS (2312.14937); SteepGS (2505.05587), EvoGS (2609.00994), SaRO-GS (2412.06299); Bookstein (1989, thin-plate splines) i Rueckert i in. (1999, FFD z energią zginania); rozdział o kinematyce z podręcznika mechaniki ośrodków ciągłych (np. Bonet i Wood).

## 14. Pytania do promotora

- [ ] Dostęp do GPU: jaka karta, ile godzin tygodniowo, PLGrid?
- [ ] Czy GMUM ma wytrenowane modele Deformable 3DGS lub D-MiSo na D-NeRF?
- [ ] Czy sprzężenie F Σ Fᵀ jest ciekawe w kontekście D-MiSo / GaMeS?
- [ ] Która runda ICIAP i czy preprint na arXiv przed zgłoszeniem jest w porządku?
- [ ] Co idzie do pracy, a co tylko do artykułu? Czy publikacja jest wymagana do obrony?
- [ ] Wspólna baza kodu (np. gsplat) czy oryginalne repo?
