# Decision Trees on Unbalanced Datasets

Autorzy: Viktoriia Nowotka, Paweł Łasica  

---

## Cel projektu

Porównanie metryk jakości binarnych drzew decyzyjnych CART dla różnego stopnia niezrównoważenia zbioru treningowego. Badamy iteracyjne podejście do rozszerzania zbioru treningowego po undersamplingu — jak zmienia się jakość kolejnych modeli i jak ma się ona do modelu bazowego?

---

## Algorytm

### Krok 1 — stratyfikacja

10% próbek z każdej klasy odkładane jest jako zbiór testowy (niezmienny przez cały eksperyment). Pozostałe 90% to pula treningowa.

### Krok 2 — model bazowy M_0

Model CART trenowany na niezbalansowanym zbiorze treningowym. Wyniki na zbiorze testowym zapisywane jako punkt odniesienia.

### Krok 3 — undersampling

Ze zbioru treningowego:
- brane są wszystkie próbki klasy mniejszościowej,
- losowo dobierana jest taka sama liczba próbek klasy większościowej.

Powstaje zbalansowany zbiór treningowy. Pozostałe (nieużyte) próbki klasy większościowej tworzą pulę kandydatów.

### Krok 4 — pętla iteracyjna (25 × 9)

Dla 25 powtórzeń (`run`), dla każdej wartości `n_samples` ∈ `{0.5%, 1%, 2%, 5%, 10%, 15%, 25%, 35%, reszta}`:

```
i.   Trenuj nowy model M_i_j na aktualnym zbiorze treningowym
ii.  Oceń M_i_j na zbiorze testowym -> zapisz metryki
iii. M_i_j klasyfikuje pulę kandydatów (nieużyte próbki klasy większościowej)
iv.  Wyznacz false positives (błędnie zaklasyfikowane jako klasa mniejszościowa)
v.   Losowo wybierz n_samples % z tych próbek
vi.  Dodaj je do zbioru treningowego
```

### Krok 5 — agregacja

Wyniki ze wszystkich 25 powtórzeń są uśredniane i zbierane do tabeli wynikowej.

---

## Klasyfikator CART

Kryterium podziału: indeks Gini

```
Gini = 1 − (p_1^2 + p_2^2)
```

gdzie `p_1` — udział klasy pozytywnej, `p_2` — klasy negatywnej w węźle.

W każdym węźle wybierany jest podział minimalizujący nieczystość. Lewa gałąź: wartość `< próg`, prawa: `>= próg`. Warunek stopu: minimalna liczba próbek w węźle lub jednolita klasa.

---

## Metryki jakości

| Metryka | Wzór |
|---|---|
| Dokładność | `(TP + TN) / (TP + FP + FN + TN)` |
| Błąd | `1 − Dokładność` |
| TPR (czułość) | `TP / (TP + FN)` |
| FPR | `FP / (FP + TN)` |
| Precyzja | `TP / (TP + FP)` |
| F1-score | `2 · Precyzja · TPR / (Precyzja + TPR)` |

F1-score jest metryką wiodącą — uwzględnia kompromis między precyzją a czułością, co jest kluczowe przy niezbalansowanych klasach.

---

## Datasety

Pobierz i umieść pliki CSV w `data/raw/<nazwa>/*index.csv*` (folder nie jest śledzony przez git).

| Dataset | Próbki | Większościowa | Mniejszościowa | Num. | Cat. | Niezbalans. | Źródło |
|---|---|---|---|---|---|---|---|
| Credit Card Fraud | 284 807 | 284 315 | 492 | 29 | 0 | 0.17% | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Give Me Some Credit | 150 000 | 139 974 | 10 026 | 10 | 0 | 7.16% | [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) |
| Adult Income | 48 842 | 37 155 | 11 687 | 5 | 8 | 31.45% | [UCI](https://archive.ics.uci.edu/dataset/2/adult) |

Trzy datasety celowo różnią się stopniem niezbalansowania (bardzo wysokie / średnie / niskie), żeby zbadać wpływ struktury danych na skuteczność algorytmu.

---

## Architektura projektu

```
.
├── data/
│   ├── raw/                            # surowe CSV — nie w repo (gitignored)
│   │   ├── credit_card_fraud/
│   │   ├── give_me_some_credit/
│   │   └── adult_income/
│   └── processed/                      # przetworzone dane — nie w repo
│       ├── credit_card_fraud/
│       ├── give_me_some_credit/
│       └── adult_income/
│
├── src/
│   ├── data/
│   │   ├── loader.py                   # wczytywanie datasetów do DataFrame
│   │   └── preprocessor.py             # czyszczenie, enkodowanie, inżynieria cech
│   ├── models/
│   │   └── cart.py                     # wrapper DecisionTreeClassifier (kryterium=gini)
│   ├── training/
│   │   ├── sampling.py                 # stratified_split, undersample
│   │   └── iterative.py                # główna pętla: run_iterative_training()
│   ├── evaluation/
│   │   └── metrics.py                  # compute_metrics() -> accuracy, TPR, FPR, F1, precision
│   └── experiments/
│       ├── config.py                   # dataclass ExperimentConfig (n_samples, n_runs, …)
│       └── runner.py                   # run_experiment() -> wyniki per dataset, zapis CSV
│
├── experiments/
│   ├── configs/                        # konfiguracje YAML per dataset
│   │   ├── credit_card_fraud.yaml
│   │   ├── give_me_some_credit.yaml
│   │   └── adult_income.yaml
│   └── results/                        # wyjście: CSV z metrykami (PNG/PDF gitignored)
│       ├── credit_card_fraud/
│       ├── give_me_some_credit/
│       └── adult_income/
│
├── notebooks/                          # EDA i analiza wyników
│   ├── 01_eda_credit_card_fraud.ipynb
│   ├── 02_eda_give_me_some_credit.ipynb
│   ├── 03_eda_adult_income.ipynb
│   └── 04_results_analysis.ipynb
│
├── tests/                              # testy jednostkowe
│   ├── test_sampling.py
│   ├── test_metrics.py
│   └── test_iterative.py
│
├── main.py                             # punkt wejścia CLI
└── requirements.txt
```

### Przepływ danych

```
loader.py -> preprocessor.py -> sampling.py (stratify + undersample)
                                     │
                              iterative.py (pętla 25×9)
                                     │
                              metrics.py -> runner.py -> results/*.csv
```

---

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

Wszystkie eksperymenty (wszystkie 3 datasety):

```bash
python main.py
```

Wybrany dataset:

```bash
python main.py --dataset credit_card_fraud
python main.py --dataset give_me_some_credit
python main.py --dataset adult_income
```

Wyniki (CSV) trafiają do `experiments/results/<dataset>/`.

---

## Zależności

Python 3.10+

| Biblioteka | Zastosowanie |
|---|---|
| `scikit-learn` | CART (`DecisionTreeClassifier`, `train_test_split`) |
| `pandas`, `numpy` | przetwarzanie danych |
| `matplotlib`, `seaborn` | wizualizacja wyników |
| `pyyaml` | konfiguracja eksperymentów |
