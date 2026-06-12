import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Klasa wezla drzewa pozostaje struktura lekka dzieki __slots__
class _Node:
    # __slots__ blokuje tworzenie slownika __dict__, co oszczedza pamiec przy duzych drzewach
    __slots__ = ("feat", "thresh", "left", "right", "value")

    # Inicjalizacja wezla
    def __init__(self, value=None):
        # Indeks cechy, wzgledem ktorej nastepuje podzial w tym wezle
        self.feat = None
        # Wartosc progowa podzialu dla wybranej cechy
        self.thresh = None
        # Wskaznik na lewe poddrzewo (dla x < thresh)
        self.left = None
        # Wskaznik na prawe poddrzewo (dla x >= thresh)
        self.right = None
        # Wartosc predykcji, zdefiniowana glownie dla lisci (najczestsza klasa)
        self.value = value


# Glowna klasa modelu drzewa decyzyjnego
class CARTModel:
    # Konstruktor przyjmujacy parametry regularyzacyjne
    def __init__(self, max_depth=None, min_samples_leaf=1, random_state=None):
        # Maksymalna dopuszczalna glebokosc drzewa (None = brak limitu)
        self.max_depth = max_depth
        # Rzeczywista minimalna liczba probek wymagana w kazdym z lisci po podziale
        self.min_samples_leaf = min_samples_leaf
        # Nowoczesny, bezpieczny wielowatkowo generator liczb pseudolosowych
        self.rng = np.random.default_rng(random_state)
        # Wskaznik na glowny wezel (korzen) drzewa
        self.root = None
        # Tablica przetrzymujaca unikalne etykiety klas (pozwala na uzywanie np. stringow)
        self.classes_ = None
        # Liczba unikalnych klas w zbiorze treningowym
        self.n_classes_ = 0

    # Metoda trenujaca model
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARTModel":
        # Znajdujemy unikalne etykiety i mapujemy oryginalne y na wartosci od 0 do C-1
        self.classes_, y_int = np.unique(y, return_inverse=True)
        # Zapisujemy liczbe zidentyfikowanych klas
        self.n_classes_ = len(self.classes_)
        # Inicjujemy rekurencyjne budowanie drzewa od poziomu 0
        self.root = self._build(X, y_int, 0)
        # Zwracamy wyuczony obiekt
        return self

    # Rekurencyjna metoda budujaca strukture wezlow
    def _build(self, X, y, depth):
        # Liczba probek, ktore dotarly do tego wezla
        n_samples = len(y)
        # Zliczamy wystapienia kazdej klasy w biezacym wezle uzywajac wewnetrznych intow
        counts = np.bincount(y, minlength=self.n_classes_)
        # Odczytujemy oryginalna etykiete najczesciej wystepujacej klasy
        node_value = self.classes_[counts.argmax()]
        # Tworzymy wezel przypisujac mu dominujaca etykiete jako wartosc predykcji
        node = _Node(value=node_value)

        # Warunki stopu rekurencji
        # 1. Osiagnieto maksymalna glebokosc
        # 2. Nie mamy wystarczajaco probek, aby zachowac min_samples_leaf po obu stronach
        # 3. Wezel jest w 100% czysty (wszystkie probki naleza do jednej klasy)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < 2 * self.min_samples_leaf) or \
           (np.max(counts) == n_samples):
            # Zwracamy wezel bez podzialow (staje sie lisciem)
            return node

        # Zmienne przechowujace informacje o najlepszym znalezionym podziale
        best_gini = float('inf')
        best_feat = None
        best_thresh = None

        # Iterujemy przez wszystkie dostepne cechy (kolumny w X)
        for feat_idx in range(X.shape[1]):
            # Wektor wartosci dla danej cechy
            feature_values = X[:, feat_idx]
            
            # Aby uniknac zlozonosci O(N^2), sortujemy probki po wartosciach biezacej cechy
            order = np.argsort(feature_values)
            X_sorted = feature_values[order]
            y_sorted = y[order]

            # Inicjujemy liczniki klas dla lewej czesci podzialu (poczatkowo puste)
            left_counts = np.zeros(self.n_classes_, dtype=int)
            # Inicjujemy liczniki klas dla prawej czesci (poczatkowo wszystkie probki)
            right_counts = counts.copy()

            # Iteracyjnie przenosimy probki z prawej strony na lewa, analizujac podzial miedzy nimi
            for i in range(1, n_samples):
                # Klasa przenoszonej wlasnie probki
                c = y_sorted[i - 1]
                # Zwiekszamy licznik tej klasy po lewej stronie
                left_counts[c] += 1
                # Zmniejszamy licznik tej klasy po prawej stronie
                right_counts[c] -= 1

                # Jesli wartosc cechy sie nie zmienila miedzy sasiadami, nie mozemy w tym miejscu przeciac
                if X_sorted[i] == X_sorted[i - 1]:
                    # Pomijamy ten punkt i idziemy do kolejnej probki
                    continue

                # Sprawdzamy prawdziwy warunek min_samples_leaf (dla obu powstajacych dzieci)
                if i < self.min_samples_leaf or (n_samples - i) < self.min_samples_leaf:
                    # Jesli ktoras strona ma za malo probek, odrzucamy ten potencjalny podzial
                    continue

                # Obliczamy sume kwadratow zliczen potrzebna do wzoru Giniego
                sum_left_sq = np.sum(left_counts ** 2)
                sum_right_sq = np.sum(right_counts ** 2)
                
                # Zanieczyszczenie Giniego dla lewego wezla-dziecka (G = 1 - sum(p_i^2))
                gini_left = 1.0 - sum_left_sq / (i * i)
                # Zanieczyszczenie Giniego dla prawego wezla-dziecka
                gini_right = 1.0 - sum_right_sq / ((n_samples - i) * (n_samples - i))
                
                # Wazone zanieczyszczenie obu wezlow na podstawie liczby probek w kazdym z nich
                weighted_gini = (i * gini_left + (n_samples - i) * gini_right) / n_samples

                # Sprawdzamy czy znaleziony wynik jest wyraznie lepszy od dotychczasowego
                # Lub, jesli jest identyczny, decydujemy rzutem monety dla zlamania remisow
                if weighted_gini < best_gini - 1e-9 or \
                   (abs(weighted_gini - best_gini) <= 1e-9 and self.rng.random() < 0.5):
                    
                    # Zapisujemy nowe najlepsze wyniki
                    best_gini = weighted_gini
                    # Zapisujemy indeks najlepszej cechy
                    best_feat = feat_idx
                    # Wartosc odciecia to srednia z dwoch sasiadujacych punktow podzialu
                    best_thresh = (X_sorted[i] + X_sorted[i - 1]) / 2.0

        # Jesli nie znaleziono zadnego validnego podzialu (np. brak spelnienia min_samples_leaf)
        if best_feat is None:
            # Wezel staje sie lisciem
            return node

        # Generujemy maske logiczna dla najlepszego podzialu na calym wektorze (bez sortowania)
        left_mask = X[:, best_feat] < best_thresh
        
        # Zapisujemy parametry podzialu we wlasciwosciach wezla
        node.feat = best_feat
        node.thresh = best_thresh
        
        # Rekurencyjnie budujemy lewe odgalezienie, przekazujac przefiltrowane dane
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        # Rekurencyjnie budujemy prawe odgalezienie na pozostalej czesci danych
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        
        # Zwracamy wezel z poprawnie doczepionymi odgalezieniami
        return node

    # Metoda dokonujaca predykcji dla wektora badz macierzy wejsciowej X
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Tworzymy pusta tablice wynikowa o typie takim samym jak oryginalne etykiety
        out = np.empty(len(X), dtype=self.classes_.dtype)
        
        # Iterujemy przez wszystkie wektory cech w macierzy X
        for i, x in enumerate(X):
            # Zaczynamy poszukiwania od korzenia drzewa
            node = self.root
            # Dopoki wezel posiada zdefiniowany podzial (nie jest lisciem)
            while node.feat is not None:
                # Wybieramy lewa lub prawa strone na podstawie progu i wartosci cechy probki
                node = node.left if x[node.feat] < node.thresh else node.right
            # Odczytana po dojsciu do liscia wartosc staje sie nasza predykcja
            out[i] = node.value
            
        # Zwracamy caly wektor wynikowy
        return out
        
    # Rekurencyjna metoda zliczajaca maksymalna glebokosc od wybranego wezla
    def _tree_depth(self, node):
        # Jesli jestesmy w lisciu, osiagnelismy glebokosc 1 na tej sciezce
        if node.feat is None:
            return 1
        # Zwracamy 1 plus wieksza z glebokosci miedzy lewym a prawym dzieckiem
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))

    # Rekurencyjna metoda zliczajaca ilosc ostatecznych lisci w strukturze
    def _count_leaves(self, node):
        # Jesli wezel jest lisciem, zliczamy go jako 1
        if node.feat is None:
            return 1
        # W przeciwnym razie jest to suma lisci w lewym i prawym poddrzewie
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    # Dekorator property pozwala na pobranie glebokosci jak atrybutu (bez nawiasow obj.depth)
    @property
    def depth(self) -> int:
        # Przekazujemy korzen do metody liczacej glebokosc
        return self._tree_depth(self.root)

    # Dekorator property pozwala na pobranie liczby lisci (obj.n_leaves)
    @property
    def n_leaves(self) -> int:
        # Przekazujemy korzen do metody zliczajacej liscie
        return self._count_leaves(self.root)

class SklearnCARTModel:
    """Binary decision tree using sklearn's DecisionTreeClassifier (Gini)."""

    def __init__(self, max_depth=10, min_samples_leaf=1, random_state=None):
        self._clf = DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnCARTModel":
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    @property
    def depth(self) -> int:
        return self._clf.get_depth()

    @property
    def n_leaves(self) -> int:
        return self._clf.get_n_leaves()


MODELS = {"cart": CARTModel, "sklearn": SklearnCARTModel}
