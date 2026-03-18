"""
Konfiguráció a filmajánló rendszerhez.

Ebben a fájlban állíthatók a felhasználói adatok útvonalai,
a train/test felosztás aránya, a kollaboratív szűrés paraméterei
és az értékelési küszöbök.
"""

# ========== Adatfájlok helye ==========
# A projekt gyökérhez viszonyított mappa, ahol a MovieLens CSV-k vannak
DATA_DIR = "src/archive/ml-latest-small"
# Az értékelések fájlja (userId, movieId, rating, timestamp)
RATINGS_FILE = "ratings.csv"
# A filmek listája (movieId, title, genres)
MOVIES_FILE = "movies.csv"

# ========== Train / test felosztás ==========
# A teljes adatból mekkora hányad legyen teszt (0.3 = 30% teszt, 70% tanító)
TEST_SIZE = 0.3
# Véletlenszám-generátor magja: ugyanazzal mindig ugyanaz a felosztás
RANDOM_STATE = 42
# Ha egész szám: csak ennyi teszt párra értékelünk (gyorsabb futtatás).
# None esetén az összes teszt páron fut az értékelés.
TEST_SAMPLE_SIZE = None  # Pl. 500 gyors próbához; None = teljes kiértékelés

# ========== Kollaboratív szűrés ==========
# Hány legközelebbi szomszéd (user vagy item) legyen az előrejelzésnél
K_NEIGHBORS = 30
# Minimum hány közös értékelés kell két user/item között a hasonlóság számításához
MIN_COMMON = 2

# ========== Értékelés ==========
# E feletti értékelés számít „releváns”-nak (pl. Precision@k, Recall@k)
RELEVANT_RATING_THRESHOLD = 4.0
# Top-k ajánlás: ennyi legjobb ajánlást nézünk felhasználónként
TOP_K_RECOMMENDATIONS = 10
