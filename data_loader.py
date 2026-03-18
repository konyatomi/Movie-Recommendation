"""
Adatbetöltés és felosztás a filmajánló rendszerhez.

Felelősség: CSV-k beolvasása, oszlopok ellenőrzése, 70/30 train–test split,
valamint user–item mátrix építése a tanító adatokból.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config

# A ratings.csv-ban kötelezően kell lennie ezeknek az oszlopoknak
REQUIRED_RATINGS_COLUMNS = {"userId", "movieId", "rating"}
# A movies.csv-ban kötelező oszlopok
REQUIRED_MOVIES_COLUMNS = {"movieId", "title", "genres"}


def _get_data_path(filename: str) -> Path:
    """
    Megkeresi az adatfájl teljes útvonalát.

    Először a szkript mappájához, majd ha nem találja, a szülő mappához
    viszonyítva próbálja a DATA_DIR-t. Így működik akkor is, ha más
    munkamappából futtatjuk a programot.

    Args:
        filename: A fájl neve (pl. ratings.csv).

    Returns:
        A fájl Path objektuma.

    Raises:
        FileNotFoundError: Ha a fájl sehol nem található.
    """
    base = Path(__file__).resolve().parent
    path = base / config.DATA_DIR / filename
    if not path.exists():
        path = base.parent / config.DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


def load_data():
    """
    Betölti az értékelések és a filmek CSV-ját, majd ellenőrzi és tisztítja az adatokat.

    - Ellenőrzi, hogy minden kötelező oszlop megvan.
    - A ratingot számra konvertálja, a hibás sorokat eldobja.
    - userId és movieId egész szám lesz.

    Returns:
        (ratings DataFrame, movies DataFrame) – csak a szükséges oszlopokkal.

    Raises:
        FileNotFoundError: Ha valamelyik CSV hiányzik.
        ValueError: Ha hiányzik oszlop vagy nincs érvényes sor.
        IOError: Ha a fájl olvasása sikertelen.
    """
    ratings_path = _get_data_path(config.RATINGS_FILE)
    movies_path = _get_data_path(config.MOVIES_FILE)

    try:
        ratings = pd.read_csv(ratings_path, encoding="utf-8")
    except Exception as e:
        raise IOError(f"Failed to read {ratings_path}: {e}") from e

    try:
        movies = pd.read_csv(movies_path, encoding="utf-8")
    except Exception as e:
        raise IOError(f"Failed to read {movies_path}: {e}") from e

    # Oszlopok ellenőrzése
    missing_ratings = REQUIRED_RATINGS_COLUMNS - set(ratings.columns)
    if missing_ratings:
        raise ValueError(f"ratings.csv missing columns: {missing_ratings}")

    missing_movies = REQUIRED_MOVIES_COLUMNS - set(movies.columns)
    if missing_movies:
        raise ValueError(f"movies.csv missing columns: {missing_movies}")

    # Csak a szükséges oszlopok, és tisztítás
    ratings = ratings[["userId", "movieId", "rating"]].copy()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    if ratings.empty:
        raise ValueError("ratings.csv has no valid rows after cleaning")

    return ratings, movies


def split_data(ratings: pd.DataFrame, test_size: float = None, random_state: int = None):
    """
    Véletlenszerűen felosztja az értékeléseket tanító és teszt halmazra.

    A sklearn train_test_split-et használja, így a sorok véletlenszerűen
    kerülnek train vagy testbe. A random_state miatt ugyanaz a felosztás
    minden futtatáskor reprodukálható.

    Args:
        ratings: DataFrame userId, movieId, rating oszlopokkal.
        test_size: A teszt halmaz aránya (0–1). None = config-ból.
        random_state: Véletlenszám mag. None = config-ból.

    Returns:
        (train_ratings, test_ratings) – mindkettő reset_index(drop=True)-val.

    Raises:
        ValueError: Ha test_size nem 0 és 1 között van.
    """
    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = config.RANDOM_STATE if random_state is None else random_state

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    return train_ratings.reset_index(drop=True), test_ratings.reset_index(drop=True)


def build_user_item_matrix(ratings: pd.DataFrame):
    """
    User–item mátrixot készít: sorok = userek, oszlopok = filmek, érték = rating.

    Pivot tábla: minden (userId, movieId) párhoz egy rating. Ha valaki
    nem értékelte a filmet, ott NaN marad. A CF algoritmusok ezt a mátrixot
    vagy ennek transzponáltját használják.

    Args:
        ratings: DataFrame userId, movieId, rating oszlopokkal.

    Returns:
        DataFrame: index=userId, columns=movieId, values=rating.
    """
    matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    return matrix
