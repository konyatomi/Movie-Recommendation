"""
Elem-alapú (item-based) kollaboratív szűrés.

A gondolat: ha egy felhasználó hasonló filmeket értékel hasonlóan, akkor
egy új, hasonló filmet is hasonlóan fog. Ezért egy adott (user, film) párra:
megnézzük, mit értékeltek a user által már értékelt filmek közül, kiszámoljuk
a célfilm és ezek közötti Pearson-hasonlóságot, majd a k legközelebbi filmre
adott user-értékeléseket súlyozott átlagoljuk.
"""

import numpy as np
import pandas as pd

import config


def _pearson_similarity(ratings_a: np.ndarray, ratings_b: np.ndarray, min_common: int) -> float:
    """
    Két értékelés-vektor Pearson-korrelációja csak a közös usereken.

    Közös userek: akik mindkét filmet értékelték. Ha kevesebb ilyen van
    mint min_common, vagy az egyik szórása 0, 0-t adunk vissza.
    """
    mask = ~(np.isnan(ratings_a) | np.isnan(ratings_b))
    if np.sum(mask) < min_common:
        return 0.0
    a = ratings_a[mask].astype(float)
    b = ratings_b[mask].astype(float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def _get_item_means(item_user: pd.DataFrame) -> pd.Series:
    """Filmenkénti átlagos értékelés (NaN-ok nélkül). Fallback előrejelzéshez."""
    return item_user.mean(axis=1)


def _cache_key(a: int, b: int) -> tuple:
    """(a, b) és (b, a) ugyanaz a cache kulcs – a hasonlóság szimmetrikus."""
    return (min(a, b), max(a, b))


def _build_user_to_movies(train_ratings: pd.DataFrame):
    """
    Userhez rendeli: mely filmeket értékelte és milyen ponttal.

    Ezt használjuk: adott usernek mely filmjei közül keressük a célfilmhez
    hasonló elemeket.
    """
    user_to_movies = {}
    for user_id, group in train_ratings.groupby("userId"):
        user_to_movies[user_id] = list(zip(group["movieId"].values, group["rating"].values))
    return user_to_movies


class ItemBasedCF:
    """
    Elem-alapú kollaboratív szűrés: Pearson + k legközelebbi film (item).

    fit(): item–user mátrix (film x user), item átlagok, user→filmek map.
    predict(): egy (user_id, movie_id) párra előrejelzés; fallback = item átlag.
    predict_batch(): teszt tábla minden sorára előrejelzés.
    """

    def __init__(self, k: int = None, min_common: int = None):
        self.k = config.K_NEIGHBORS if k is None else k
        self.min_common = config.MIN_COMMON if min_common is None else min_common
        self.item_user_ = None       # pivot: movie x user
        self.item_means_ = None      # filmenkénti átlag
        self.user_to_movies_ = None  # userId -> [(movieId, rating), ...]
        self.train_users_ = None
        self.train_movies_ = None
        self._sim_cache_ = {}        # (movieId1, movieId2) -> Pearson

    def fit(self, train_ratings: pd.DataFrame):
        """
        Modell tanítása: item–user mátrix, item átlagok, user→filmek index.
        """
        self.item_user_ = train_ratings.pivot_table(
            index="movieId", columns="userId", values="rating"
        )
        self.item_means_ = _get_item_means(self.item_user_)
        self.user_to_movies_ = _build_user_to_movies(train_ratings)
        self.train_users_ = set(self.item_user_.columns)
        self.train_movies_ = set(self.item_user_.index)
        self._sim_cache_ = {}
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Egy (user_id, movie_id) párra előrejelzés.

        - Ha a user nincs a trainben: globális átlag.
        - Ha a film nincs a trainben: item (film) átlaga.
        - Egyébként: a user által értékelt filmek közül a célfilmhez
          k legközelebbiekre adott értékelések súlyozott átlaga (csak
          pozitív hasonlóságú itemek). Ha nincs ilyen: item átlag.
        - Eredmény 0.5–5.0 között.
        """
        if self.item_user_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.train_users_:
            return float(self.item_means_.mean())
        if movie_id not in self.train_movies_:
            return float(self.item_means_.get(movie_id, self.item_means_.mean()))

        user_rated_movies = self.user_to_movies_.get(user_id, [])
        if not user_rated_movies:
            return float(self.item_means_.get(movie_id, self.item_means_.mean()))

        movie_row = self.item_user_.loc[movie_id]
        similarities = []
        user_ratings = []

        for other_mid, user_rating in user_rated_movies:
            if other_mid == movie_id:
                continue
            if other_mid not in self.item_user_.index:
                continue
            key = _cache_key(movie_id, other_mid)
            if key not in self._sim_cache_:
                other_row = self.item_user_.loc[other_mid]
                common = movie_row.index.intersection(other_row.index)
                if len(common) < self.min_common:
                    self._sim_cache_[key] = 0.0
                    continue
                r_a = movie_row[common].values
                r_b = other_row[common].values
                self._sim_cache_[key] = _pearson_similarity(r_a, r_b, self.min_common)
            sim = self._sim_cache_[key]
            if sim > 0:
                similarities.append(sim)
                user_ratings.append(user_rating)

        if not similarities:
            return float(self.item_means_.get(movie_id, self.item_means_.mean()))

        similarities = np.array(similarities)
        user_ratings = np.array(user_ratings)
        top_k = min(self.k, len(similarities))
        idx = np.argsort(similarities)[::-1][:top_k]
        sim_k = similarities[idx]
        rat_k = user_ratings[idx]
        pred = np.average(rat_k, weights=sim_k)
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, test_ratings: pd.DataFrame) -> np.ndarray:
        """
        A test_ratings minden sorára (userId, movieId) előrejelzés,
        ugyanabban a sorrendben visszaadva.
        """
        return np.array([
            self.predict(int(row["userId"]), int(row["movieId"]))
            for _, row in test_ratings.iterrows()
        ])
