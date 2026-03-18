"""
Felhasználó-alapú kollaboratív szűrés (user-based CF).

A gondolat: ha két felhasználó hasonlóan értékelt filmeket, akkor egy
új filmre várhatóan hasonlóan fognak értékelni. Ezért egy adott (user, film)
párra: megnézzük, kik értékelték a filmet a tanító halmazban, kiszámoljuk
a céluser és ezek közötti Pearson-hasonlóságot, majd a k legközelebbi user
értékelését súlyozott átlagoljuk.
"""

import numpy as np
import pandas as pd

import config


def _pearson_similarity(ratings_a: np.ndarray, ratings_b: np.ndarray, min_common: int) -> float:
    """
    Két értékelés-vektor Pearson-korrelációja csak a közös elemeken.

    A közös elemek: azok a filmek, amelyekre mindkét user adott értékelést.
    Ha kevesebb ilyen van, mint min_common, vagy az egyik vektor szórása 0,
    akkor 0-t adunk vissza (nincs hasonlóság).

    Args:
        ratings_a: Egyik user értékelései (közös filmekre).
        ratings_b: Másik user értékelései (ugyanaz a sorrend).
        min_common: Minimum közös értékelés száma.

    Returns:
        Pearson [-1, 1], vagy 0.0 ha nem számolható.
    """
    # Mask: mindkét helyen van érték (nincs NaN)
    mask = ~(np.isnan(ratings_a) | np.isnan(ratings_b))
    if np.sum(mask) < min_common:
        return 0.0
    a = ratings_a[mask].astype(float)
    b = ratings_b[mask].astype(float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def _get_user_means(user_item: pd.DataFrame) -> pd.Series:
    """Userenkénti átlagos értékelés (NaN-ok nélkül). Fallback előrejelzéshez kell."""
    return user_item.mean(axis=1)


def _build_movie_to_users(train_ratings: pd.DataFrame):
    """
    Filmhez rendeli: kik értékelték és milyen ponttal.

    Ezt használjuk: adott filmre mely usereket kell megnéznünk
    a user-based előrejelzésnél.
    """
    movie_to_users = {}
    for movie_id, group in train_ratings.groupby("movieId"):
        movie_to_users[movie_id] = list(zip(group["userId"].values, group["rating"].values))
    return movie_to_users


def _cache_key(a: int, b: int) -> tuple:
    """
    (a, b) és (b, a) ugyanazt a cache kulcsot kapja, mert a hasonlóság szimmetrikus.
    Így nem számoljuk kétszer ugyanazt a user-párt.
    """
    return (min(a, b), max(a, b))


class UserBasedCF:
    """
    Felhasználó-alapú kollaboratív szűrés: Pearson + k legközelebbi user.

    fit(): tanító értékelésekből user–item mátrix, user átlagok, film→userek map.
    predict(): egy (user_id, movie_id) párra előrejelzés; fallback = user átlag.
    predict_batch(): teszt tábla minden sorára előrejelzés.
    """

    def __init__(self, k: int = None, min_common: int = None):
        self.k = config.K_NEIGHBORS if k is None else k
        self.min_common = config.MIN_COMMON if min_common is None else min_common
        self.user_item_ = None       # pivot: user x movie
        self.user_means_ = None      # userenkénti átlag
        self.movie_to_users_ = None # movieId -> [(userId, rating), ...]
        self.train_users_ = None     # tanító halmazban lévő user id-k
        self.train_movies_ = None    # tanító halmazban lévő film id-k
        self._sim_cache_ = {}        # (uid1, uid2) -> Pearson, gyorsításra

    def fit(self, train_ratings: pd.DataFrame):
        """
        Modell „tanítása”: user–item mátrix, user átlagok, film→userek index.
        Csak a train adatot használja.
        """
        self.user_item_ = train_ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self.user_means_ = _get_user_means(self.user_item_)
        self.movie_to_users_ = _build_movie_to_users(train_ratings)
        self.train_users_ = set(self.user_item_.index)
        self.train_movies_ = set(self.user_item_.columns)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Egy (user_id, movie_id) párra előrejelzés.

        - Ha a user nincs a trainben: globális átlag.
        - Ha a film nincs a trainben: user átlaga.
        - Egyébként: a filmet értékelő userek közül k legközelebbiek
          (Pearson) értékelésének súlyozott átlaga; csak pozitív hasonlóságú
          szomszédokat használunk. Ha nincs ilyen: user átlag.
        - Eredményt 0.5 és 5.0 közé vágjuk.
        """
        if self.user_item_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.train_users_:
            return float(self.user_means_.mean())
        if movie_id not in self.movie_to_users_:
            return float(self.user_means_.get(user_id, self.user_means_.mean()))

        users_rated_movie = self.movie_to_users_[movie_id]
        if not users_rated_movie:
            return float(self.user_means_.get(user_id, self.user_means_.mean()))

        user_row = self.user_item_.loc[user_id]
        similarities = []
        ratings_for_movie = []

        for other_uid, other_rating in users_rated_movie:
            if other_uid == user_id:
                continue
            key = _cache_key(user_id, other_uid)
            if key not in self._sim_cache_:
                other_row = self.user_item_.loc[other_uid]
                common = user_row.index.intersection(other_row.index)
                if len(common) < self.min_common:
                    self._sim_cache_[key] = 0.0
                    continue
                r_a = user_row[common].values
                r_b = other_row[common].values
                self._sim_cache_[key] = _pearson_similarity(r_a, r_b, self.min_common)
            sim = self._sim_cache_[key]
            if sim > 0:
                similarities.append(sim)
                ratings_for_movie.append(other_rating)

        if not similarities or len(similarities) == 0:
            return float(self.user_means_.get(user_id, self.user_means_.mean()))

        similarities = np.array(similarities)
        ratings_for_movie = np.array(ratings_for_movie)
        top_k = min(self.k, len(similarities))
        idx = np.argsort(similarities)[::-1][:top_k]
        sim_k = similarities[idx]
        rat_k = ratings_for_movie[idx]
        pred = np.average(rat_k, weights=sim_k)
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, test_ratings: pd.DataFrame) -> np.ndarray:
        """
        A test_ratings minden sorára (userId, movieId) meghívja a predict-et,
        és az előrejelzéseket ugyanabban a sorrendben adja vissza.
        """
        return np.array([
            self.predict(int(row["userId"]), int(row["movieId"]))
            for _, row in test_ratings.iterrows()
        ])
