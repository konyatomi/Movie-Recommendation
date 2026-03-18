"""
Értékelési metrikák az ajánló rendszerhez.

RMSE és MAE: előrejelzési pontosság (mennyire közelítik a valós értékelést).
Precision@k és Recall@k: ajánlási minőség (a top-k ajánlott közül hány releváns,
és a relevánsak hányadát találtuk meg). Releváns = rating >= küszöb (pl. 4).
"""

import numpy as np
import pandas as pd

import config


def rmse(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Gyöknégyzetes középhiba (Root Mean Squared Error).

    Képlet: sqrt(átlag((pred - actual)^2)). Nagy hibákat jobban büntet
    mint az MAE. Minél kisebb, annál jobb az előrejelzés.

    Args:
        predictions: Előrejelzett értékelések (ugyanolyan sorrendben mint actual).
        actual: Valós értékelések.

    Returns:
        RMSE érték.
    """
    if len(predictions) != len(actual) or len(predictions) == 0:
        raise ValueError("predictions and actual must have same non-zero length")
    return float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(actual)) ** 2)))


def mae(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Átlagos abszolút hiba (Mean Absolute Error).

    Képlet: átlag(|pred - actual|). Minél kisebb, annál jobb.
    Kevésbé érzékeny extrém hibákra mint az RMSE.
    """
    if len(predictions) != len(actual) or len(predictions) == 0:
        raise ValueError("predictions and actual must have same non-zero length")
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(actual))))


def precision_recall_at_k(
    test_ratings: pd.DataFrame,
    pred_user: np.ndarray,
    pred_item: np.ndarray,
    k: int = None,
    relevant_threshold: float = None,
) -> tuple[float, float]:
    """
    Macro átlagolt Precision@k és Recall@k mindkét modellre (user-based és item-based).

    Logika: felhasználónként a tesztben lévő (film, értékelés) párokat
    előrejelzés szerint rangsoroljuk; a top-k az „ajánlott”. Releváns = valós
    rating >= relevant_threshold (pl. 4). Precision@k = (releváns a top-k-ban) / k,
    Recall@k = (releváns a top-k-ban) / (összes releváns a user tesztjében).
    Ezeket userenként számoljuk, majd átlagoljuk (macro).

    Args:
        test_ratings: userId, movieId, rating oszlopokkal.
        pred_user: User-based előrejelzések, ugyanolyan sorrendben mint a test.
        pred_item: Item-based előrejelzések, ugyanolyan sorrendben.
        k: Top-k (alapértelmezett config-ból).
        relevant_threshold: E feletti rating = releváns (config-ból).

    Returns:
        ((prec_user, rec_user), (prec_item, rec_item)).
    """
    k = config.TOP_K_RECOMMENDATIONS if k is None else k
    relevant_threshold = (
        config.RELEVANT_RATING_THRESHOLD if relevant_threshold is None else relevant_threshold
    )
    test_df = test_ratings.copy()
    if "rating" not in test_df.columns:
        raise ValueError("test_ratings must have 'rating' column")
    if len(pred_user) != len(test_df):
        raise ValueError("pred_user length must match test_ratings")
    if len(pred_item) != len(test_df):
        raise ValueError("pred_item length must match test_ratings")

    test_df["pred_user"] = pred_user
    test_df["pred_item"] = pred_item
    test_df["relevant"] = test_df["rating"] >= relevant_threshold

    def _precision_recall_one(pred_col: str) -> tuple[float, float]:
        """Egy előrejelzés oszlopra (pred_user vagy pred_item) számol prec/rec userenként."""
        precisions = []
        recalls = []
        for user_id, group in test_df.groupby("userId"):
            # User teszt elemeit előrejelzés szerint csökkenőben
            group = group.sort_values(pred_col, ascending=False).reset_index(drop=True)
            top_k = group.head(k)
            n_relevant_in_top = top_k["relevant"].sum()
            n_relevant_total = group["relevant"].sum()
            precisions.append(n_relevant_in_top / k if k > 0 else 0.0)
            recalls.append(
                n_relevant_in_top / n_relevant_total if n_relevant_total > 0 else 0.0
            )
        if not precisions:
            return 0.0, 0.0
        return float(np.mean(precisions)), float(np.mean(recalls))

    prec_user, rec_user = _precision_recall_one("pred_user")
    prec_item, rec_item = _precision_recall_one("pred_item")
    return (prec_user, rec_user), (prec_item, rec_item)


def evaluate(
    test_ratings: pd.DataFrame,
    pred_user: np.ndarray,
    pred_item: np.ndarray,
) -> dict:
    """
    Összes metrika kiszámítása user-based és item-based előrejelzésekre.

    Visszaadott szótár kulcsai: rmse_user, rmse_item, mae_user, mae_item,
    precision_at_k_user, recall_at_k_user, precision_at_k_item, recall_at_k_item.
    Az actual értékelések a test_ratings["rating"] oszlopból jönnek.
    """
    actual = test_ratings["rating"].values
    (prec_user, rec_user), (prec_item, rec_item) = precision_recall_at_k(
        test_ratings, pred_user, pred_item
    )
    return {
        "rmse_user": rmse(pred_user, actual),
        "rmse_item": rmse(pred_item, actual),
        "mae_user": mae(pred_user, actual),
        "mae_item": mae(pred_item, actual),
        "precision_at_k_user": prec_user,
        "recall_at_k_user": rec_user,
        "precision_at_k_item": prec_item,
        "recall_at_k_item": rec_item,
    }
