"""
Főprogram: filmajánló rendszer futtatása és összehasonlítás.

Lépések: adatok betöltése → train/test split (config.TEST_SIZE) → cold-start szűrés (opcionális mintavételezés)
→ user-based és item-based modell tanítása → előrejelzés a teszt halmazon
→ RMSE, MAE, Precision@10, Recall@10 számítása → összehasonlító táblázat kiírása.
"""

import sys

import pandas as pd

import config
from data_loader import load_data, split_data
from evaluation import evaluate
from item_based import ItemBasedCF
from user_based import UserBasedCF


def main():
    # ---- 1. Adatok betöltése ----
    print("Loading data...", flush=True)
    try:
        ratings, movies = load_data()
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Ratings: {len(ratings)}, Movies: {len(movies)}", flush=True)
    train_pct = round((1.0 - config.TEST_SIZE) * 100)
    test_pct = round(config.TEST_SIZE * 100)
    print(f"Splitting {train_pct}/{test_pct} train/test...", flush=True)
    train_ratings, test_ratings = split_data(ratings)
    print(f"Train: {len(train_ratings)}, Test: {len(test_ratings)}", flush=True)

    # ---- 2. Teszt halmaz: csak olyan párok, ahol user és film is szerepelt a trainben ----
    # Cold start (új user vagy új film) esetén nincs értelme a CF előrejelzésnek,
    # ezért ezeket kihagyjuk az értékelésből.
    train_users = set(train_ratings["userId"])
    train_movies = set(train_ratings["movieId"])
    test_mask = (
        test_ratings["userId"].isin(train_users) & test_ratings["movieId"].isin(train_movies)
    )
    test_eval = test_ratings.loc[test_mask].reset_index(drop=True)
    if len(test_eval) < len(test_ratings):
        print(f"Evaluating on {len(test_eval)} test pairs (cold-start excluded).", flush=True)
    # Opcionális: gyors futtatáshoz csak N véletlenszerű teszt pár
    if config.TEST_SAMPLE_SIZE is not None:
        n = min(config.TEST_SAMPLE_SIZE, len(test_eval))
        test_eval = test_eval.sample(n=n, random_state=config.RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled {n} test pairs for evaluation.", flush=True)

    # ---- 3. Modell tanítás ----
    print("Fitting User-based CF...", flush=True)
    user_cf = UserBasedCF(k=config.K_NEIGHBORS, min_common=config.MIN_COMMON)
    user_cf.fit(train_ratings)

    print("Fitting Item-based CF...", flush=True)
    item_cf = ItemBasedCF(k=config.K_NEIGHBORS, min_common=config.MIN_COMMON)
    item_cf.fit(train_ratings)

    # ---- 4. Előrejelzés a teszt párokon ----
    print("Predicting (User-based)...", flush=True)
    pred_user = user_cf.predict_batch(test_eval)

    print("Predicting (Item-based)...", flush=True)
    pred_item = item_cf.predict_batch(test_eval)

    # ---- 5. Metrikák és összehasonlító táblázat ----
    print("Computing metrics...", flush=True)
    metrics = evaluate(test_eval, pred_user, pred_item)

    results = pd.DataFrame({
        "Metrika": [
            "RMSE",
            "MAE",
            f"Precision@{config.TOP_K_RECOMMENDATIONS}",
            f"Recall@{config.TOP_K_RECOMMENDATIONS}",
        ],
        "User-based": [
            metrics["rmse_user"],
            metrics["mae_user"],
            metrics["precision_at_k_user"],
            metrics["recall_at_k_user"],
        ],
        "Item-based": [
            metrics["rmse_item"],
            metrics["mae_item"],
            metrics["precision_at_k_item"],
            metrics["recall_at_k_item"],
        ],
    })

    print("\n=== Összehasonlítás: User-based vs Item-based ===\n")
    print(results.to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
