# Movie Recommendation System – Collaborative Filtering

🇭🇺 [Magyar verzió](README.hu.md) · 📄 [Full paper (PDF, English)](docs/report_en.pdf) · 📄 [Teljes tanulmány (PDF, magyar)](docs/report_hu.pdf) · 🖥️ [Slides (PDF)](docs/slides_hu.pdf) · 📓 [Kaggle notebook](notebooks/movie-recommendation-kaggle.ipynb)

University research project (Óbuda University, Infocommunication Techniques course): a comparative study of user-based and item-based collaborative filtering on the MovieLens `ml-latest-small` dataset.

**Author:** Kónya Tamás Ádám ([konyatomi16@stud.uni-obuda.hu](mailto:konyatomi16@stud.uni-obuda.hu))

## Summary

The study compares two classic neighborhood-based collaborative filtering methods on the MovieLens `ml-latest-small` dataset (100,000+ ratings, 9,742 movies, 610 users). Similarity is computed with Pearson correlation, and predictions are the weighted average of the k nearest neighbors. Both models are evaluated under three train/test splits (70/30, 75/25, 80/20), using prediction accuracy (RMSE, MAE) and recommendation quality (Precision@10, Recall@10).

**Key finding:** user-based collaborative filtering outperforms the item-based approach on every metric and every split. The best prediction accuracy is achieved at the 75/25 split (RMSE = 0.9704; MAE = 0.7492), while the largest training set (80/20 split) yields the highest Recall@10 (0.6747).

### Table I – Prediction accuracy (RMSE, MAE)

| Split | Method      | RMSE   | MAE    |
|-------|-------------|--------|--------|
| 70/30 | User-based  | 0.9785 | 0.7547 |
| 70/30 | Item-based  | 0.9848 | 0.7616 |
| 75/25 | User-based  | **0.9704** | **0.7492** |
| 75/25 | Item-based  | 0.9807 | 0.7596 |
| 80/20 | User-based  | 0.9723 | 0.7502 |
| 80/20 | Item-based  | 0.9832 | 0.7622 |

### Table II – Recommendation quality (Precision@10, Recall@10)

| Split | Method      | Precision@10 | Recall@10 |
|-------|-------------|--------------|-----------|
| 70/30 | User-based  | 0.6180       | 0.5814    |
| 70/30 | Item-based  | 0.5398       | 0.5450    |
| 75/25 | User-based  | 0.5921       | 0.6222    |
| 75/25 | Item-based  | 0.5200       | 0.5845    |
| 80/20 | User-based  | 0.5554       | **0.6747** |
| 80/20 | Item-based  | 0.4915       | 0.6397    |

For the literature review, methodology details, and limitations, see the [full paper](docs/report_en.pdf).

## Repository layout

```
.
├── README.md / README.hu.md   – project description (EN / HU)
├── config.py                  – settings (data path, train/test ratio, k, thresholds)
├── data_loader.py             – CSV loading, train/test split, user–item matrix
├── user_based.py              – user-based collaborative filtering
├── item_based.py              – item-based collaborative filtering
├── evaluation.py              – RMSE, MAE, Precision@k, Recall@k
├── main.py                    – runnable pipeline, comparison table
├── requirements.txt           – Python dependencies
├── data/ml-latest-small/      – MovieLens dataset (CSV files)
├── notebooks/                 – the single-file notebook version submitted to Kaggle
└── docs/                      – paper (EN/HU) and slides (PDF)
```

This repository contains the **original, modular implementation** of the research (as it was developed), and the `notebooks/` folder also contains the **single-file notebook version submitted to Kaggle**, which reproduces the same study on the Kaggle platform: [kaggle.com/code/tamsknya/movierecommendationsystem-in-kaggle](https://www.kaggle.com/code/tamsknya/movierecommendationsystem-in-kaggle).

## Data

The dataset lives in `data/ml-latest-small/` (MovieLens, [GroupLens](http://grouplens.org/datasets/)):

- `ratings.csv` – userId, movieId, rating, timestamp
- `movies.csv` – movieId, title, genres
- `links.csv`, `tags.csv` – auxiliary metadata (not used directly in this study)

## Usage

From the project root:

```bash
pip install -r requirements.txt
python main.py
```

The program:

1. Loads the ratings and movies data
2. Splits into train/test (per `config.TEST_SIZE`, default 70/30)
3. Fits both the user-based and item-based CF models
4. Predicts on the test set with both methods
5. Prints RMSE, MAE, Precision@10 and Recall@10 in a comparison table

A full run (~29k test pairs) can take several minutes; the modules cache similarity computations to speed this up. For a quick test, set e.g. `TEST_SAMPLE_SIZE = 500` in `config.py`.

## Configuration

Adjustable in `config.py`:

- `TEST_SIZE` – test fraction (default 0.3; the paper also reports 0.25 and 0.2)
- `K_NEIGHBORS` – number of nearest neighbors (default 30)
- `MIN_COMMON` – minimum shared ratings required for a similarity score (default 2)
- `RANDOM_STATE` – seed for reproducibility

## Modules

- **config.py** – all settings in one place: data paths, train/test ratio, k neighbors, minimum common ratings, relevance threshold, etc.
- **data_loader.py** – loads and validates the CSVs (ratings, movies); `split_data()` for the train–test split; `build_user_item_matrix()` builds the pivot table.
- **user_based.py** – user-based CF: Pearson similarity between users, weighted average of the k nearest users' ratings, falls back to the user mean, with a similarity cache.
- **item_based.py** – item-based CF: Pearson similarity between movies, weighted average of the k nearest items' ratings, falls back to the item mean, with a cache.
- **evaluation.py** – RMSE, MAE (prediction error); Precision@k, Recall@k (relevant = rating ≥ 4) for both models.
- **main.py** – runnable pipeline: load → split → cold-start filtering → fit user + item CF → predict_batch → evaluate → print the comparison table.

## License

The source code is shared for educational/research purposes without a separate software license. The MovieLens dataset used here is subject to [GroupLens' usage terms](http://grouplens.org/datasets/).
