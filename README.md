# Filmajánló rendszer – Kollaboratív szűrés

User-based és item-based kollaboratív szűrés a MovieLens ml-latest-small adatkészleten, összehasonlító értékeléssel.

## Követelmények

- Python 3.9+
- Függőségek: `pip install -r requirements.txt`

## Adat

Az adatok a `src/archive/ml-latest-small/` mappában vannak (MovieLens adatkészlet):

- `ratings.csv` – userId, movieId, rating, timestamp
- `movies.csv` – movieId, title, genres

## Használat

A projekt gyökérkönyvtárából:

```bash
pip install -r requirements.txt
python main.py
```

A program:

1. Betölti a rating és movie adatokat
2. 70/30 arányban felosztja train/testre
3. User-based és item-based CF modellt tanít
4. A teszt halmazon előrejelzést készít mindkét módszerrel
5. Kiírja az RMSE, MAE, Precision@10 és Recall@10 metrikákat összehasonlító táblázatban

A teljes futtatás (kb. 29k teszt pár) több percig is eltarthat; a modulok hasonlósági cache-t használnak a gyorsítás érdekében. Gyors próbához a `config.py`-ban állítsd be pl. `TEST_SAMPLE_SIZE = 500`-at.

## Konfiguráció

A `config.py` fájlban módosítható:

- `TEST_SIZE` – teszt arány (alapértelmezett 0.3)
- `K_NEIGHBORS` – k legközelebbi szomszéd (alapértelmezett 30)
- `MIN_COMMON` – minimum közös értékelés a hasonlósághoz (alapértelmezett 2)
- `RANDOM_STATE` – véletlenség reprodukálhatósága

## Modulok – mi mit csinál

- **config.py** – Összes beállítás egy helyen: adat útvonalak, train/test arány (70/30), k szomszéd, minimum közös értékelés, relevancia küszöb, stb.
- **data_loader.py** – CSV-k betöltése (ratings, movies), oszlopok ellenőrzése, tisztítás; `split_data()` 70/30 felosztás; `build_user_item_matrix()` pivot tábla.
- **user_based.py** – Felhasználó-alapú CF: Pearson hasonlóság userek között, k legközelebbi user értékelésének súlyozott átlaga; fallback user átlagra; hasonlósági cache.
- **item_based.py** – Elem-alapú CF: Pearson a filmek között, k legközelebbi filmre adott user-értékelések súlyozott átlaga; fallback item átlagra; cache.
- **evaluation.py** – RMSE, MAE (előrejelzési hiba); Precision@k, Recall@k (releváns = rating ≥ 4); mindkét modellre, macro átlaggal.
- **main.py** – Futtatható pipeline: betöltés → split → cold-start szűrés → fit user + item CF → predict_batch → evaluate → összehasonlító táblázat kiírása.

## Licenc

A MovieLens adatkészlet használati feltételei: [GroupLens](http://grouplens.org/datasets/).
