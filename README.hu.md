# Filmajánló rendszer – Kollaboratív szűrés

🇬🇧 [English version](README.md) · 📄 [Teljes tanulmány (PDF, magyar)](docs/report_hu.pdf) · 📄 [Full paper (PDF, English)](docs/report_en.pdf) · 🖥️ [Prezentáció (PDF)](docs/slides_hu.pdf) · 📓 [Kaggle notebook](notebooks/movie-recommendation-kaggle.ipynb)

Kutatási projekt az Óbudai Egyetem Infokommunikációs technikák tárgyához: felhasználó-alapú (user-based) és elem-alapú (item-based) kollaboratív szűrés összehasonlító vizsgálata a MovieLens `ml-latest-small` adatkészleten.

**Szerző:** Kónya Tamás Ádám ([konyatomi16@stud.uni-obuda.hu](mailto:konyatomi16@stud.uni-obuda.hu))

## Összefoglaló

A kutatás a MovieLens `ml-latest-small` adatkészletén (több mint 100 000 értékelés, 9 742 film, 610 felhasználó) hasonlítja össze a két klasszikus, szomszédság-alapú kollaboratív szűrési módszert. A hasonlóságot Pearson-korrelációval számítjuk, az előrejelzés a k legközelebbi szomszéd súlyozott átlagából adódik. A modelleket három különböző train/test felosztás (70/30, 75/25, 80/20) mellett értékeljük, előrejelzési pontosság (RMSE, MAE) és ajánlási minőség (Precision@10, Recall@10) szerint.

**Fő eredmény:** a felhasználó-alapú módszer minden vizsgált metrikában és felosztásban felülmúlja az elem-alapú megközelítést. A legjobb előrejelzési pontosságot a 75/25 felosztás adja (RMSE = 0,9704; MAE = 0,7492), míg a legmagasabb Recall@10-et a nagyobb tanítóhalmazt használó 80/20 felosztás (0,6747).

### I. táblázat – Előrejelzési pontosság (RMSE, MAE)

| Felosztás | Módszer     | RMSE   | MAE    |
|-----------|-------------|--------|--------|
| 70/30     | User-based  | 0,9785 | 0,7547 |
| 70/30     | Item-based  | 0,9848 | 0,7616 |
| 75/25     | User-based  | **0,9704** | **0,7492** |
| 75/25     | Item-based  | 0,9807 | 0,7596 |
| 80/20     | User-based  | 0,9723 | 0,7502 |
| 80/20     | Item-based  | 0,9832 | 0,7622 |

### II. táblázat – Ajánlási minőség (Precision@10, Recall@10)

| Felosztás | Módszer     | Precision@10 | Recall@10 |
|-----------|-------------|--------------|-----------|
| 70/30     | User-based  | 0,6180       | 0,5814    |
| 70/30     | Item-based  | 0,5398       | 0,5450    |
| 75/25     | User-based  | 0,5921       | 0,6222    |
| 75/25     | Item-based  | 0,5200       | 0,5845    |
| 80/20     | User-based  | 0,5554       | **0,6747** |
| 80/20     | Item-based  | 0,4915       | 0,6397    |

A kutatás korlátait, a szakirodalmi áttekintést és a módszertan részleteit lásd a [teljes tanulmányban](docs/report_hu.pdf).

## Repó felépítése

```
.
├── README.md / README.hu.md   – projekt leírás (EN / HU)
├── config.py                  – beállítások (adatútvonal, train/test arány, k, küszöbök)
├── data_loader.py             – CSV betöltés, train/test split, user–item mátrix
├── user_based.py              – felhasználó-alapú kollaboratív szűrés
├── item_based.py              – elem-alapú kollaboratív szűrés
├── evaluation.py              – RMSE, MAE, Precision@k, Recall@k
├── main.py                    – futtatható pipeline, összehasonlító táblázat
├── requirements.txt           – Python függőségek
├── data/ml-latest-small/      – MovieLens adatkészlet (CSV-k)
├── notebooks/                 – a kutatás Kaggle-re összevont, egyfájlos jegyzetfüzet-verziója
└── docs/                      – tanulmány (HU/EN) és prezentáció (PDF)
```

Ez a repó a kutatás **eredeti, moduláris kódját** tartalmazza (ahogy fejlesztés közben készült), a `notebooks/` mappában pedig a **Kaggle-re beadott, egyfájlos jegyzetfüzet-verzió** is elérhető, amellyel a kutatás online (Kaggle platformon) is reprodukálható: [kaggle.com/code/tamsknya/movierecommendationsystem-in-kaggle](https://www.kaggle.com/code/tamsknya/movierecommendationsystem-in-kaggle).

## Adat

Az adatok a `data/ml-latest-small/` mappában vannak (MovieLens adatkészlet, [GroupLens](http://grouplens.org/datasets/)):

- `ratings.csv` – userId, movieId, rating, timestamp
- `movies.csv` – movieId, title, genres
- `links.csv`, `tags.csv` – kiegészítő metaadatok (ebben a kutatásban nem használtuk közvetlenül)

## Használat

A projekt gyökérkönyvtárából:

```bash
pip install -r requirements.txt
python main.py
```

A program:

1. Betölti a rating és movie adatokat
2. Felosztja train/testre (`config.TEST_SIZE` szerint, alapértelmezetten 70/30)
3. User-based és item-based CF modellt tanít
4. A teszt halmazon előrejelzést készít mindkét módszerrel
5. Kiírja az RMSE, MAE, Precision@10 és Recall@10 metrikákat összehasonlító táblázatban

A teljes futtatás (kb. 29k teszt pár) több percig is eltarthat; a modulok hasonlósági cache-t használnak a gyorsítás érdekében. Gyors próbához a `config.py`-ban állítsd be pl. `TEST_SAMPLE_SIZE = 500`-at.

## Konfiguráció

A `config.py` fájlban módosítható:

- `TEST_SIZE` – teszt arány (alapértelmezett 0,3; a tanulmányban 0,3 / 0,25 / 0,2 mellett is futtattuk)
- `K_NEIGHBORS` – k legközelebbi szomszéd (alapértelmezett 30)
- `MIN_COMMON` – minimum közös értékelés a hasonlósághoz (alapértelmezett 2)
- `RANDOM_STATE` – véletlenség reprodukálhatósága

## Modulok – mi mit csinál

- **config.py** – Összes beállítás egy helyen: adat útvonalak, train/test arány, k szomszéd, minimum közös értékelés, relevancia küszöb, stb.
- **data_loader.py** – CSV-k betöltése (ratings, movies), oszlopok ellenőrzése, tisztítás; `split_data()` a train–test felosztáshoz; `build_user_item_matrix()` pivot tábla.
- **user_based.py** – Felhasználó-alapú CF: Pearson hasonlóság userek között, k legközelebbi user értékelésének súlyozott átlaga; fallback user átlagra; hasonlósági cache.
- **item_based.py** – Elem-alapú CF: Pearson a filmek között, k legközelebbi filmre adott user-értékelések súlyozott átlaga; fallback item átlagra; cache.
- **evaluation.py** – RMSE, MAE (előrejelzési hiba); Precision@k, Recall@k (releváns = rating ≥ 4); mindkét modellre.
- **main.py** – Futtatható pipeline: betöltés → split → cold-start szűrés → fit user + item CF → predict_batch → evaluate → összehasonlító táblázat kiírása.

## Licenc

A repó forráskódját oktatási/kutatási célra osztjuk meg, külön szoftverlicenc nélkül. A felhasznált MovieLens adatkészletre a [GroupLens használati feltételei](http://grouplens.org/datasets/) vonatkoznak.
