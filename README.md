# Časové řady a prediktivní modely

Tento projekt se zaměřuje na tvorbu prediktivních modelů na základě historických dat s cílem předpovědět budoucí prodeje. Takové řešení pomáhá lépe plánovat zásoby a nákupy u dodavatelů.

## Cíl projektu

Vytvořit model, který na základě dostupných dat dokáže předpovídat budoucí prodeje produktů. To umožní efektivnější řízení zásob a zamezení výpadků nebo přebytků zboží.

## Použité datové sady

Projekt využívá následující datasety:

- `sell_data.csv` – Denní záznamy o prodejích
- `stock.csv` – Stav zásob
- `marketing_campaign.csv` – Informace o marketingových kampaních
- `products.csv` – Informace o produktech

Součástí jsou také grafy a vizualizace pro lepší interpretaci dat.

## Použité technologie a knihovny

Analýza a modelování byly prováděny v jazyce Python za použití těchto knihoven:

- `pandas` – Práce s daty
- `numpy` – Matematické výpočty
- `matplotlib` – Vizualizace dat
- `statsmodels` – Modely časových řad
- `scikit-learn` – Strojové učení
- `xgboost` – Gradient boosting
- `prophet` – Knihovna pro predikci časových řad od Meta

## Použité prediktivní modely

Vyzkoušeli jsme různé přístupy k predikci prodejů:

- **Prophet**
- **XGBoost**
- **RandomForest Regressor**

Výsledky byly porovnávány s reálnými hodnotami pomocí grafů.

## Co ovlivnilo přesnost modelů?

Přesnost predikce ovlivnilo několik klíčových faktorů:

- **Sezónnost** – Opakující se vzory v prodejích
- **Výpadky skladu** – Neúplné údaje kvůli nedostatku zboží
- **Neúplná data** – Chybějící záznamy v historických datech
- **Typ modelu** – Výběr algoritmu měl vliv na kvalitu predikce

## Závěr

Prediktivní modely ukázaly slibné výsledky. Jejich přesnost je však závislá na kvalitě vstupních dat a konkrétním způsobu modelování. Projekt dokazuje, že datová predikce může výrazně přispět k optimalizaci zásobování.
