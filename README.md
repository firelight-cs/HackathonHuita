# Hackathon2025

A Python-based project developed during a hackathon. This project utilizes Jupyter notebooks and is structured using a virtual environment for dependency management.
**Tested and developed with Python 3.12.**

## 📁 Project Structure

```
Hackathon2025/
├── models/                               # Python scripts for modular ML models           
│   ├── prophet.py                        # Time series model using Facebook's Prophet
│   ├── randomforestmodel.py              # Random Forest model logic
│   └── xgboost_model.py                  # XGBoost model script, good for structured data          
├── Casove rady a prediktivni modely.pdf  # Presentation
├── prophet.ipynb                 # Notebook exploring time series forecasting with Prophet
├── randomforestmodel.ipynb       # Jupyter notebook for Random Forest experimentation
├── README.md                     # Project overview, setup instructions, and usage guide
├── requirements.txt              # List of Python dependencies (e.g., scikit-learn, xgboost)
└── xgboost_model.ipynb           # Notebook using XGBoost for modeling and evaluation
```

## 🚀 Setup Instructions

Follow these steps to get the project up and running on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/firelight-cs/Hahathon2025.git
cd Hackathon2025
```

### 2. Create a Virtual Environment

It’s recommended to use a virtual environment to avoid conflicts with global packages.

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### On Windows (Command Prompt)

```cmd
.venv\Scripts\activate
```

#### On macOS/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Virtual Environment to Jupyter

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

After running this, you should see `Python (.venv)` as an option in your Jupyter kernels.

---

## 🧪 Running Notebooks

Start Jupyter Lab or Notebook:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

---

## 🛠️ Troubleshooting

* If the `.venv` kernel doesn't show up in Jupyter:

  * Make sure the virtual environment is activated.
  * Run the `ipykernel install` command again.
  * Restart Jupyter.

---


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

- **Prophet** – model pro predikci časových řad vyvinutý společností Meta. Je navržený tak, aby zvládal sezónní a trendové změny a zároveň byl snadno použitelný i bez hlubokých znalostí statistiky.
- **XGBoost** – výkonný algoritmus založený na principu gradientního boostingu. Je velmi efektivní při práci s tabulkovými daty a dosahuje špičkových výsledků v predikčních úlohách.
- **RandomForest Regressor** – model založený na souboru rozhodovacích stromů, který kombinuje více stromů k dosažení stabilní a přesné predikce. Je robustní vůči šumu v datech a přetrénování.

Výsledky byly porovnávány s reálnými hodnotami pomocí grafů.

## Co ovlivnilo přesnost modelů?

Přesnost predikce ovlivnilo několik klíčových faktorů:

- **Sezónnost** – Opakující se vzory v prodejích
- **Výpadky skladu** – Neúplné údaje kvůli nedostatku zboží
- **Neúplná data** – Chybějící záznamy v historických datech
- **Typ modelu** – Výběr algoritmu měl vliv na kvalitu predikce

## Závěr

Prediktivní modely ukázaly slibné výsledky. Jejich přesnost je však závislá na kvalitě vstupních dat a konkrétním způsobu modelování. Projekt dokazuje, že datová predikce může výrazně přispět k optimalizaci zásobování.
