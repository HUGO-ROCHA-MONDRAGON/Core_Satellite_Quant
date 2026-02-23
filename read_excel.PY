import pandas as pd
import numpy as np

path = "core_ETF.xlsx"
sheet = 0

meta = pd.read_excel(path, sheet_name=sheet, header=None, nrows=3)
tickers = meta.iloc[0].tolist()

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

tickers = [clean(x) for x in tickers]

df = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)

# 1) calendrier référence = 1ère colonne date
ref_raw = df.iloc[:, 0]
ref_num = pd.to_numeric(ref_raw, errors="coerce")
mask_excel = ref_num.between(20000, 60000)

ref_dates = ref_raw.copy()
ref_dates.loc[mask_excel] = pd.to_datetime(ref_num[mask_excel], unit="D", origin="1899-12-30")
ref_dates.loc[~mask_excel] = pd.to_datetime(ref_raw[~mask_excel], errors="coerce")

# garder seulement les lignes où la date est valide
valid = ref_dates.notna()
df = df.loc[valid].reset_index(drop=True)
ref_dates = pd.to_datetime(ref_dates.loc[valid]).reset_index(drop=True)

# 2) wide = index = ref_dates, colonnes = tickers, valeurs = colonnes "Price" uniquement
wide = pd.DataFrame(index=ref_dates)

for j in range(0, df.shape[1], 2):
    price_col = j + 1
    if price_col >= df.shape[1]:
        break

    # ticker associé à la colonne prix
    tkr = tickers[price_col] if price_col < len(tickers) and tickers[price_col] not in ("", "Date") else f"UNKNOWN_{j//2}"

    px = pd.to_numeric(df.iloc[:, price_col], errors="coerce").values
    wide[tkr] = px

wide = wide.sort_index()

# 3) Forcer les dates : combler trous intra-séries (si un ETF manque un jour)
wide = wide.ffill()

print(wide.head())
print("index min/max:", wide.index.min(), wide.index.max())
print("colonnes:", wide.shape[1])