import pandas as pd
import numpy as np

path = "core_ETF.xlsx"
sheet = 0

# --- 1) Meta : 3 lignes (tickers / bucket / description)
meta = pd.read_excel(path, sheet_name=sheet, header=None, nrows=3)

tickers = meta.iloc[0].tolist()
buckets = meta.iloc[1].tolist()
descs   = meta.iloc[2].tolist()

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

tickers = [clean(x) for x in tickers]
buckets = [clean(x) for x in buckets]
descs   = [clean(x) for x in descs]

# --- 2) Données (à partir ligne 4)
df = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)

out = []
for j in range(0, df.shape[1], 2):
    # ticker souvent au-dessus de la colonne prix (j+1), sinon fallback j
    cand = []
    if j < len(tickers): cand.append(tickers[j])
    if j+1 < len(tickers): cand.append(tickers[j+1])

    # choisir le premier non vide / non "Date"
    tkr = next((c for c in cand if c not in ("", "Date")), f"UNKNOWN_{j//2}")

    bucket = buckets[j+1] if j+1 < len(buckets) else buckets[j]
    desc   = descs[j+1] if j+1 < len(descs) else descs[j]

    tmp = df.iloc[:, [j, j+1]].copy()
    tmp.columns = ["date", "px"]
    tmp["ticker"] = tkr
    tmp["bucket"] = bucket
    tmp["description"] = desc
    out.append(tmp)

long = pd.concat(out, ignore_index=True)

# --- 3) Nettoyage
long["date"] = pd.to_datetime(long["date"], errors="coerce")
long["px"] = pd.to_numeric(long["px"], errors="coerce")
long = long.dropna(subset=["date", "px"])

# enlever les lignes où ticker est vide / Date
long = long[~long["ticker"].isin(["", "Date"])]

# dédupliquer si nécessaire (même date,ticker)
long = (long.sort_values(["ticker", "date"])
            .drop_duplicates(subset=["date", "ticker"], keep="last"))

# --- 4) Pivot wide (date x ticker)
wide = (long.pivot(index="date", columns="ticker", values="px")
            .sort_index())

print(long.head())
print(wide.head())
print("tickers uniques:", long["ticker"].nunique(), " | colonnes wide:", wide.shape[1])