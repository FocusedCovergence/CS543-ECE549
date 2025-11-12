import pandas as pd
from pathlib import Path
from urllib.parse import urlparse


csv_path = r"Fitzpatrick17k_original_files/fitzpatrick17k-main/fitzpatrick17k-main/fitzpatrick17k.csv"
df = pd.read_csv(csv_path)


out = Path(r"Fitzpatrick17k_current_files/Fitzpatrick17k_images")


existing = {p.name for p in out.iterdir() if p.is_file()}
print("img count:", len(existing))

def expected_name(row):
    u = row["url"]
    if pd.isna(u):
        return None

    md5 = row.get("md5hash")
    if (md5 is not None) and (not pd.isna(md5)) and str(md5).strip():
        return str(md5).strip() + ".jpg"
    else:
        return Path(urlparse(str(u)).path).name

mask = []
for _, row in df.iterrows():
    fname = expected_name(row)
    if fname is not None and fname in existing:
        mask.append(True)
    else:
        mask.append(False)

df_filtered = df[mask].copy()
print("Init lines:", len(df), "kept lines:", len(df_filtered))

df_filtered.to_csv("Fitzpatrick17k_current_files/fitzpatrick17k_downloaded.csv", index=False)