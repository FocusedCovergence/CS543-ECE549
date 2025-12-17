import pandas as pd, requests, concurrent.futures as cf
from pathlib import Path

file_path = "/home/havens3/CS543-ECE549/data/fitzpatrick17k/fitzpatrick17k.csv"
df = pd.read_csv(file_path)
out = Path("Fitzpatrick17k_images")
out.mkdir(parents=True, exist_ok=True)


def download(row):
    u = row["url"]
    if pd.isna(u):
        return
    name = (
        (str(row.get("md5hash")).strip() + ".jpg")
        if not pd.isna(row.get("md5hash")) and str(row.get("md5hash")).strip()
        else Path(str(u)).name
    )
    p = out / name
    for _ in range(3):
        try:
            with requests.Session() as s:
                r = s.get(
                    u, timeout=30, stream=True, headers={"User-Agent": "Mozilla/5.0"}
                )
                r.raise_for_status()
                with open(p, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            return
        except Exception:
            pass


with cf.ThreadPoolExecutor(max_workers=64) as ex:
    list(ex.map(download, [r for _, r in df.iterrows()]))
