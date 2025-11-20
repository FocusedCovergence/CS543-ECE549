import re


FITZPATRICK_TYPES = {
    1: "ivory white",
    2: "white",
    3: "white to light brown",
    4: "light brown",
    5: "brown",
    6: "dark brown",
}


def get_fitz(fitz_scale, fitz_centaur):
    fitz = fitz_scale if fitz_scale >= 1 else fitz_centaur
    color = FITZPATRICK_TYPES[fitz]
    return fitz, color


def format_as_caption(row):
    fitz_scale, fitz_centaur = row.fitzpatrick_scale, row.fitzpatrick_centaur
    if fitz_scale < 1 and fitz_centaur < 1:
        raise ValueError("No color information provided")
    fitz, color = get_fitz(fitz_scale, fitz_centaur)
    nine_partition, three_partition = (
        row.nine_partition_label,
        row.three_partition_label,
    )
    label = row.label

    return (
        f"A dermatology image showing {label} ({nine_partition}, {three_partition})"
        f" on a patient with {color} skin (Fitzpatrick {fitz})."
    )


def modify_caption_skin_tone(caption: str, target_fitz: int):
    split_token = "on a patient with"
    color = FITZPATRICK_TYPES[target_fitz]
    prefix = caption.split(split_token)[0].rstrip()
    return f"{prefix} on a patient with {color} skin (Fitzpatrick {target_fitz})."


def main():
    from pathlib import Path
    import pandas as pd
    from constants import get_cfg_defaults

    cfg = get_cfg_defaults()
    root = Path(cfg.DATA.ROOT)
    path = root / cfg.DATA.FITZPATRICK_17K.CSV
    df = pd.read_csv(path)

    print(df.head())


if __name__ == "__main__":
    main()
