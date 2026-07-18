from __future__ import annotations
import argparse
import csv
import hashlib
import html
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Iterator

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import CFG

SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EN_HINTS = {"the", "is", "of", "and", "to", "in", "for", "with"}
SW_HINTS = {"ya", "na", "kwa", "katika", "ni", "wa", "hii", "ambayo"}

def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()

def likely_language(text: str, lang: str) -> bool:
    """Very lightweight safety filter; it avoids an expensive external model."""
    words = set(re.findall(r"[A-Za-zÀ-ÿ]+", text.lower()))
    if len(words) < 3:
        return True
    en = len(words & EN_HINTS)
    sw = len(words & SW_HINTS)
    return en >= sw if lang == "en" else sw >= en or sw > 0

def valid_pair(src: str, tgt: str) -> tuple[bool, str]:
    if not src or not tgt:
        return False, "empty"
    sw, tw = src.split(), tgt.split()
    if len(sw) < CFG.min_words or len(tw) < CFG.min_words:
        return False, "too_short"
    if len(sw) > CFG.max_words or len(tw) > CFG.max_words:
        return False, "too_long"
    ratio = max(len(sw) / max(len(tw), 1), len(tw) / max(len(sw), 1))
    if ratio > CFG.max_length_ratio:
        return False, "length_ratio"
    if src == tgt and len(src) > 20:
        return False, "identical"
    return True, "ok"

def iter_parallel(src_path: Path, tgt_path: Path) -> Iterator[tuple[int, str, str]]:
    with src_path.open(encoding="utf-8", errors="replace") as sf, \
         tgt_path.open(encoding="utf-8", errors="replace") as tf:
        line_no = 0
        while True:
            s = sf.readline()
            t = tf.readline()
            if not s and not t:
                break
            line_no += 1
            if not s or not t:
                raise ValueError(
                    f"Line-count mismatch at line {line_no}: {src_path.name} vs {tgt_path.name}"
                )
            yield line_no, normalize_text(s), normalize_text(t)

def prepare() -> None:
    CFG.processed_dir.mkdir(parents=True, exist_ok=True)
    CFG.split_dir.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    audit: dict[str, dict[str, int]] = {}

    for corpus, src_name, tgt_name in CFG.corpora:
        src_path, tgt_path = CFG.original_dir / src_name, CFG.original_dir / tgt_name
        if not src_path.exists() or not tgt_path.exists():
            print(f"[SKIP] Missing pair: {src_name}, {tgt_name}")
            continue
        stats = {"read": 0, "kept": 0, "duplicate": 0}
        audit[corpus] = stats
        for line_no, src, tgt in tqdm(iter_parallel(src_path, tgt_path), desc=corpus):
            stats["read"] += 1
            ok, reason = valid_pair(src, tgt)
            if not ok:
                stats[reason] = stats.get(reason, 0) + 1
                continue
            key = hashlib.sha1((src + "\t" + tgt).encode()).hexdigest()
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            rows.append({"source": src, "target": tgt, "corpus": corpus, "line_no": line_no})
            stats["kept"] += 1

    if not rows:
        raise RuntimeError(
            f"No data found. Copy the .en and .sw files into {CFG.original_dir}"
        )

    df = pd.DataFrame(rows).sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    df.to_csv(CFG.processed_dir / "parallel_clean.csv", index=False)

    train, temp = train_test_split(
        df, test_size=CFG.valid_ratio + CFG.test_ratio, random_state=CFG.seed
    )
    relative_test = CFG.test_ratio / (CFG.valid_ratio + CFG.test_ratio)
    valid, test = train_test_split(temp, test_size=relative_test, random_state=CFG.seed)

    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        part.reset_index(drop=True).to_csv(CFG.split_dir / f"{name}.csv", index=False)

    pd.DataFrame([
        {"corpus": corpus, **stats} for corpus, stats in audit.items()
    ]).fillna(0).to_csv(CFG.results_dir / "preprocessing_audit.csv", index=False)

    print(f"Total clean pairs: {len(df):,}")
    print(f"Train={len(train):,}, validation={len(valid):,}, test={len(test):,}")

def copy_uploaded(upload_dir: Path) -> None:
    CFG.original_dir.mkdir(parents=True, exist_ok=True)
    for _, en_name, sw_name in CFG.corpora:
        for name in (en_name, sw_name):
            src = upload_dir / name
            if src.exists():
                shutil.copy2(src, CFG.original_dir / name)
                print(f"Copied {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-from", type=Path, help="Directory containing attached corpus files")
    args = parser.parse_args()
    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_from:
        copy_uploaded(args.copy_from)
    prepare()
