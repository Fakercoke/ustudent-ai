"""Evaluate chunk size / overlap against data/golden/rag-eval.json.

Distance alone is a misleading signal — tiny chunks score the best distances
while returning truncated, unusable text. This scores each configuration on
what actually matters downstream:

  hit@3        top-3 中至少一块包含标准答案的关键事实  (召回率，越高越好)
  self@1       top-1 单独一块就包含关键事实            (答案自足性，越高越好)
  d_yes        有答案问题的平均 top-1 距离             (越低越好)
  d_no         无答案问题的平均 top-1 距离             (越高越好)
  gap          d_no - d_yes                            (可分性，决定兜底阈值好不好定)
  chars@3      top-3 拼起来的字符数                    (塞进 prompt 的成本，越低越好)

Usage:  python eval_chunking.py
"""
from __future__ import annotations

import json

import chromadb

from index_handbook import CORPUS, DB_PATH, ROOT, chunk_text

# Exact substrings that appear verbatim in the corpus. A retrieved chunk
# "contains the answer" only if one of these survives the chunk boundary.
KEY_FACT = {
    "g1": "120 credits",
    "g2": "18 credits",
    "g3": "30 completed credit hours",
    "g4": "credit-weighted average",
    "g5": "do not affect GPA",
    "g6": "full refund and no academic record",
}

CONFIGS = [
    (100, 16),
    (300, 50),
    (600, 0),      # 同样 600，但完全不重叠 —— 单独看 overlap 的作用
    (600, 100),    # starter 默认
    (600, 300),    # 重叠 50%
    (1200, 200),
    (3000, 500),
]


def load_golden() -> list[dict]:
    path = ROOT / "data/golden/rag-eval.json"
    return json.loads(path.read_text())["items"]


def build(client, size: int, overlap: int) -> tuple[str, int]:
    name = f"eval_{size}_{overlap}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    total = 0
    for rel in CORPUS:
        chunks = chunk_text((ROOT / rel).read_text(), size=size, overlap=overlap)
        col.add(documents=chunks, ids=[f"{rel}::{i}" for i in range(len(chunks))])
        total += len(chunks)
    return name, total


def evaluate(col, golden: list[dict], verbose: bool = True) -> dict:
    hits = selfs = answerable = 0
    d_yes: list[float] = []
    d_no: list[float] = []
    chars: list[int] = []

    for item in golden:
        res = col.query(query_texts=[item["q"]], n_results=3)
        docs = res["documents"][0]
        top1 = res["distances"][0][0]
        chars.append(sum(len(d) for d in docs))

        if not item["in_handbook"]:
            d_no.append(top1)
            if verbose:
                print(f"      [无答案题] top1距离 {top1:.3f}   {item['q'][:44]}")
            continue

        d_yes.append(top1)
        answerable += 1
        fact = KEY_FACT[item["id"]]
        in3 = any(fact in d for d in docs)
        in1 = fact in docs[0]
        if in3:
            hits += 1
        if in1:
            selfs += 1
        if verbose:
            print(
                f"      前3{'✅' if in3 else '❌'} 第1{'✅' if in1 else '❌'}"
                f"  top1距离 {top1:.3f}   {item['q'][:40]}"
            )
            print(f"                找的关键字: \"{fact}\"")

    return {
        "hit@3": hits / answerable,
        "self@1": selfs / answerable,
        "d_yes": sum(d_yes) / len(d_yes),
        "d_no": sum(d_no) / len(d_no),
        "gap": sum(d_no) / len(d_no) - sum(d_yes) / len(d_yes),
        "chars@3": sum(chars) / len(chars),
    }


def main() -> None:
    golden = load_golden()
    client = chromadb.PersistentClient(path=DB_PATH)
    rows = []

    print(f"标准答案集: data/golden/rag-eval.json  共 {len(golden)} 道题 "
          f"({sum(i['in_handbook'] for i in golden)} 道有答案 / "
          f"{sum(not i['in_handbook'] for i in golden)} 道故意没答案)")
    print(f"要测 {len(CONFIGS)} 组参数配置\n")

    for idx, (size, overlap) in enumerate(CONFIGS, 1):
        print("=" * 78)
        print(f"[{idx}/{len(CONFIGS)}]  size={size}  overlap={overlap}")
        print("=" * 78)
        print("  ① 按这组参数切块 + 建库（每块都要过一遍 embedding 模型，慢）...")
        name, n = build(client, size, overlap)
        print(f"     -> 切出 {n} 块，已存入临时 collection '{name}'")
        print("  ② 拿 8 道标准题去考它:")
        m = evaluate(client.get_collection(name), golden)
        print(f"  ③ 这一组的成绩: 前3命中 {m['hit@3']:.2f} | 第1完整 {m['self@1']:.2f} "
              f"| 有答案距离 {m['d_yes']:.3f} | 无答案距离 {m['d_no']:.3f} "
              f"| 给AI的字符数 {m['chars@3']:.0f}")
        rows.append((size, overlap, n, m))
        client.delete_collection(name)
        print(f"  ④ 删掉临时 collection '{name}'，不留垃圾\n")

    print("\n" + "=" * 82)
    print(f"{'size':>6}{'overlap':>9}{'chunks':>8}{'hit@3':>8}{'self@1':>8}"
          f"{'d_yes':>8}{'d_no':>8}{'gap':>8}{'chars@3':>10}")
    print("-" * 82)
    for size, overlap, n, m in rows:
        print(f"{size:>6}{overlap:>9}{n:>8}"
              f"{m['hit@3']:>8.2f}{m['self@1']:>8.2f}"
              f"{m['d_yes']:>8.3f}{m['d_no']:>8.3f}{m['gap']:>8.3f}"
              f"{m['chars@3']:>10.0f}")
    print("=" * 82)
    print("hit@3 / self@1 越高越好；d_yes 越低越好；gap 越大越好；chars@3 越低越省钱")


if __name__ == "__main__":
    main()
