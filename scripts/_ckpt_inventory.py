import os
import glob
from collections import defaultdict

root = r"C:/Users/28129/Desktop/论文改写整理发表？/xjtu_sy_rul"
allpt = glob.glob(os.path.join(root, "artifacts", "**", "checkpoints", "*.pt"), recursive=True)
print("total .pt:", len(allpt))
by = defaultdict(int)
for p in allpt:
    pp = p.replace("\\", "/")
    marker = "/xjtu_sy_rul/"
    rel = pp.split(marker)[1] if marker in pp else pp
    parts = rel.split("/")
    i = parts.index("checkpoints") if "checkpoints" in parts else -1
    key = "/".join(parts[:i]) if i > 0 else rel
    by[key] += 1
for k in sorted(by):
    print(f"{by[k]:4d}  {k}")
