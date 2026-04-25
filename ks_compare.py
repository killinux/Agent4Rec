import pickle, glob, os, numpy as np
from collections import Counter
from scipy.stats import ks_2samp

BASE = "/opt/workspace/hehe/Agent4Rec"

def load(name):
    r = []
    for fp in sorted(glob.glob(os.path.join(BASE, "storage/ml-1m/LightGCN", name, "behavior/*.pkl"))):
        with open(fp, "rb") as f:
            d = pickle.load(f)
        for pg in d.values():
            r.extend([int(x) for x in pg.get("rating", []) if x])
    return np.array(r)

def load_real():
    with open(os.path.join(BASE, "datasets/ml-1m/raw_data/user_id_map.pkl"), "rb") as f:
        m = pickle.load(f)
    s = set(m.keys())
    r = []
    with open(os.path.join(BASE, "datasets/ml-1m/raw_data/ratings.dat"), encoding="latin-1") as f:
        for l in f:
            p = l.strip().split("::")
            if len(p) >= 3 and int(p[0]) in s:
                r.append(int(p[2]))
    return np.array(r)

real = load_real()
rd = Counter(real)
rn = len(real)

fmt = "{:>15} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6}"
print(fmt.format("Version", "N", "1*", "2*", "3*", "4*", "5*", "KS-D", "Mean"))
print("-" * 80)
print(fmt.format("Real", rn,
    "{:.1f}%".format(rd[1]/rn*100), "{:.1f}%".format(rd[2]/rn*100),
    "{:.1f}%".format(rd[3]/rn*100), "{:.1f}%".format(rd[4]/rn*100),
    "{:.1f}%".format(rd[5]/rn*100), "-", "{:.3f}".format(real.mean())))

for v in ["plus_full", "calibrated_v1", "calibrated_v2", "calibrated_v3", "calibrated_v4"]:
    p = os.path.join(BASE, "storage/ml-1m/LightGCN", v, "behavior")
    if not os.path.exists(p):
        continue
    sim = load(v)
    if len(sim) == 0:
        continue
    c = Counter(sim)
    n = len(sim)
    D, pp = ks_2samp(sim, real)
    print(fmt.format(v, n,
        "{:.1f}%".format(c[1]/n*100), "{:.1f}%".format(c[2]/n*100),
        "{:.1f}%".format(c[3]/n*100), "{:.1f}%".format(c[4]/n*100),
        "{:.1f}%".format(c[5]/n*100), "{:.4f}".format(D), "{:.3f}".format(sim.mean())))
