"""
P3: Feedback Loop Stability Experiment
=======================================
Agent4Rec users simulate -> ratings fed back -> retrain LightGCN -> re-simulate.
Track KL divergence of rating distribution vs real across rounds.

Usage:
    cd /opt/workspace/hehe/Agent4Rec
    /opt/workspace/hehe/agent4rec_venv/bin/python /tmp/p3_feedback_loop.py
"""
import os, sys, shutil, pickle, subprocess, time, json
import numpy as np
from collections import Counter, defaultdict

BASE = "/opt/workspace/hehe/Agent4Rec"
VENV_PYTHON = "/opt/workspace/hehe/agent4rec_venv/bin/python"
N_ROUNDS = 5
N_AVATARS = 100
MAX_PAGES = 3
CF_DATA_ORIG = os.path.join(BASE, "datasets/ml-1m/cf_data")
RESULTS_DIR = os.path.join(BASE, "p3_results")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(BASE)

# ========== Helpers ==========

def load_real_ratings():
    """Load real MovieLens 1-5 star ratings for KL baseline."""
    ratings = []
    with open(os.path.join(BASE, "datasets/ml-1m/raw_data/ratings.dat"), encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            if len(p) >= 3:
                ratings.append(int(p[2]))
    return ratings


def rating_distribution(ratings):
    """Return normalized distribution over 1-5."""
    c = Counter(ratings)
    total = sum(c.values())
    if total == 0:
        return {k: 0.2 for k in range(1, 6)}
    return {k: c.get(k, 0) / total for k in range(1, 6)}


def kl_divergence(p_dict, q_dict, eps=1e-10):
    """KL(P || Q) where P=sim, Q=real."""
    kl = 0.0
    for k in range(1, 6):
        p = p_dict.get(k, 0) + eps
        q = q_dict.get(k, 0) + eps
        kl += p * np.log(p / q)
    return kl


def extract_sim_ratings(sim_name):
    """Extract all 1-5 star ratings from behavior/*.pkl."""
    ratings = []
    watched_per_user = defaultdict(list)
    bdir = os.path.join(BASE, "storage/ml-1m/LightGCN", sim_name, "behavior")
    if not os.path.exists(bdir):
        return ratings, watched_per_user
    for fname in os.listdir(bdir):
        if not fname.endswith(".pkl"):
            continue
        aid = int(fname.replace(".pkl", ""))
        with open(os.path.join(bdir, fname), "rb") as f:
            d = pickle.load(f)
        for pg in d.values():
            for r in pg.get("rating", []):
                if r:
                    ratings.append(int(r))
            for wid in pg.get("watch_id", []):
                watched_per_user[aid].append(wid)
    return ratings, watched_per_user


def augment_cf_data(round_num, watched_per_user):
    """Create augmented cf_data for next round by appending sim-watched items to train.txt."""
    src_dir = CF_DATA_ORIG if round_num == 0 else os.path.join(RESULTS_DIR, "cf_data_round_{}".format(round_num - 1))
    dst_dir = os.path.join(RESULTS_DIR, "cf_data_round_{}".format(round_num))
    os.makedirs(dst_dir, exist_ok=True)

    # Copy all files
    for fname in os.listdir(src_dir):
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    # Augment train.txt
    train_path = os.path.join(dst_dir, "train.txt")
    lines = []
    with open(train_path) as f:
        for line in f:
            lines.append(line.strip())

    augmented = 0
    for uid, items in watched_per_user.items():
        if uid < len(lines):
            existing = set(lines[uid].split()[1:])  # skip user_id
            new_items = [str(i) for i in items if str(i) not in existing]
            if new_items:
                lines[uid] = lines[uid] + " " + " ".join(new_items)
                augmented += 1

    with open(train_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print("  Augmented {}/{} users in round {} train.txt".format(augmented, len(watched_per_user), round_num))
    return dst_dir


def retrain_lightgcn(cf_data_dir, round_num):
    """Retrain LightGCN with augmented data. Returns checkpoint path."""
    # Create a symlink dataset that points to our augmented cf_data
    dataset_name = "ml-1m_loop_r{}".format(round_num)
    dataset_dir = os.path.join(BASE, "datasets", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Symlink cf_data
    cf_link = os.path.join(dataset_dir, "cf_data")
    if os.path.exists(cf_link):
        os.remove(cf_link)
    os.symlink(cf_data_dir, cf_link)

    # Also need raw_data for the simulation (movie info etc)
    raw_link = os.path.join(dataset_dir, "raw_data")
    if os.path.exists(raw_link):
        os.remove(raw_link)
    os.symlink(os.path.join(BASE, "datasets/ml-1m/raw_data"), raw_link)

    sim_link = os.path.join(dataset_dir, "simulation")
    if os.path.exists(sim_link):
        os.remove(sim_link)
    os.symlink(os.path.join(BASE, "datasets/ml-1m/simulation"), sim_link)

    # Run training
    save_id = "loop_r{}".format(round_num)
    cmd = [
        VENV_PYTHON, "train_recommender.py",
        "--no_wandb",
        "--clear_checkpoints",
        "--dataset", dataset_name,
        "--data_path", os.path.join(BASE, "datasets") + "/",
        "--modeltype", "LightGCN",
        "--n_layers", "2",
        "--patience", "5",
        "--epoch", "100",
        "--saveID", save_id,
    ]
    print("  Training LightGCN round {}...".format(round_num))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.join(BASE, "recommenders"),
                          capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    print("  Training done in {:.1f}s".format(dt))

    if result.returncode != 0:
        print("  TRAINING STDERR:", result.stderr[-500:])
        return None

    # Find the checkpoint
    weights_base = os.path.join(BASE, "recommenders/weights", dataset_name, "LightGCN")
    if os.path.exists(weights_base):
        subdirs = [d for d in os.listdir(weights_base)
                   if save_id in d and os.path.isdir(os.path.join(weights_base, d))]
        if subdirs:
            ckpt_dir = os.path.join(weights_base, subdirs[0])
            # Copy to the standard location for Agent4Rec to pick up
            dst = os.path.join(BASE, "recommenders/weights/ml-1m/LightGCN/Saved")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(ckpt_dir, dst)
            print("  Checkpoint copied to {}".format(dst))
            return dst

    print("  WARNING: checkpoint not found")
    return None


def run_simulation(round_num):
    """Run Agent4Rec simulation for one round."""
    sim_name = "p3_round_{}".format(round_num)
    # Clean previous
    sim_dir = os.path.join(BASE, "storage/ml-1m/LightGCN", sim_name)
    if os.path.exists(sim_dir):
        shutil.rmtree(sim_dir)

    env = os.environ.copy()
    env["HF_HOME"] = "/opt/workspace/hehe/hf_cache"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    cmd = [
        VENV_PYTHON, "-u", "main.py",
        "--n_avatars", str(N_AVATARS),
        "--max_pages", str(MAX_PAGES),
        "--execution_mode", "parallel",
        "--simulation_name", sim_name,
    ]
    print("  Running simulation round {} ({} avatars x {} pages)...".format(round_num, N_AVATARS, MAX_PAGES))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, env=env, timeout=1800)
    dt = time.time() - t0
    print("  Simulation done in {:.1f}s".format(dt))

    if result.returncode != 0:
        print("  SIM STDERR (last 500):", result.stderr[-500:])

    return sim_name


# ========== Main Loop ==========

def main():
    real_ratings = load_real_ratings()
    real_dist = rating_distribution(real_ratings)
    print("Real distribution:", {k: round(v, 4) for k, v in real_dist.items()})
    print("Real mean:", round(np.mean(real_ratings), 3))
    print()

    results = []

    for rnd in range(N_ROUNDS):
        print("=" * 60)
        print("ROUND {}".format(rnd))
        print("=" * 60)

        # 1. Run simulation
        sim_name = run_simulation(rnd)

        # 2. Extract ratings
        sim_ratings, watched_per_user = extract_sim_ratings(sim_name)
        sim_dist = rating_distribution(sim_ratings)
        kl = kl_divergence(sim_dist, real_dist)
        sim_mean = np.mean(sim_ratings) if sim_ratings else 0

        print("  Sim ratings: {} total".format(len(sim_ratings)))
        print("  Sim distribution:", {k: round(v, 4) for k, v in sim_dist.items()})
        print("  Sim mean: {:.3f}".format(sim_mean))
        print("  KL(sim || real) = {:.4f}".format(kl))

        round_result = {
            "round": rnd,
            "n_ratings": len(sim_ratings),
            "sim_dist": sim_dist,
            "sim_mean": float(sim_mean),
            "kl": float(kl),
            "n_users_watched": len(watched_per_user),
        }
        results.append(round_result)

        # 3. Augment training data
        cf_dir = augment_cf_data(rnd, watched_per_user)

        # 4. Retrain LightGCN
        retrain_lightgcn(cf_dir, rnd)

        print()

    # ========== Final Summary ==========
    print("=" * 60)
    print("P3 FEEDBACK LOOP SUMMARY")
    print("=" * 60)
    print("{:>6} {:>10} {:>10} {:>10} {:>10}".format(
        "Round", "N_ratings", "Mean", "KL", "Users"))
    for r in results:
        print("{:>6} {:>10} {:>10.3f} {:>10.4f} {:>10}".format(
            r["round"], r["n_ratings"], r["sim_mean"], r["kl"], r["n_users_watched"]))

    print()
    print("KL trend: {} -> {}".format(
        round(results[0]["kl"], 4), round(results[-1]["kl"], 4)))
    delta_kl = results[-1]["kl"] - results[0]["kl"]
    if delta_kl < -0.01:
        verdict = "CONVERGING (KL decreasing)"
    elif delta_kl > 0.01:
        verdict = "DIVERGING (KL increasing)"
    else:
        verdict = "STABLE (KL roughly constant)"
    print("Verdict:", verdict)

    # Save results
    out_path = os.path.join(RESULTS_DIR, "p3_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to", out_path)


if __name__ == "__main__":
    main()
