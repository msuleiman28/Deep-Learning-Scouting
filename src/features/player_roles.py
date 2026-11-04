"""Player-role clustering within broad positions (Plan.md section 5, milestone 1).

Input: ``data/processed/player_features.csv`` (Wyscout 2017/18, one row per player).
Time window: the 2017/18 season only, which ended before every milestone 1
transfer, so roles are pre-transfer.

Method:
- Players with >= MIN_MINUTES league minutes, so per-90 stats aren't noise.
- Defenders (DF), midfielders (MD) and forwards (FW) are clustered separately;
  goalkeepers (GK) form one role.
- ROLE_FEATURES describe *how* a player plays, not how well: goals, assists,
  duels won and fouls are left out (outcome-driven or discipline), and so is
  avg_y (which side, not a role).
- Per-90 counts are skewed, so they get log1p before z-scoring within each
  position; otherwise a few extreme players dominate k-means distances.
- k-means with a fixed seed; k is chosen per position from silhouette score,
  bootstrap stability (adjusted Rand index) and whether clusters can be named.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

MIN_MINUTES = 450
SEED = 42
OUTFIELD = ["DF", "MD", "FW"]

ROLE_FEATURES = [
    "passes_p90", "pass_acc", "progressive_passes_p90", "key_passes_p90", "crosses_p90",
    "shots_p90", "dribbles_p90",
    "duels_p90", "interceptions_p90",
    "avg_x", "avg_width", "final_third_share",
]
LOG_FEATURES = [f for f in ROLE_FEATURES if f.endswith("_p90")]


def eligible_players(pf: pd.DataFrame, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """Players with enough minutes and a known role."""
    return pf[(pf.wy_minutes >= min_minutes) & pf.wy_role.notna()].reset_index(drop=True)


def position_matrix(df: pd.DataFrame, features: list[str] = ROLE_FEATURES) -> np.ndarray:
    """log1p on per-90 counts, then z-score each feature within this position's players."""
    X = df[features].astype(float).copy()
    for f in features:
        if f in LOG_FEATURES:
            X[f] = np.log1p(X[f])
    X = X.fillna(X.median())
    return ((X - X.mean()) / X.std(ddof=0)).to_numpy()


def _kmeans(k: int, seed: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=seed)


def bootstrap_stability(X: np.ndarray, k: int, n_boot: int = 50, seed: int = SEED) -> tuple[float, float]:
    """Mean and std of the adjusted Rand index between the full-data clustering and clusterings
    refitted on bootstrap resamples (each applied to all players). 1 = identical groupings."""
    rng = np.random.default_rng(seed)
    base = _kmeans(k, seed).fit_predict(X)
    scores = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        model = _kmeans(k, seed + b + 1).fit(X[idx])
        scores.append(adjusted_rand_score(base, model.predict(X)))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_k(X: np.ndarray, ks=range(2, 7), n_boot: int = 50, seed: int = SEED) -> pd.DataFrame:
    """Silhouette score and bootstrap stability for each candidate k."""
    rows = []
    for k in ks:
        labels = _kmeans(k, seed).fit_predict(X)
        ari_mean, ari_std = bootstrap_stability(X, k, n_boot, seed)
        sizes = np.bincount(labels)
        rows.append((k, silhouette_score(X, labels), ari_mean, ari_std, sizes.min(), sizes.max()))
    return pd.DataFrame(rows, columns=["k", "silhouette", "stability_ari", "stability_sd",
                                       "smallest_cluster", "largest_cluster"])


def fit_roles(X: np.ndarray, k: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Cluster labels and each player's distance to their cluster centre (in z-score units)."""
    model = _kmeans(k, seed).fit(X)
    dist = np.linalg.norm(X - model.cluster_centers_[model.labels_], axis=1)
    return model.labels_, dist


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray, X: np.ndarray,
                     features: list[str] = ROLE_FEATURES) -> pd.DataFrame:
    """Per cluster: size and mean z-score of each feature (positive = above the position's average)."""
    z = pd.DataFrame(X, columns=features)
    z["cluster"] = labels
    prof = z.groupby("cluster")[features].mean()
    prof.insert(0, "n", z.groupby("cluster").size())
    return prof


def names_from_anchors(wy_ids: pd.Series, labels: np.ndarray, anchors: dict[str, int]) -> dict[int, str]:
    """Map cluster number -> role name via anchor players ({role name: wy_id}).

    k-means numbers its clusters arbitrarily (the numbering can change with library versions or
    data), so roles are named by "the cluster containing this player" instead of by number. Raises
    if an anchor is missing, two anchors share a cluster, or a cluster has no anchor.
    """
    label_of = dict(zip(wy_ids, labels))
    mapping = {}
    for name, wid in anchors.items():
        if wid not in label_of:
            raise ValueError(f"anchor {wid} for '{name}' is not among the clustered players")
        c = int(label_of[wid])
        if c in mapping:
            raise ValueError(f"anchors for '{mapping[c]}' and '{name}' fall in the same cluster {c}")
        mapping[c] = name
    unnamed = set(map(int, np.unique(labels))) - set(mapping)
    if unnamed:
        raise ValueError(f"clusters {sorted(unnamed)} have no anchor player")
    return mapping


def assign_roles(pf: pd.DataFrame, k_by_position: dict[str, int],
                 anchors: dict[str, dict[str, int]] | None = None, seed: int = SEED) -> pd.DataFrame:
    """One row per eligible player: wy_id, position, role cluster, role name and distance to centre.

    Goalkeepers get a single role. ``anchors`` maps position -> {role name: anchor wy_id} (see
    ``names_from_anchors``); positions without anchors get '<position> <cluster>' labels.
    """
    anchors = anchors or {}
    df = eligible_players(pf)
    out = []
    for pos, g in df.groupby("wy_role"):
        g = g.reset_index(drop=True)
        if pos in OUTFIELD:
            labels, dist = fit_roles(position_matrix(g), k_by_position[pos], seed)
            names = names_from_anchors(g.wy_id, labels, anchors[pos]) if pos in anchors else {}
        else:
            labels, dist = np.zeros(len(g), dtype=int), np.full(len(g), np.nan)
            names = {0: "Goalkeeper"}
        role = [names.get(int(c), f"{pos} {c}") for c in labels]
        out.append(pd.DataFrame({"wy_id": g.wy_id, "season": g.season, "position": pos,
                                 "role_cluster": labels, "role": role, "distance_to_centre": dist}))
    return pd.concat(out, ignore_index=True)
