# Project Plan

**Every team member and every AI agent follows this plan.** Read it before
starting work. Anyone can pick up any task: say which one you're taking so
work isn't duplicated. If a task is not in this plan, raise it with the team
before doing it. When something changes (a task finishes, a decision is made),
update the **Status** and **Decisions** sections in the same commit.

Last updated: 2026-11-04

## 1. Goal

For a candidate player and a destination team, predict the player's
**playing-time share in the 12 months after the transfer** and use that score
to rank candidates. The model must beat the baselines (constant mean, linear
regression) on mean squared error (MSE) to justify being a neural network.
Player-role and team-style clusters make recommendations interpretable. Scores
are scouting support, not a replacement for human judgement.

## 2. Milestones

The professor's milestone 1 feedback (9/10) said the full plan is "four
projects" (ID matching, player-role clustering, team-style clustering, neural
fit model) and asked us to get the join working first, with **one clustering**
in milestone 1. Everything else stays in the project and comes in later
milestones.

### Milestone 1: join, baselines, one clustering (Wyscout 2017/18, small model)

| Task | Status |
|---|---|
| Reproducible, pinned data download (`src/data/download_data.py`) | ✅ done |
| Player matching Wyscout ↔ Transfermarkt | ✅ 98.5% matched (2,523 / 2,561) |
| 12-month playing-time label (league games only, D2) + transfer funnel | ✅ N = 276 eligible transfers |
| Matching, label and funnel moved into `src/` (`python -m src.pipelines.build_transfers`), reproducing notebook 01 exactly | ✅ |
| Fixed train/test split (`data/processed/split.csv`) | ✅ 220 train / 56 test (regenerated for D2) |
| Constant-mean and linear-regression baselines | ✅ cross-validation MSE 0.0815 / 0.0865; test MSE 0.0973 / 0.1012 |
| Player features table for clustering (§5, step 3): per-90 stats + location features, 2,561 players | ✅ `data/processed/player_features.csv` |
| **Player-role clustering** (§5, D14): 12 roles, 1,947 players | ✅ `data/processed/player_roles.csv`, `notebooks/side_work/03_player_roles.ipynb` |
| EA FC light add-on (D12): download, 2017/18 matching, role-cluster check, optional baseline features | ⬜ to do, **not a dependency** |
| Milestone 1 hand-in notebook, in the brief's five parts + roles (D15) | ✅ `notebooks/04_milestone1.ipynb` |
| Milestone 1 written report | ⬜ to do |

N = 276 is under the 300 threshold in our proposal, so milestone 1 follows the
proposal's mitigation: **small models, few features**. After milestone 1 the
multi-season dataset removes that limit (D7, D11).

**EA FC in milestone 1 is a light add-on (D12):**
- It is used as an outside check on the role clusters (§5, step 5) and,
  optionally, for a handful of extra baseline features compared by
  cross-validation.
- Every other milestone 1 task must be finished and reportable **without**
  it: if the download or the matching runs into trouble, drop the add-on
  rather than delay the milestone.

### Later milestones (in this order)

1. **Multi-season dataset** (§6): transfers after each feature season, with
   player features from Transfermarkt, Understat and EA FC (D8, D11).
   This is what makes a non-small neural model possible. Estimated N:

   | Player source | Feature seasons | N |
   |---|---|---|
   | Understat | 2014/15 – 2020/21 | **≈ 1,950**, about 7× milestone 1 |
   | EA FC, big-5 selling clubs | 2015/16 – 2022/23 | **≈ 2,800** |
   | EA FC, any selling league (D13, open) | 2015/16 – 2022/23 | up to **≈ 5,800** |
2. **Team-style features and clustering:** team style vectors and clusters.
   Rich version from Wyscout for 2017/18. A multi-season version from:
   - Transfermarkt formations
   - Understat shot/xG profiles
   - Elo
   - possibly EA FC team ratings and tactics settings (§6, item 6)
3. **Team weakness features:** where each squad was weak or thin, expressed in
   player roles and positions (e.g. few minutes from ball-winning midfielders).
   With EA FC (D11), roles can be described for every position in every
   season, not only for Wyscout 2017/18.
4. **Neural fit model** (§8).
5. **Evaluation:**
   - a single run on the time-based test season (D10)
   - case studies ranking candidates for 2–3 teams
   - a comparison with the rich-feature Wyscout 2017/18 model
   - limitations and the poster

Rough timing from the proposal: features weeks 6–7, neural model weeks 8–10,
tuning weeks 11–12, back-test and case studies week 13, poster week 14.
**Gate:** if the network doesn't beat the constant mean in validation by the
end of week 9, shrink it and report that honestly.

### Already built, outside the milestones

- **Club context** (`notebooks/side_work/02_club_context_features.ipynb`,
  `src/features/club_context.py`): ClubElo team strength and destination
  manager tenure. No standalone gain on the 276 transfers. Kept for fit
  features and error analysis (D4), and reusable for every season.

**Result to keep in mind:** neither the linear regression nor the club context
beats the constant mean on the 276 transfers. Player-only and club-only
information does not predict playing time there, so the project's value has to
come from **player–team fit** and from **more data**.

## 3. Decisions

### Open

- **D3. Manager at summer transfers** (low priority): 21% of transfers have a
  different manager before and after the move. Keep the strict pre-transfer
  definition, or use the manager in the first game after the transfer?
- **D9. COVID-affected labels:** transfers after the 2018/19 and 2019/20
  seasons have labels in the stoppage, or in seasons with 5 substitutes,
  crowded fixtures and no fans. Keep them with a `covid_label` flag
  (≈ 1,950 transfers) or drop them (≈ 1,300)? Recommendation: keep them with
  the flag, and report results with and without them. With EA FC seasons
  (D11, big-5 selling clubs) the same choice is ≈ 2,800 vs ≈ 2,200.
- **D10. Multi-season split:** recommendation is a **time-based split**: test
  on transfers after the 2020/21 season (≈ 390, with normal 2021/22 labels)
  and train/validate on the earlier seasons. This replaces the random split
  for later milestones; `split.csv` stays the milestone 1 split. With EA FC
  adopted alongside Understat (D11), there are two candidate test seasons:
  - **After 2020/21** (≈ 390): both Understat and EA FC features exist, so
    the test rows look like the training rows.
  - **After 2022/23** (≈ 540, normal 2023/24 labels): more recent and
    larger, but EA FC only, because Understat stops at 2020/21. The model
    would have to cope with missing Understat features, or be compared as
    an EA-only model.
- **D13. Selling clubs outside the big 5?** EA FC covers more leagues than
  Understat. With non-big-5 selling clubs, N is ≈ 5,800 over 2015/16 –
  2022/23, vs ≈ 2,800 big-5 only.
  - The 5,800 is an **upper bound**. It counts every league Transfermarkt
    tracks, but EA's licensed leagues change by edition (e.g. Russia removed
    in 2022; Ukraine and Serbia mostly missing; generated names for some
    Brazilian players).
  - Those transfers would have EA features but **no Understat features**,
    because Understat only covers the big 5.
  - The estimate uses the D7 method, re-run 2026-11-04 with the old
    all-competitions label (D2 lowers it slightly). Per feature season, big
    5 / other leagues: 2015/16 202 / 204, 2016/17 264 / 262, 2017/18
    282 / 281, 2018/19 343 / 336, 2019/20 297 / 322, 2020/21 389 / 474,
    2021/22 499 / 531, 2022/23 539 / 568.

  Until the team decides, §6 keeps big-5 selling clubs.

### Made

- **D1. Milestone 1 data** (2026-10-21): keep the current window (transfers
  15 May 2018 – 15 Feb 2019, ≥ 450 Wyscout minutes) with a small model.
  N = 278, then 276 after the league-only label (D2). More data comes from
  D7, not from relaxing filters.
- **D4. Elo** (2026-10-21): keep the code and columns. Use them inside fit
  features (e.g. player level vs destination level) or for error analysis.
  Don't add them to the model on their own: no gain in cross-validation on
  the 278, re-checked on the 276 after D2.
- **D5. Data pinning** (2026-10-20/21):
  - Transfermarkt snapshot of 2026-09-05, checked by MD5
  - ClubElo archive commit `2e8632b`, checked by MD5
  - Understat repo commit `87960c2`, pinned by commit
- **D6. Milestone 1 clustering = player roles** (2026-10-21), not team styles.
  - ~1,900 players with ≥ 450 minutes vs only 98 teams, so the clusters are
    more stable.
  - Every one of the 276 transfers has a Wyscout player, so coverage is 100%.
  - Per-90 stats already exist.
  - Team weakness features will be expressed in roles, so roles are needed
    first anyway.
- **D7. More data after milestone 1** (2026-10-21): build a multi-season
  dataset (2014/15 – 2020/21 feature seasons) from Transfermarkt + Understat,
  instead of extra Wyscout tournaments, summer 2019 only, or StatsBomb.
  - Why: the bottleneck is Wyscout's single season, not missing players.
  - Estimated N per feature season: 172, 202, 264, 282, 343, 297, 389, so
    ≈ 1,950 in total. The method was checked against milestone 1 (282
    estimated vs 278 actual).
  - Trade-off: players are described less richly (Understat is mostly about
    attacking; no passes, duels or touch locations), so defenders are weakly
    described. Wyscout 2017/18 stays the rich, interpretable layer.
- **D2. Label counts league games only** (2026-11-04):
  - **Definition:** y = the player's league minutes for the new club ÷ (90 ×
    the club's league games) in the 365 days after the transfer, capped at 1.
    League = Transfermarkt `competition_type == "domestic_league"`.
  - **Why:** a player isn't penalised for cup or European games he wasn't
    registered for.
  - **Scope:** `prev_share_tm` (the year before the move) uses the same rule.
    Notebook 01's config flag `LABEL_LEAGUE_ONLY` switches it.
  - **Effect on milestone 1:**
    - N 278 → 276. The two dropped transfers were moves to Dinamo Zagreb,
      whose league Transfermarkt doesn't track. Under the old rule they got
      y = 0 from European games alone: fake zeros, now removed.
    - The other labels barely changed (correlation 0.991). Goalkeepers
      changed most (+0.03), as backup keepers often play the cups.
  - **Split and baselines:** `split.csv` was regenerated (220 / 56; only 28
    of the 56 test transfers are the same as before) and the baselines
    re-run:

    | Model | Cross-validation MSE | Test MSE |
    |---|---|---|
    | Constant mean | 0.0815 ± 0.0087 | 0.0973 |
    | Linear regression | 0.0865 | 0.1012 |

    Test MSE rose from 0.0833 mostly because the 56-transfer test set was
    reshuffled. It's a reminder of how noisy that test set is.
  - **Multi-season estimates:** D7 and D11 were computed with the old rule;
    expect slightly lower N.
- **D12. EA FC in milestone 1, used lightly** (2026-11-04):
  - Add the EA FC dataset in milestone 1 for the 2017/18 players, matched to
    Transfermarkt by date of birth + fuzzy name (§6, item 6). Use the latest
    update dated on or before 15 May 2018 (§4, Leakage).
  - Two uses only:
    1. an **outside check on the Wyscout role clusters** (§5, step 5)
    2. **optional** extra baseline features: at most the six summary
       ratings (pace, shooting, passing, dribbling, defending, physic),
       within the ≤ 20-input rule. Kept only if they beat the baseline in
       cross-validation.
  - The clustering itself stays Wyscout-based, and no milestone 1
    deliverable depends on EA. If the add-on causes trouble, drop it.
- **D8. Understat is acceptable** (2026-11-04, approved by the professor):
  use the published copy (`douglasbc/scraping-understat-dataset`, commit
  `87960c2`) for the course project. Understat's `robots.txt` asks not to be
  crawled and the copy has no licence, so the conditions stay as rules:
  - never scrape Understat ourselves
  - never commit or redistribute the files
  - credit both Understat and the copy in any write-up
- **D11. EA FC is a player-feature source for the multi-season dataset**
  (2026-11-04, approved by the professor). It **complements** Understat,
  since D8 keeps Understat. The professor accepted all three questions:
  1. the change of player-feature source
  2. EA ratings as features, even though they are expert opinions rather
     than observed events (the label stays observed Transfermarkt playing
     time)
  3. the licence: Kaggle lists CC0 (checked 2026-11-03, version 4)

  Why EA FC: Understat describes players mostly through attacking (the D7
  trade-off, weak for defenders). EA FC rates passing, defending, physical
  and mental attributes for every position, across feature seasons
  2015/16 – 2022/23. §6 item 6 describes how it's used; selling clubs
  outside the big 5 are still open (D13).
- **D15. Milestone 1 hand-in** (2026-11-05), following the professor's worked
  example:
  - **Notebook:** `notebooks/04_milestone1.ipynb`, standalone. It downloads
    data and builds missing tables with `src/`; notebooks 01–03 stay as the
    working behind it.
  - **Kept simple (2026-11-05):** the baselines and models are written inline
    with plain pandas and scikit-learn (`src/models/` was removed). Each
    section has a "notes for the write-up" cell for the team to replace with
    their own text.
  - **Split on the player:** one transfer each, so 0 players on both sides.
    Destination-club overlap (32 of 111 clubs; 80% of test transfers) is
    reported as a caveat; the shared split is unchanged.
  - **Metric:** MSE, with the reason stated. An "overestimated by ≥ 0.25"
    count reflects that overestimating playing time is the costly mistake.
  - **Baselines in one table:**

    | Baseline | Cross-validation MSE |
    |---|---|
    | Always predict the average (best) | 0.0815 |
    | Average for his role | 0.0849 |
    | Linear regression | 0.0865 |
    | "Plays as much as last year" (`prev_share_tm`) | 0.146 |
  - **Leakage demo:** using the market value one year after the transfer
    "improves" test MSE to 0.070, showing why every feature must be dated
    before the transfer.
  - **Optimisation checks:**
    - 20-row check: about 98% fit (MSE 0.002). Only 19 independent
      parameters (verified rank), so an exact fit isn't possible.
    - SGD training curve: validation MSE is lowest after ~7 passes and never
      below the average (overfitting).
- **D14. Player roles: k and names** (2026-11-04):
  - **k per position:** defenders 4, midfielders 4, forwards 3. Each is the
    largest k that stays stable when the data is resampled (adjusted Rand
    index 0.89 / 0.79 / 0.88). Silhouette always prefers k = 2, which only
    separates central from wide players.
  - **Forwards at k = 4** would split strikers into "target striker" vs
    "finisher". That's football-meaningful, but unstable: ARI 0.62 when
    resampled, and 0.91 even with just a different random start on the same
    data, vs 1.00 for k = 3. 21 creative forwards switch sides, and two
    clusters have only ~80 players. If that distinction matters later, use
    a Gaussian mixture (soft probabilities) or a continuous feature, rather
    than a fourth hard cluster.
  - **Names and anchor players:** each role is named by the cluster
    containing its anchor (k-means numbers clusters arbitrarily).

    | Position | Roles (anchor) |
    |---|---|
    | DF | Attacking full-back (Bertrand), Defensive full-back (Zabaleta), Ball-playing centre-back (Bonucci), Defensive centre-back (Maguire) |
    | MD | Attacking midfielder (De Bruyne), Deep-lying midfielder (Busquets), Ball-winning midfielder (Saúl), Wide midfielder (Perišić) |
    | FW | Creative forward (Messi), Central striker (Kane), Wide support forward (Gerard Moreno) |
    | GK | Goalkeeper (one role, not clustered) |
  - **"Deep-lying midfielder", not "playmaker":** Kanté and Casemiro share
    that role with Busquets and Jorginho, because per-90 counts partly
    reflect team possession (see §9).

## 4. Rules for every agent and team member

### Evaluation protocol

- Milestone 1: `data/processed/split.csv` is the fixed split. Later
  milestones: the time-based split (D10).
- **All choices (features, architecture, hyperparameters, number of clusters
  when used as features) are made on training/validation data only.**
  Milestone 1 uses `RepeatedKFold(5 splits, 10 repeats, random_state=42)`, as
  in notebook 01. Later milestones use validation on the seasons before the
  test season.
- **Test transfers are used once, in the final evaluation.** Each extra look
  makes the test score less trustworthy.
- Metric: MSE, with RMSE also reported. The bar is the best baseline on the
  same split. On the milestone 1 split that's the constant mean
  (cross-validation MSE 0.0815 ± 0.0087; test MSE 0.0973). Quote the spread of the
  cross-validation or validation scores with every comparison: differences
  smaller than that are noise.

### Leakage

- Features use only information available **before** the transfer date.
  Write each feature's time window next to its code.
- Season features describe the season that ended **before** the transfer
  window. Transfers made during a feature season are excluded, because their
  stats would mix matches from before and after the move.
- Post-transfer columns (e.g. `dest_mgr_change_in_window`) are **diagnostics
  only** and never model inputs.
- **EA FC ratings:**
  - For each feature season, use each player's **latest EA update dated on
    or before the start of that season's transfer window (15 May)**,
    consistent with "season features describe the season that ended before
    the transfer window". In milestone 1 that's the latest update on or
    before 15 May 2018.
  - Never use an update from after the transfer: September updates (new
    game editions) often already show the player at his new club.

### Model size

- **Milestone 1 (N = 276):** small models, at most about 20 model inputs,
  regularisation, several random seeds.
- **Later milestones (N ≈ 1,950 with Understat; ≈ 2,800 – 5,800 with EA FC,
  see D11):** larger networks are allowed. Their size is justified by
  validation results, not set in advance.

### Code and data

- Raw data is never modified. Download only through
  `src/data/download_data.py`, where every source is pinned.
- Reusable code goes in `src/`; notebooks are numbered (`01_`, `02_`, …) and
  call `src/`.
- Outputs go to `data/processed/` (model-ready) or `data/interim/` (matching
  tables and other intermediates). Only `split.csv` is committed. Never
  commit raw data (D8).
- **EA FC data:**
  - Downloaded only through `src/data/download_data.py`, pinned like the
    sources in D5 (Kaggle dataset version, currently 4, + MD5).
  - Needs a Kaggle API token. Document a manual download from the Kaggle
    page as the fallback.
  - Never commit or redistribute it.
  - `male_players.csv` has millions of rows: read it with `usecols` and
    explicit dtypes, and cache a filtered copy in `data/interim/`.
- Keep player, club, competition and season identifiers in every table.
- Set random seeds for clustering, training and sampling.

### Teaching

- Follow `AGENTS.md`: explain the approach and why before implementing, so the
  team can review and learn from it.

## 5. Player-role clustering (milestone 1)

**Goal:** group players into interpretable roles within each broad position,
for example "ball-winning midfielder" vs "deep playmaker".

1. **Players:** everyone with ≥ 450 Wyscout minutes in 2017/18 (the same
   threshold as the transfer dataset), not only the 276 who transferred, so
   the roles describe the whole league population.
2. **Positions:** cluster defenders, midfielders and forwards separately
   (Wyscout `role`). Goalkeepers are one role; their event stats don't
   separate playing styles.
3. **Features:** the per-90 stats from notebook 01, plus a few **location
   features** from the events, because counts alone mostly separate busy from
   quiet players:
   - average pitch position (x, y)
   - share of actions in the final third
   - crosses, dribbles and progressive passes per 90

   Standardise within each position. Keep it to about 10–15 features.

   **Built (2026-11-04):** `python -m src.pipelines.build_player_features`
   writes `data/processed/player_features.csv`. The code is in
   `src/features/player_features.py`, whose docstring has the exact
   definitions.
   - One row per player with 2017/18 league minutes (2,561; 1,947 with
     ≥ 450 minutes). Minutes are kept, so each user applies their own
     threshold.
   - The original per-90 columns match notebook 01 exactly for all 276
     transfers.
   - **Use `avg_width` (distance from the centre line), not `avg_y`, for
     roles:** a left-back and a right-back play the same role.
   - **Dribbles** are take-ons (Wyscout tags 503/504). **Progressive passes**
     are accurate passes ending ≥ 25% closer to goal, measured in metres.
   - **Data quirk handled:** goal kicks and goalkeeper save events have
     placeholder positions, (0, 0) or (100, 100), and are left out of the
     location averages.
   - **Known limitation:** goalkeepers' long launches count as progressive
     passes. Goalkeepers aren't clustered, so this doesn't affect roles.
   - **Sanity checks passed:** average x rises GK 12 → DF 41 → MD 53 → FW 64,
     and the leaders look right:
     - dribbles: Neymar, Januzaj, Messi
     - crosses: Candreva, Marcelo, Zappacosta
     - progressive passes: Fàbregas, Banega, Boateng
4. **Method:** k-means as the main method, testing k = 2–6 per position and
   choosing k by silhouette score **and** whether the clusters can be named.
   Optionally use a Gaussian mixture as well, to get soft role probabilities
   the model can use later.
5. **Checks:**
   - Name each cluster from its average profile.
   - Look at well-known players in each cluster (e.g. Busquets, Kanté,
     Salah).
   - Check stability: rerun with different seeds and on bootstrap samples, and
     compare with the adjusted Rand index.
   - **Outside check with EA FC (optional, D12):** compare each cluster's
     average EA attributes (e.g. defending vs passing vs attacking) and EA
     `player_positions` against its name. A "ball-winning midfielder"
     cluster should have high EA defending ratings. EA is used only to
     check the clusters, never as a clustering input, so the clusters don't
     depend on it.
6. **Output:** `data/processed/player_roles.csv`, keyed by `wy_id`, with
   broad position, role cluster, role probabilities (if a Gaussian mixture is
   used) and distance to the cluster centre.
7. **Link to the label (descriptive only):** mean playing-time share by role
   for the 276 transfers. Whether roles help the model is decided later, by
   cross-validation.

**Built (2026-11-04):** `src/features/player_roles.py` and
`notebooks/side_work/03_player_roles.ipynb`; k and names in D14.
- **Features (12):** `passes_p90`, `pass_acc`, `progressive_passes_p90`,
  `key_passes_p90`, `crosses_p90`, `shots_p90`, `dribbles_p90`,
  `duels_p90`, `interceptions_p90`, `avg_x`, `avg_width`,
  `final_third_share`. Left out: goals and assists (outcome-driven), duels
  won (duplicates duels), fouls (discipline) and `avg_y` (side, not role).
- **Preprocessing:** `log1p` on per-90 counts, then z-scored within each
  position.
- **Method:** k-means, seed 42, 20 starts.
- **Stability:** 50 bootstrap refits per k, compared with the adjusted Rand
  index.
- **Famous-player check passed:**
  - Marcelo and Kimmich are attacking full-backs
  - Ramos and Hummels are ball-playing centre-backs
  - Modrić and De Bruyne are attacking midfielders
  - Messi, Neymar and Griezmann are creative forwards
  - Kane and Lewandowski are central strikers
- **Roles of the 276 transfers (descriptive, noisy):** all have a role.
  - Highest mean y: central strikers 0.56, goalkeepers 0.56, deep-lying
    midfielders 0.55.
  - Lowest: attacking midfielders 0.37, wide support forwards 0.40.
  - With 17–36 transfers per role and a spread of about 0.3, these
    differences aren't evidence yet.
- **Not done (optional):** a Gaussian mixture for soft role probabilities.

Leakage: 2017/18 data only, and every transfer is after May 2018, so the roles
are pre-transfer.

## 6. Multi-season dataset (first milestone after milestone 1)

**Build order** (decided 2026-11-04; keeps D8 and D11, only orders the work):

- **First: Transfermarkt + EA FC.** EA matching is easier (it has dates of
  birth), and EA describes every position.
- **Next: Understat team profiles,** when building team style: team xG
  for/against and shot locations and situations, per club and season. They
  are our main *observed* multi-season source for team style and strength.
- **Last: Understat player features, and only if they improve validation
  scores.** They add observed quality measures (xG, xA, `xGChain`,
  `xGBuildup`), mainly for attackers, and serve as a check on EA's ratings.

1. **Transfers and labels:**
   - Generalise notebook 01's funnel and `time_share` label to every feature
     season the player sources cover: 2014/15 – 2020/21 for Understat,
     2015/16 – 2022/23 for EA FC (D11).
   - Window per season: 15 May after the season – 15 Feb the following year.
   - Same filters as milestone 1: club-to-club moves, no loan returns, first
     move per player, destination club with ≥ 10 recorded league games, and the
     league-only label (D2).
   - Add `feature_season` and `covid_label` columns.
   - Move the logic into `src/` so both datasets share it. **Done for
     milestone 1 (2026-11-05):** `src/data/player_matching.py`,
     `src/features/labels.py` and `src/pipelines/build_transfers.py`, which
     reproduce notebook 01's link table and split byte-for-byte and its
     baselines to 17 decimals. The multi-season version extends these
     instead of copying them.
2. **Understat ↔ Transfermarkt player matching:**
   - Understat has no dates of birth, so match by name, constrained to the
     same club and season (Transfermarkt appearances give each player's club
     per season).
   - Report the match rate per league and season, and eyeball a sample, as
     in milestone 1.
   - Output: `data/interim/understat_tm_link.csv`.
3. **Player-season features**, all from the feature season:
   - **Transfermarkt:** minutes share, starts, goals and assists per 90,
     cards, age, market value, position.
   - **Understat:** xG, npxG, xA, shots, key passes, `xGChain` and
     `xGBuildup` per 90; position string.
   - **EA FC (D11):** see item 6.
4. **Team-season features:**
   - Elo (reuse `club_context.py`)
   - formation usage from Transfermarkt games
   - team xG for/against and shot profile from Understat shots
   - squad depth per position from Transfermarkt minutes
   - possibly EA FC team ratings and tactics settings (item 6)
5. **Coverage:**
   - Understat only covers the big 5, so the **player** features need a
     big-5 feature season; destinations can be anywhere Transfermarkt
     covers. EA FC covers more selling leagues, but §6 stays big-5 until D13
     is decided.
   - **The two player sources cover different seasons** (D8, D11):

     | Feature seasons | Player features available |
     |---|---|
     | 2014/15 | Understat only (no EA update before the window) |
     | 2015/16 – 2020/21 | Understat **and** EA FC |
     | 2021/22 – 2022/23 | EA FC only (the Understat copy stops at 2020/21) |

     Keep every row and add a "source available" flag per source, so the
     model learns from missing features instead of dropping seasons. The
     choice of test season interacts with this (D10).
   - Report the funnel per season and per source, as in milestone 1.
6. **EA FC player features** (D11, adopted for the multi-season dataset;
   the 2017/18 part is also milestone 1's light add-on, D12):
   - **EA ↔ Transfermarkt player matching:**
     - EA has dates of birth, so reuse milestone 1's date-of-birth +
       fuzzy-name matcher.
     - Validate precision against the Reep register
       (`github.com/withqwerty/reep`, CC0), which has `key_transfermarkt`
       and `key_sofifa` columns. Precision is measured on the Reep subset
       with both IDs (~19k people, counted 2026-11-03), which likely
       over-represents well-known players.
     - Report the match rate per league and season.
     - Output: `data/interim/eafc_tm_link.csv`.
   - **Features:**
     - **Column check** (2026-11-04, from the header of a public copy of
       `male_players.csv`):
       - 109 columns in total.
       - **29 detailed outfield attributes** in six groups: `attacking_*` 5,
         `skill_*` 5, `movement_*` 5, `power_*` 5, `mentality_*` 6,
         `defending_*` 3.
       - **6 `goalkeeping_*` attributes**, including `goalkeeping_speed`,
         which is empty for outfield players.
       - `skill_moves` shares the `skill_` prefix but is a separate 1–5 star
         rating (like `weak_foot`), not one of the 29.
       - The update date column there is `update_as_of`; the Kaggle
         description calls it `fifa_update_date`.
       - The copy is 96,156,653 bytes vs 96,336,675 for Kaggle version 4, so
         it is probably an earlier version. **Re-check the columns when the
         download script fetches version 4.**
     - Keep: the 29 detailed outfield attributes, plus the 6
       `goalkeeping_*` attributes for goalkeepers. Also keep
       `overall`, `potential`,
       `height_cm`, `weight_kg`, `weak_foot`, and `work_rate` split into
       attacking and defensive.
     - Drop:
       - `skill_moves` (decided 2026-11-04)
       - the per-position ratings (`ls` … `gk`, stored as text like "85+2")
       - image URLs, `real_face` and `body_type`
       - `value_eur`, `wage_eur` and `release_clause_eur` (market value
         stays Transfermarkt)
     - Standardise attributes within each `fifa_version`, because rating
       scales drift between editions.
   - **Coverage:**
     - EA updates start on 2015-09-10, so check coverage per feature season
       rather than assuming it.
     - 2014/15 has no update before its window (15 May 2015).
     - 2023/24's only update (2023-09-22) is from the start of that season,
       so it doesn't describe the season. The usable feature seasons are
       therefore 2015/16 – 2022/23.
     - EA's licensed leagues change by edition. Report the funnel per
       season with and without EA.
   - **EA team table** (`male_teams.csv`: team ratings, tactics settings): a
     possible input to the team-season features (item 4). The tactics
     columns vary by game version and need a completeness check before use.

## 7. Data hand-offs between tasks

Each task produces **one table with a fixed key**, so a model dataset is a
simple join. All tables live in `data/processed/` unless noted.

| File | Key | Content | Time window | Milestone |
|---|---|---|---|---|
| `transfers_joined.csv` | `player_id, wy_id, transfer_date, to_club_id` | label `y`, baseline features (Wyscout 2017/18), built by `src/pipelines/build_transfers.py` | label: 365 days after the transfer; features: before it | 1 ✅ |
| `split.csv` | same as above | `train` / `test` | fixed | 1 ✅ |
| `player_features.csv` | `wy_id` | identity, role, minutes, per-90 event stats, pass accuracy, location features | 2017/18 season | 1 ✅ |
| `player_roles.csv` | `wy_id` | position, role cluster, role name, distance to the cluster centre | 2017/18 season | 1 ✅ |
| `transfers_multiseason.csv` | `player_id, transfer_date, to_club_id` | label `y`, `feature_season`, `covid_label`, split | as above, per season | later |
| `data/interim/understat_tm_link.csv` | `understat_id, season ↔ player_id` | match with score | — | later |
| `player_season_features.csv` | `player_id, feature_season` | Transfermarkt + Understat player features | feature season | later |
| `data/interim/eafc_tm_link.csv` | `eafc_player_id ↔ player_id` | EA FC ↔ Transfermarkt player match, with score | — | 1 (2017/18 players, D12), later all seasons |
| `player_season_features.csv` (EA columns) | `player_id, feature_season` | EA FC attributes (§6, item 6) | latest EA update on or before the window start (15 May) | 1 (2017/18 only, D12), later all seasons |
| `data/interim/team_link.csv` | `wy_team_id ↔ club_id` | Wyscout team ↔ Transfermarkt club, with match score | — | later |
| `team_style.csv` | `club_id, season` | team style vector and style cluster | that season | later |
| `team_weakness.csv` | `club_id, season, position` | where the squad was weak or thin | that season | later |
| `transfers_club_context.csv` | transfer key | Elo, manager tenure (+ diagnostics) | before the transfer | ✅ built (2018 window) |

**Coverage risk for Wyscout team features:** the rich team style only exists
for clubs in the 2017/18 Wyscout leagues. Before building it, count how many
of the 276 destination clubs have Wyscout data, then either drop the
uncovered transfers or fill in the league average with a "missing" flag.
Record the choice under Decisions.

The model dataset builders (`src/pipelines/`) join these tables and are the
only input to the models.

## 8. Neural fit model (later milestone, multi-season data)

Start simple and add only what validation supports:

1. **MLP:** concatenate baseline, player and destination team features
   (player features include Understat and EA FC attributes, D8 and D11).
   Learned embeddings for categorical inputs (position, league), and a
   sigmoid output because y is between 0 and 1. Trained with MSE. Width and
   depth are chosen by validation.
2. **Fit model:** a player encoder and a team encoder, each producing an
   embedding. The prediction combines a baseline term with the player–team
   interaction (dot product or a small MLP on both embeddings). This is the
   part that models "fit".
3. **Ranking:** score every candidate player for a destination team and sort.
   Explain the top candidates with their role and style clusters and with
   feature attributions. With EA FC, candidates can come from any league EA
   covers, not only those with event data.

Each version is accepted only if it beats the previous one in validation by
more than the spread of the validation scores. The milestone 1 version
(Wyscout, N = 276) stays small, per §4.

## 9. Known limitations (keep updated for the poster)

- Wyscout covers one season (2017/18) and the top five leagues only. The
  multi-season dataset has broader coverage but less detail on each player,
  especially defenders. EA FC attributes (D11) fill that gap, but as expert
  ratings rather than observed events.
- Milestone 1 has 276 examples: small-sample noise, so differences in MSE
  need cross-validation standard deviations alongside them.
- Player roles come from per-90 counts that partly reflect **team
  possession**, not only the player (D14): e.g. Kanté shares a role with
  Busquets. Midfielder roles are the least stable (ARI 0.79). Possession-
  adjusted stats are a possible improvement.
- Injuries are not modelled, and manager changes after the transfer (56% of
  destination clubs within the label window) can't be predicted.
- Transfermarkt coverage is limited to tracked leagues; transfers to uncovered
  clubs are dropped rather than given a fake label.
- The label counts league games only (D2): cup and European minutes are
  ignored, and clubs in leagues Transfermarkt doesn't track get no label.
  COVID affects the labels for two seasons (D9).
- Understat data is a third-party scrape with no licence, approved by the
  professor for this course project (D8).
- EA FC attributes are expert ratings, not observed events. They are shaped
  by reputation and less accurate for smaller leagues. Rating scales also
  drift between game editions, and EA's licensed leagues change by edition
  (D11).
