"""
Gemini LOO / Sequential Simulation
====================================
Two complementary evaluations for the paper:

1. SEQUENTIAL (experiments 11-20):
   Simulate what Gemini would have predicted for the new 10 participants
   if it had been active during their sessions. Experiment k uses
   aggregated preferences from experiments 1..k-1 (the same sequential
   structure used live for experiments 1-10).

2. LOO — Leave-One-Out (all 20 users):
   For each held-out user i, Gemini is given aggregated preferences
   from the other 19 users and predicts each step.
   This gives the fair head-to-head accuracy against the deterministic
   LOO baseline.

Both evaluations use:
  - Neutral user profile (since no profile was collected during static sessions)
  - Within-session ground-truth history (steps completed before the current one)
  - N_REPEATS repetitions to average out LLM non-determinism (T=0.7)

Estimated runtime: ~45-55 minutes total (960 API calls at 2s rate limit).

Outputs:
  analysis/gemini_sequential_11_20.csv  — per-step predictions, experiments 11-20
  analysis/gemini_loo_all20.csv         — per-step predictions, all 20 LOO users
"""

import sys
import os
import json
import time

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Project root and path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = r'C:\Users\sara.masiero\PycharmProjects\adaptive-GUIs-experiments'
sys.path.insert(0, PROJECT_ROOT)

# Load API key from .env
_env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith('GOOGLE_API_KEY'):
                os.environ['GOOGLE_API_KEY'] = _line.split('=', 1)[1]
                print(f"API key loaded from .env (prefix: {os.environ['GOOGLE_API_KEY'][:8]})")
                break

from services.sentient_gemini_api import adapt_step

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_REPEATS = 3  # repetitions per user/step to average non-determinism

EXPERIMENTS = [
    'sequential',
    'loo'
]

GT_PATH = os.path.join(PROJECT_ROOT, 'analysis', 'all_20_experiments.csv')
STEPS_PATH = os.path.join(PROJECT_ROOT, 'settings', 'steps_sources.json')
INTERACTIONS_PATH = os.path.join(PROJECT_ROOT, 'settings', 'enabled_interactions.json')

CHECKPOINT_SEQ = os.path.join(PROJECT_ROOT, 'analysis', 'gemini_sequential_11_20_checkpoint.csv')
CHECKPOINT_LOO = os.path.join(PROJECT_ROOT, 'analysis', 'gemini_loo_all20_checkpoint.csv')
SHUFFLE_SEQ_PATH = os.path.join(PROJECT_ROOT, 'analysis', 'gemini_sequential_shuffle_orders.json')

RUN_ID = time.strftime('%Y%m%d_%H%M%S')

FORMAT_COLS = ['short_text_viewed', 'long_text_viewed', 'single_pieces_viewed',
               'assembly_viewed', 'video_viewed']
FORMAT_KEYS = ['short_text', 'long_text', 'single_pieces', 'assembly', 'video']

NEUTRAL_PROFILE = {
    'language': 'English',
    'training_objective': 'Learn the assembly process step by step',
    'screen_setup': 'Standard tablet at comfortable viewing distance',
    'prior_experience': 'Some experience with assembly tasks',
    'visual_comfort': {'high_contrast': False, 'large_text': False, 'color_blind_assist': False},
    'other_requests': 'None',
}

# ---------------------------------------------------------------------------
# Load ground truth and step definitions
# ---------------------------------------------------------------------------
gt = pd.read_csv(GT_PATH, sep=';')
gt['experiment_id'] = gt['experiment_id'].astype(int)
gt['step_id'] = gt['step_id'].astype(int)
for col in FORMAT_COLS:
    gt[col] = gt[col].astype(int)

with open(STEPS_PATH) as f:
    steps_by_id = {s['id']: s for s in json.load(f)['assembly_process']}

with open(INTERACTIONS_PATH) as f:
    enabled_interactions = json.load(f)

all_exp_ids = [int(i) for i in sorted(gt['experiment_id'].unique())]
N = len(all_exp_ids)
print(f"Loaded {len(gt)} rows, {N} experiments: {all_exp_ids}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def compute_aggregated_prefs(training_exp_ids, step_id):
    """Per-format proportion from training users for a given step."""
    train_step = gt[(gt['experiment_id'].isin(training_exp_ids)) & (gt['step_id'] == step_id)]
    if len(train_step) == 0:
        return {k: 0.0 for k in FORMAT_KEYS}
    return {
        'short_text': float(train_step['short_text_viewed'].mean()),
        'long_text': float(train_step['long_text_viewed'].mean()),
        'single_pieces': float(train_step['single_pieces_viewed'].mean()),
        'assembly': float(train_step['assembly_viewed'].mean()),
        'video': float(train_step['video_viewed'].mean()),
    }


def format_history(user_id, completed_step_ids):
    """Within-session history string for steps already completed."""
    if not completed_step_ids:
        return "CURRENT USER: No previous interactions\n"
    user_data = gt[(gt['experiment_id'] == user_id) & (gt['step_id'].isin(completed_step_ids))]
    text = "CURRENT USER PREVIOUS INTERACTIONS:\n"
    for _, row in user_data.sort_values('step_id').iterrows():
        viewed = [k for k, c in zip(FORMAT_KEYS, FORMAT_COLS) if row[c]]
        text += f"  Step {int(row['step_id'])}: viewed [{', '.join(viewed) if viewed else 'nothing'}]\n"
    return text


def build_step_payload(step_id):
    s = steps_by_id[step_id]
    return {'step_id': step_id, 'name': s['name'],
            'category': s['category'], 'adaptive_fields': s['adaptive_fields']}


def call_gemini(user_id, step_id, training_exp_ids, completed_steps):
    """One Gemini call; returns dict of binary predictions."""
    agg = compute_aggregated_prefs(training_exp_ids, step_id)
    hist = format_history(user_id, completed_steps)
    payload = build_step_payload(step_id)

    result = adapt_step(
        user_profile=NEUTRAL_PROFILE,
        style_profile_token='neutral_profile',
        step_payload=payload,
        user_history_formatted=hist,
        aggregated_preferences=agg,
        enabled_interactions=enabled_interactions,
    )
    vis = result.get('initial_visibility', {})
    return {k: int(bool(vis.get(k, False))) for k in FORMAT_KEYS}


def step_accuracy(pred, gt_row):
    """Mean binary accuracy over all 5 format flags."""
    pairs = zip([pred[k] for k in FORMAT_KEYS],
                [int(gt_row[c]) for c in FORMAT_COLS])
    return sum(p == g for p, g in pairs) / 5


def load_checkpoint(path):
    """Return (records_list, done_set) from a checkpoint CSV, or ([], set()) if none."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        records = df.to_dict('records')
        done = set(zip(df['experiment_id'].astype(int),
                       df['step_id'].astype(int),
                       df['repeat'].astype(int)))
        print(f"  Checkpoint found: {len(records)} records, resuming.")
        return records, done
    return [], set()


def save_checkpoint(records, path):
    pd.DataFrame(records).to_csv(path, index=False)


def load_shuffle_orders(path):
    """Return {repeat_int: {'base': [...], 'seq': [...]}} from JSON, or {} if none."""
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return {}


def save_shuffle_orders(orders, path):
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in orders.items()}, f)


def step_accuracy_available(pred, gt_row):
    """Binary accuracy restricted to formats available for this step type."""
    step_id = int(gt_row['step_id'])
    for cfg in enabled_interactions['steps']:
        if cfg['step_id'] == step_id:
            avail = cfg['buttons']
            break
    else:
        avail = {k: True for k in FORMAT_KEYS}

    pairs = [(pred[k], int(gt_row[c]))
             for k, c in zip(FORMAT_KEYS, FORMAT_COLS)
             if avail.get(k, False)]
    return sum(p == g for p, g in pairs) / len(pairs) if pairs else 0.0


# ---------------------------------------------------------------------------
# EVALUATION 1 — Sequential simulation for experiments 11-20
# ---------------------------------------------------------------------------
if 'sequential' in EXPERIMENTS:
    print("\n" + "=" * 60)
    print("EVALUATION 1: Sequential simulation — experiments 11-20")
    print("=" * 60)

    seq_records, seq_done = load_checkpoint(CHECKPOINT_SEQ)
    seq_shuffle_orders = load_shuffle_orders(SHUFFLE_SEQ_PATH)
    total_seq = N * 10 * N_REPEATS
    done_seq = 0
    t0 = time.time()

    for repeat in range(1, N_REPEATS + 1):
        if repeat in seq_shuffle_orders:
            base_exps = seq_shuffle_orders[repeat]['base']
            seq_exps = seq_shuffle_orders[repeat]['seq']
            print(f"  Repeat {repeat} | restored shuffle | base: {base_exps} | sequential: {seq_exps}")
        else:
            shuffled_all = list(all_exp_ids)
            np.random.shuffle(shuffled_all)
            base_exps = shuffled_all[:10]
            seq_exps = shuffled_all[10:]
            seq_shuffle_orders[repeat] = {'base': list(base_exps), 'seq': list(seq_exps)}
            save_shuffle_orders(seq_shuffle_orders, SHUFFLE_SEQ_PATH)
            print(f"  Repeat {repeat} | base: {base_exps} | sequential: {seq_exps}")

        for i, exp_id in enumerate(seq_exps):
            training_exps = base_exps + seq_exps[:i]
            user_steps = gt[gt['experiment_id'] == exp_id].sort_values('step_id')

            completed = []
            for _, row in user_steps.iterrows():
                step_id = int(row['step_id'])

                if (exp_id, step_id, repeat) in seq_done:
                    completed.append(step_id)
                    done_seq += 1
                    continue

                try:
                    pred = call_gemini(exp_id, step_id, training_exps, completed)
                    acc5 = step_accuracy(pred, row)
                    acc_avail = step_accuracy_available(pred, row)
                    seq_records.append({
                        'experiment_id': exp_id,
                        'prior_users': len(training_exps),
                        'step_id': step_id,
                        'repeat': repeat,
                        'accuracy_5': acc5,
                        'accuracy_available': acc_avail,
                        **{f'pred_{k}': pred[k] for k in FORMAT_KEYS},
                        **{f'gt_{k}': int(row[c]) for k, c in zip(FORMAT_KEYS, FORMAT_COLS)},
                    })
                    save_checkpoint(seq_records, CHECKPOINT_SEQ)
                except Exception as e:
                    print(f"  [ERROR] exp={exp_id} step={step_id} repeat={repeat}: {e}")
                completed.append(step_id)
                done_seq += 1

            elapsed = time.time() - t0
            eta = (elapsed / done_seq) * (total_seq - done_seq) if done_seq else 0
            print(f"  Exp {exp_id} | prior={len(training_exps)} | repeat={repeat} "
                  f"| elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m")

    seq_df = pd.DataFrame(seq_records)
    seq_out = os.path.join(PROJECT_ROOT, 'analysis', f'gemini_sequential_{RUN_ID}.csv')
    seq_df.to_csv(seq_out, index=False)
    if os.path.exists(CHECKPOINT_SEQ):
        os.remove(CHECKPOINT_SEQ)
    if os.path.exists(SHUFFLE_SEQ_PATH):
        os.remove(SHUFFLE_SEQ_PATH)
    print(f"\nSequential results saved to: {seq_out}")

    # Summary
    seq_per_exp = seq_df.groupby('experiment_id')['accuracy_5'].mean()
    print("\n--- Sequential: mean accuracy per experiment (avg over repeats) ---")
    for eid, acc in seq_per_exp.items():
        print(f"  Exp {eid:2d} | prior users: {eid - 1:2d} | accuracy: {acc:.3f}")
    print(f"  Overall mean: {seq_per_exp.mean():.3f}")

# ---------------------------------------------------------------------------
# EVALUATION 2 — LOO evaluation (all 20 users)
# ---------------------------------------------------------------------------
if 'loo' in EXPERIMENTS:
    print("\n" + "=" * 60)
    print("EVALUATION 2: LOO evaluation — all 20 users")
    print("=" * 60)

    loo_records, loo_done = load_checkpoint(CHECKPOINT_LOO)
    total_loo = N * 16 * N_REPEATS
    done_loo = 0
    t0 = time.time()

    for repeat in range(1, N_REPEATS + 1):
        shuffled_loo_ids = list(all_exp_ids)
        np.random.shuffle(shuffled_loo_ids)
        print(f"  Repeat {repeat} experiment order: {shuffled_loo_ids}")

        for exp_id in shuffled_loo_ids:
            training_exps = [e for e in all_exp_ids if e != exp_id]
            user_steps = gt[gt['experiment_id'] == exp_id].sort_values('step_id')

            completed = []
            for _, row in user_steps.iterrows():
                step_id = int(row['step_id'])

                if (exp_id, step_id, repeat) in loo_done:
                    completed.append(step_id)
                    done_loo += 1
                    continue

                try:
                    pred = call_gemini(exp_id, step_id, training_exps, completed)
                    acc5 = step_accuracy(pred, row)
                    acc_avail = step_accuracy_available(pred, row)
                    loo_records.append({
                        'experiment_id': exp_id,
                        'step_id': step_id,
                        'repeat': repeat,
                        'accuracy_5': acc5,
                        'accuracy_available': acc_avail,
                        **{f'pred_{k}': pred[k] for k in FORMAT_KEYS},
                        **{f'gt_{k}': int(row[c]) for k, c in zip(FORMAT_KEYS, FORMAT_COLS)},
                    })
                    save_checkpoint(loo_records, CHECKPOINT_LOO)
                except Exception as e:
                    print(f"  [ERROR] exp={exp_id} step={step_id} repeat={repeat}: {e}")
                completed.append(step_id)
                done_loo += 1

            elapsed = time.time() - t0
            eta = (elapsed / done_loo) * (total_loo - done_loo) if done_loo else 0
            print(f"  User {exp_id:2d} | repeat={repeat} "
                  f"| elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m")

    loo_df = pd.DataFrame(loo_records)
    loo_out = os.path.join(PROJECT_ROOT, 'analysis', f'gemini_loo_{RUN_ID}.csv')
    loo_df.to_csv(loo_out, index=False)
    if os.path.exists(CHECKPOINT_LOO):
        os.remove(CHECKPOINT_LOO)
    print(f"\nLOO results saved to: {loo_out}")

    # Summary
    loo_per_user = loo_df.groupby('experiment_id')['accuracy_5'].mean()
    print("\n--- LOO: mean accuracy per user (avg over repeats) ---")
    for eid, acc in loo_per_user.items():
        print(f"  User {eid:2d}: {acc:.3f}")
    print(f"\n  LOO Gemini mean:  {loo_per_user.mean():.3f}")
    print(f"  LOO Gemini SD:    {loo_per_user.std():.3f}")

    # Format-level breakdown (LOO)
    print("\n--- LOO per-format accuracy ---")
    for k, c in zip(FORMAT_KEYS, FORMAT_COLS):
        pred_col = f'pred_{k}'
        gt_col = f'gt_{k}'
        fmt_acc = (loo_df[pred_col] == loo_df[gt_col]).mean()
        print(f"  {k:20s}: {fmt_acc:.3f}")

# Naive baseline for reference
naive_acc = (gt[FORMAT_COLS] == 0).mean(axis=1).mean()
print(f"\n  Naive baseline (predict all 0): {naive_acc:.3f}")
