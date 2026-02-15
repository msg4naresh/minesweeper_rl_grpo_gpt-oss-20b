"""
Minesweeper GRPO Training v2
============================
Fixes over v1:
  1. Competition-exact prompt (train = eval)
  2. max_completion_length = 128 (match competition's max_new_tokens)
  3. 8 generations (reduce zero-std batches)
  4. Continuous reward scoring with mine probability
  5. Mixed board sizes in dataset
  6. Tuned GRPO hyperparameters
  7. Save merged model for competition
"""

import os
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Unsloth MUST be imported before trl/transformers for monkey-patches to apply
from unsloth import FastLanguageModel

import json
import math
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from datasets import Dataset
from transformers import TrainerCallback, TextStreamer
from trl import GRPOConfig, GRPOTrainer

# ============================================================
# 1. Model Loading
# ============================================================

max_seq_length = 1024
lora_rank = 16

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-BF16",
    load_in_4bit=False,
    load_in_16bit=True,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
)
print(f"Model device: {model.device}")

# ============================================================
# 2. LoRA Setup
# ============================================================

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ============================================================
# 3. Minesweeper Game Engine 
# ============================================================

@dataclass
class MinesweeperGame:
    rows: int
    cols: int
    num_mines: int
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _revealed: Set[Tuple[int, int]] = field(init=False, repr=False, default_factory=set)
    _flagged: Set[Tuple[int, int]] = field(init=False, repr=False, default_factory=set)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        if self.num_mines >= self.rows * self.cols:
            raise ValueError("Too many mines for board size")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self._place_mines()
        self._calculate_numbers()

    def _place_mines(self):
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        mine_positions = self._rng.sample(positions, self.num_mines)
        for r, c in mine_positions:
            self._board[r][c] = -1

    def _calculate_numbers(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self._board[r][c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self._board[nr][nc] == -1:
                                count += 1
                self._board[r][c] = count

    def _reveal_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed or (row, col) in self._flagged:
            return False
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in self._revealed:
                continue
            self._revealed.add((r, c))
            if self._board[r][c] == -1:
                self._state = "failed"
                return True
            if self._board[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols
                                and (nr, nc) not in self._revealed
                                and (nr, nc) not in self._flagged):
                            stack.append((nr, nc))
        return True

    def _flag_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed:
            return False
        if (row, col) in self._flagged:
            self._flagged.remove((row, col))
        else:
            self._flagged.add((row, col))
        return True

    def do_action(self, action: dict) -> str:
        if self._state != "ongoing":
            return "game_over"
        if not isinstance(action, dict):
            self._state = "failed"
            return "invalid_format"
        action_type = action.get("type")
        row = action.get("row")
        col = action.get("col")
        if action_type not in ["reveal", "flag"] or row is None or col is None:
            self._state = "failed"
            return "invalid_format"
        try:
            row, col = int(row), int(col)
        except (ValueError, TypeError):
            self._state = "failed"
            return "invalid_format"
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self._state = "failed"
            return "out_of_bounds"
        if action_type == "reveal":
            if (row, col) in self._revealed:
                self._state = "failed"
                return "already_revealed"
            if (row, col) in self._flagged:
                self._state = "failed"
                return "flagged_cell"
            valid = self._reveal_cell(row, col)
        else:
            if (row, col) in self._revealed:
                self._state = "failed"
                return "invalid_flag"
            valid = self._flag_cell(row, col)
        if not valid:
            self._state = "failed"
            return "invalid_format"
        self._check_win()
        if self._state == "failed":
            return "mine"
        if self._state == "success":
            return "win"
        return "ok"

    def _check_win(self):
        total_cells = self.rows * self.cols
        safe_cells = total_cells - self.num_mines
        if len(self._revealed) == safe_cells:
            self._state = "success"

    def get_visible_board(self) -> List[List[str]]:
        visible = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self._flagged:
                    row.append('F')
                elif (r, c) in self._revealed:
                    val = self._board[r][c]
                    row.append('*' if val == -1 else str(val))
                else:
                    row.append('.')
            visible.append(row)
        return visible

    def state(self) -> str:
        return self._state

# ============================================================
# 4. Competition-Exact Prompt (Fix 1)
# ============================================================

def build_competition_prompt(game: MinesweeperGame):
    """Build prompt exactly matching competition agent's build_prompt().

    Returns (user_prompt, system_prompt) for chat template.
    """
    state = {
        "board": game.get_visible_board(),
        "rows": game.rows,
        "cols": game.cols,
        "mines": game.num_mines,
        "flags_placed": len(game._flagged),
        "cells_revealed": len(game._revealed),
    }

    sys_prompt = "You output JSON actions for Minesweeper. No text, only JSON."

    prompt = f"""You are playing Minesweeper. Analyze the game state and output your next move.

You must output ONLY a valid JSON object. No explanation, no analysis, no text.

Just output section after assistantfinal and not anything before it in your output.

Start your response immediately with {{ and end with }}.

Do NOT output cell which is already revealed or flagged in the current state.

Game state:
{json.dumps(state, indent=2)}

Legend:
- "." = unrevealed cell
- "F" = flagged cell (suspected mine)
- "0"-"8" = number of adjacent mines
- "*" = revealed mine (game over)

Output your next action as JSON:
{{"type": "reveal", "row": <row_index>, "col": <col_index>}}
or
{{"type": "flag", "row": <row_index>, "col": <col_index>}}

Your action:"""

    return prompt, sys_prompt

# ============================================================
# 5. Action Parser (matches competition's first-valid-JSON)
# ============================================================

def parse_llm_action(response: str) -> Optional[dict]:
    """Extract first valid JSON action from LLM response.

    Competition parser takes FIRST valid JSON, so we match that.
    """
    import re
    for match in re.finditer(r'\{[^{}]*\}', response):
        try:
            action = json.loads(match.group())
            if ("type" in action and "row" in action and "col" in action
                    and action["type"] in ["reveal", "flag"]):
                action["row"] = int(action["row"])
                action["col"] = int(action["col"])
                return action
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None

# ============================================================
# 6. Helper Functions for Reward Scoring
# ============================================================

def has_revealed_neighbor(game: MinesweeperGame, row: int, col: int) -> bool:
    """Check if target cell is adjacent to any revealed cell.

    Used to distinguish informed moves from blind guesses in gameplay_reward.
    Cells near revealed info get a higher base score (+2.0) than blind guesses
    (+0.5), teaching the model to work outward from known information rather
    than randomly clicking unexplored areas.
    """
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < game.rows and 0 <= nc < game.cols:
                if (nr, nc) in game._revealed:
                    return True
    return False


def is_logically_safe(game: MinesweeperGame, row: int, col: int) -> bool:
    """Check if (row, col) is provably safe via constraint propagation.

    A cell is logically safe if any adjacent revealed number cell has ALL
    its mines accounted for by flags. Then all other unrevealed neighbors
    of that number cell (including (row, col)) must be safe.

    Ground-truth validation: only counts a flag as "accounting for a mine"
    if the flagged cell actually contains a mine (game._board[fr][fc] == -1).
    This uses information available during reward computation to prevent
    giving deduction credit based on incorrect flags.
    """
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if not (0 <= nr < game.rows and 0 <= nc < game.cols):
                continue
            if (nr, nc) not in game._revealed:
                continue
            cell_val = game._board[nr][nc]
            if cell_val <= 0:
                continue
            # Count flags that are on actual mines (ground-truth check)
            valid_flag_count = 0
            hidden_neighbors = []
            for dr2 in [-1, 0, 1]:
                for dc2 in [-1, 0, 1]:
                    if dr2 == 0 and dc2 == 0:
                        continue
                    nr2, nc2 = nr + dr2, nc + dc2
                    if 0 <= nr2 < game.rows and 0 <= nc2 < game.cols:
                        if (nr2, nc2) in game._flagged and game._board[nr2][nc2] == -1:
                            valid_flag_count += 1
                        elif (nr2, nc2) not in game._revealed and (nr2, nc2) not in game._flagged:
                            hidden_neighbors.append((nr2, nc2))
            if valid_flag_count == cell_val and (row, col) in hidden_neighbors:
                return True
    return False


def estimate_mine_probability(game: MinesweeperGame, row: int, col: int) -> float:
    """Estimate probability that (row, col) is a mine.

    Provides a continuous risk signal for the safety_bonus in gameplay_reward:
        safety_bonus = 4.0 * (1.0 - mine_probability)
    This gives a smooth reward from 0 (certain mine) to +4.0 (certainly safe),
    so GRPO can differentiate completions that pick low-risk vs high-risk cells
    even when both survive.

    Algorithm:
      For each adjacent revealed number cell, compute:
          remaining_mines / unrevealed_neighbors
      Return max across all constraints (worst-case / most informative).
      If no adjacent info, fall back to global prior: remaining_mines / total_hidden.

    Ground-truth validation: only counts flags on actual mines when computing
    remaining mine counts, preventing incorrect flags from skewing probability
    estimates downward (same rationale as is_logically_safe).
    """
    max_prob = -1.0

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if not (0 <= nr < game.rows and 0 <= nc < game.cols):
                continue
            if (nr, nc) not in game._revealed:
                continue
            cell_val = game._board[nr][nc]
            if cell_val <= 0:
                continue
            # Count flags and unrevealed around this number cell
            flag_count = 0
            unrevealed_count = 0
            for dr2 in [-1, 0, 1]:
                for dc2 in [-1, 0, 1]:
                    if dr2 == 0 and dc2 == 0:
                        continue
                    nr2, nc2 = nr + dr2, nc + dc2
                    if 0 <= nr2 < game.rows and 0 <= nc2 < game.cols:
                        if (nr2, nc2) in game._flagged and game._board[nr2][nc2] == -1:
                            flag_count += 1
                        elif (nr2, nc2) not in game._revealed:
                            unrevealed_count += 1
            remaining = cell_val - flag_count
            if unrevealed_count > 0 and remaining >= 0:
                prob = remaining / unrevealed_count
                max_prob = max(max_prob, prob)

    if max_prob >= 0:
        return min(max_prob, 1.0)

    # Global prior: remaining mines / total hidden cells
    total_hidden = game.rows * game.cols - len(game._revealed) - len(game._flagged)
    remaining_mines = game.num_mines - len(game._flagged)
    if total_hidden > 0:
        return remaining_mines / total_hidden
    return 0.5

# ============================================================
# 7. Reward Functions (Fix 4: Continuous Scoring)
# ============================================================

def format_reward(completions, **kwargs):
    """Reward for output format quality.

    Steepness is tuned so format scores don't dominate gameplay signals.
    The gap between tiers (+2 vs -3 = 5 points) is enough for GRPO to
    learn bare-JSON output, but small relative to gameplay rewards
    (e.g., safe deduced reveal = +12, mine hit = -20, win = +50).

    4 distinct levels:
      +2.0  Bare JSON (starts with '{' after stripping) — ideal competition output
      +1.0  Valid JSON found but with surrounding text — functional but wasteful tokens
      -3.0  No valid JSON parsed — model output is unusable
      -5.0  Empty/whitespace response — model produced nothing
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]

        if not response or not response.strip():
            scores.append(-5.0)
            continue

        action = parse_llm_action(response)
        if action is None:
            scores.append(-3.0)
            continue

        # Check if response is "bare" JSON (starts with { after stripping)
        stripped = response.strip()
        if stripped.startswith("{"):
            scores.append(2.0)
        else:
            scores.append(1.0)

    return scores


def gameplay_reward(completions, **kwargs):
    """Continuous gameplay scoring — the main training signal for Minesweeper.

    Design principles:
      - Tiered gating: lower tiers (format, legality) gate higher tiers (safety,
        deduction, progress). A mine hit gets ONLY -20, not mixed with bonuses.
      - Continuous scoring: GRPO needs to rank completions within a group. Flat
        scoring (all safe reveals = same score) gives zero gradient when most
        completions survive. Continuous bonuses (safety, progress) differentiate
        "safe but risky" from "safe and smart".
      - Ground-truth validation: is_logically_safe and estimate_mine_probability
        only count flags on actual mines, preventing deduction/safety credit from
        incorrect flags.
      - Assert guard: if is_logically_safe returns True for a mine cell, an
        assertion fires. With ground-truth validation this should be impossible
        — the assert catches future regressions.

    State reconstruction:
      Each completion's game state is rebuilt from seed + move_history (stored
      as JSON string in the dataset). If context is missing (e.g., idx out of
      range), returns neutral 0.0 to avoid noisy gradients.

    Reveal branch:
      Invalid JSON             → -8
      Out of bounds            → -12
      Already revealed         → -10
      Reveal flagged cell      → -6
      Mine hit                 → -20 (assert: not is_logically_safe)
      Safe reveal (blind)      → +0.5 + safety_bonus + progress_bonus
      Safe reveal (near info)  → +2.0 + safety_bonus + deduction_bonus + progress_bonus
      Win                      → +50 bonus on top of above

    Flag branch:
      Flag revealed cell       → -8
      Flag already flagged     → -10
      Correct flag (is mine)   → +10
      Wrong flag (not mine)    → -8
      Over-flagging            → -8 additive (flags > num_mines)
      Note: _flag_cell is called BEFORE scoring so len(_flagged) reflects
      the new flag when checking over-flag threshold.

    Bonus components (reveal only):
      safety_bonus   = 4.0 * (1.0 - mine_probability)  [continuous 0 to +4]
      deduction_bonus = +6.0 if is_logically_safe (provably safe via constraints)
      progress_bonus = min(log2(new_reveals + 1) * 2.0, 5.0)
        Capped at 5.0 so flood-fill doesn't overshadow deduction (+6) or win (+50).
    """
    scores = []

    seeds = kwargs.get("seed", [])
    move_histories = kwargs.get("move_history", [])
    board_configs = kwargs.get("board_config", [])

    for idx, completion in enumerate(completions):
        response = completion[0]["content"]
        action = parse_llm_action(response)

        # No valid JSON
        if action is None:
            scores.append(-8.0)
            continue

        if idx >= len(seeds) or idx >= len(move_histories):
            scores.append(0.0)
            continue

        seed = int(seeds[idx])
        move_history_raw = move_histories[idx]
        if isinstance(move_history_raw, str):
            move_history = json.loads(move_history_raw)
        else:
            move_history = move_history_raw

        # Parse board config
        if idx < len(board_configs):
            parts = str(board_configs[idx]).split(",")
            rows, cols, num_mines = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            rows, cols, num_mines = 6, 6, 5

        # Reconstruct game state
        game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
        for prev_action in move_history:
            game.do_action(prev_action)

        try:
            row, col = int(action["row"]), int(action["col"])
        except (ValueError, TypeError):
            scores.append(-8.0)
            continue

        action_type = action["type"]

        # Out of bounds
        if not (0 <= row < game.rows and 0 <= col < game.cols):
            scores.append(-12.0)
            continue

        # ── REVEAL ──
        if action_type == "reveal":
            if (row, col) in game._revealed:
                scores.append(-10.0)
                continue
            if (row, col) in game._flagged:
                scores.append(-6.0)
                continue

            # Mine hit
            if game._board[row][col] == -1:
                assert not is_logically_safe(game, row, col), \
                    f"is_logically_safe returned True for mine at ({row},{col})"
                scores.append(-20.0)
                continue

            # Safe reveal — compute continuous score
            # Base score
            near_info = has_revealed_neighbor(game, row, col)
            base = 2.0 if near_info else 0.5

            # Deduction bonus
            deduction_bonus = 6.0 if is_logically_safe(game, row, col) else 0.0

            # Safety bonus: continuous based on mine probability
            mine_prob = estimate_mine_probability(game, row, col)
            safety_bonus = 4.0 * (1.0 - mine_prob)

            # Progress bonus: execute action and count new reveals
            revealed_before = len(game._revealed)
            game.do_action({"type": "reveal", "row": row, "col": col})
            revealed_after = len(game._revealed)
            new_reveals = revealed_after - revealed_before
            progress_bonus = min(math.log2(new_reveals + 1) * 2.0, 5.0) if new_reveals > 0 else 0.0

            # Win bonus
            win_bonus = 50.0 if game.state() == "success" else 0.0

            score = base + deduction_bonus + safety_bonus + progress_bonus + win_bonus
            scores.append(score)

        # ── FLAG ──
        elif action_type == "flag":
            if (row, col) in game._revealed:
                scores.append(-8.0)
                continue
            if (row, col) in game._flagged:
                scores.append(-10.0)
                continue

            # Apply the flag to game state so over-flag check sees it
            game._flag_cell(row, col)

            if game._board[row][col] == -1:
                flag_score = 10.0
            else:
                flag_score = -8.0

            # Over-flagging penalty (flag already applied, so _flagged includes it)
            if len(game._flagged) > game.num_mines:
                flag_score -= 8.0

            scores.append(flag_score)
        else:
            scores.append(-8.0)

    return scores

# ============================================================
# 8. Dataset Generation (Fix 5: Mixed Board Sizes)
# ============================================================

BOARD_CONFIGS = [
    # (rows, cols, mines, weight)
    (6, 6, 5, 0.40),   # training board
    (5, 5, 3, 0.20),   # easy
    (7, 7, 7, 0.20),   # medium
    (8, 8, 10, 0.10),  # hard
    (9, 9, 10, 0.10),  # beginner standard
]


def _find_logical_flags(game):
    """Find cells that can be logically deduced as mines from current state.

    Used during dataset generation to ensure training states include flags,
    so flag-related rewards are actually exercised during training. Without
    this, the model would rarely see flagged board states and flag rewards
    would remain under-trained.

    For each revealed numbered cell, if its number equals the count of
    adjacent hidden (non-flagged) cells plus already-flagged cells,
    then all hidden neighbors must be mines -> flag them.

    Returns a list of (row, col) tuples to flag.
    """
    to_flag = []
    for (r, c) in list(game._revealed):
        val = game._board[r][c]
        if val <= 0:
            continue
        hidden = []
        flag_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < game.rows and 0 <= nc < game.cols):
                    continue
                if (nr, nc) in game._flagged:
                    flag_count += 1
                elif (nr, nc) not in game._revealed:
                    hidden.append((nr, nc))
        if val == flag_count + len(hidden) and len(hidden) > 0:
            to_flag.extend(hidden)
    seen = set()
    unique = []
    for pos in to_flag:
        if pos not in seen:
            seen.add(pos)
            unique.append(pos)
    return unique


def generate_game_states(num_samples=1000, rng_seed=42):
    """Generate diverse game states across multiple board sizes.

    Uses competition-exact prompt format.
    Stores board_config as "rows,cols,mines" string.
    After each reveal, logical flag placements are computed and applied,
    so training states include flags in their move history.
    """
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    # Pre-compute how many samples per config
    config_weights = [w for _, _, _, w in BOARD_CONFIGS]
    config_counts = [int(num_samples * w) for w in config_weights]
    # Distribute remainder to first config
    config_counts[0] += num_samples - sum(config_counts)

    dataset_items = []

    for (rows, cols, mines, _), count in zip(BOARD_CONFIGS, config_counts):
        generated = 0
        attempts = 0
        max_attempts = count * 5

        while generated < count and attempts < max_attempts:
            attempts += 1
            seed = int(np.random.randint(100000))
            game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)

            # 0-5 random moves
            num_moves = int(np.random.randint(0, 11))
            move_history = []

            for _ in range(num_moves):
                if game.state() != "ongoing":
                    break
                board = game.get_visible_board()
                unrevealed = [
                    (r, c)
                    for r in range(rows)
                    for c in range(cols)
                    if board[r][c] == '.' and (r, c) not in game._flagged
                ]
                if not unrevealed:
                    break
                r, c = random.choice(unrevealed)
                action = {"type": "reveal", "row": r, "col": c}
                game.do_action(action)
                move_history.append(action)

                # After each reveal, apply any logical flag deductions
                if game.state() == "ongoing":
                    flags_to_place = _find_logical_flags(game)
                    for fr, fc in flags_to_place:
                        if (fr, fc) not in game._flagged:
                            flag_action = {"type": "flag", "row": fr, "col": fc}
                            game.do_action(flag_action)
                            move_history.append(flag_action)

            if game.state() != "ongoing":
                continue

            user_prompt, sys_prompt = build_competition_prompt(game)
            dataset_items.append({
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "seed": seed,
                "move_history": json.dumps(move_history),
                "board_config": f"{rows},{cols},{mines}",
            })
            generated += 1

    # Shuffle so board sizes are interleaved
    random.shuffle(dataset_items)
    return Dataset.from_list(dataset_items)


print("Generating training dataset...")
dataset = generate_game_states(num_samples=1000, rng_seed=42)
print(f"Created {len(dataset)} training examples")

# Count by config
from collections import Counter
config_counts = Counter(dataset["board_config"])
for cfg, cnt in sorted(config_counts.items()):
    print(f"  {cfg}: {cnt} ({cnt/len(dataset)*100:.0f}%)")

fresh_count = sum(1 for mh in dataset["move_history"] if mh == "[]")
print(f"  Fresh games: {fresh_count} ({fresh_count/len(dataset)*100:.1f}%)")
flag_count = sum(1 for mh in dataset["move_history"] if '"flag"' in mh)
print(f"  States with flags: {flag_count} ({flag_count/len(dataset)*100:.1f}%)")

# ============================================================
# 9. Eval Callback (uses competition prompt)
# ============================================================

class MinesweeperEvalCallback(TrainerCallback):
    """Play games every N steps during training."""

    def __init__(self, eval_every_steps=50, num_games=5):
        self.eval_every_steps = eval_every_steps
        self.num_games = num_games

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if state.global_step % self.eval_every_steps != 0:
            return

        tok = processing_class
        if tok is None or model is None:
            return

        was_training = model.training
        model.eval()

        wins = 0
        for i in range(self.num_games):
            game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=10000 + i)
            moves = 0
            while game.state() == "ongoing" and moves < 50:
                user_prompt, sys_prompt = build_competition_prompt(game)
                text = tok.apply_chat_template(
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                output = model.generate(
                    **tok(text, return_tensors="pt").to(model.device),
                    temperature=0.3,
                    max_new_tokens=128,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
                response = tok.decode(
                    output[0][len(tok(text, return_tensors="pt")["input_ids"][0]):],
                    skip_special_tokens=True,
                )
                action = parse_llm_action(response)
                if action is None:
                    break
                result = game.do_action(action)
                if result not in ("ok", "win"):
                    break
                moves += 1
            if game.state() == "success":
                wins += 1

        win_rate = wins / self.num_games
        print(f"\n[Eval @ step {state.global_step}] Win rate: {wins}/{self.num_games} ({win_rate*100:.0f}%)\n")

        if was_training:
            model.train()

# ============================================================
# 10. GRPO Training Config (Fix 2, 3, 6)
# ============================================================

max_prompt_length = 896   # Competition prompt is longer with indent=2
max_completion_length = 128  # Fix 2: match competition max_new_tokens

training_args = GRPOConfig(
    temperature=0.8,                     # Fix 6: less noise
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    bf16=True,                           # Match model dtype (bfloat16)
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,       # 4 x 4 = 16 effective completions per update
    num_generations=4,                   # Reduced from 8 to fit in GPU memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=500,                       # ~3hr budget
    save_steps=100,
    report_to="none",
    output_dir="minesweeper_v2_outputs",
)

print(f"\nTraining config:")
print(f"  max_prompt_length: {max_prompt_length}")
print(f"  max_completion_length: {max_completion_length}")
print(f"  num_generations: {training_args.num_generations}")
print(f"  max_steps: {training_args.max_steps}")
print(f"  temperature: {training_args.temperature}")

# ============================================================
# 11. Train
# ============================================================

eval_callback = MinesweeperEvalCallback(eval_every_steps=50, num_games=5)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward,
        gameplay_reward,
    ],
    args=training_args,
    train_dataset=dataset,
    callbacks=[eval_callback],
)

print("\nStarting GRPO training v2...")
trainer.train()

# ============================================================
# 12. Post-Training Evaluation
# ============================================================

print("\n" + "=" * 60)
print("Post-Training Evaluation (20 games)")
print("=" * 60)

model.eval()
wins = 0
results = []

for i in range(20):
    game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=20000 + i)
    moves = 0
    end_reason = "max_moves"

    while game.state() == "ongoing" and moves < 50:
        user_prompt, sys_prompt = build_competition_prompt(game)
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output = model.generate(
            **tokenizer(text, return_tensors="pt").to(model.device),
            temperature=0.3,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        response = tokenizer.decode(
            output[0][len(tokenizer(text, return_tensors="pt")["input_ids"][0]):],
            skip_special_tokens=True,
        )
        action = parse_llm_action(response)
        if action is None:
            end_reason = "parse_error"
            break
        result = game.do_action(action)
        if result == "mine":
            end_reason = "mine_hit"
        elif result == "win":
            end_reason = "win"
        elif result not in ("ok", "win"):
            end_reason = f"illegal:{result}"
            break
        moves += 1

    if game.state() == "success":
        wins += 1
    tag = "WIN" if game.state() == "success" else "LOSS"
    print(f"  Game {i+1:2d}: {tag} ({end_reason}, {moves} moves)")
    results.append(end_reason)

print(f"\nWin rate: {wins}/20 ({wins/20*100:.0f}%)")
print(f"End reasons: {Counter(results)}")

# ============================================================
# 13. Save Model (Fix 7: Merged)
# ============================================================

print("\nSaving LoRA adapters...")
model.save_pretrained("my_minesweeper_model_v2")
tokenizer.save_pretrained("my_minesweeper_model_v2")

print("Saving merged model for competition...")
model.save_pretrained_merged(
    "/workspace/your_finetuned_model",
    tokenizer,
    save_method="merged_16bit",
)
print("Merged model saved to: /workspace/your_finetuned_model")
print("\nDone! Update /workspace/agents/minesweeper_model.py line 15 to point to the merged model.")
