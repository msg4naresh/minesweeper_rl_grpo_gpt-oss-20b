#!/usr/bin/env python3
"""Run notebook cells sequentially with analysis output."""
import os, sys, json, random, re, time
import numpy as np

os.environ['HF_HUB_CACHE'] = '/root/.cache/huggingface'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ═══════════════════════════════════════════════════════════════
# CELL 2: Model Loading
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 2: Loading model...")
print("=" * 60)

from unsloth import FastLanguageModel
import torch

max_seq_length = 1024
lora_rank = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/gpt-oss-20b-BF16',
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)
print(f"Model device: {model.device}")
print(f"Model dtype: {model.dtype}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print("CELL 2: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 4: LoRA Setup
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 4: Adding LoRA adapters...")
print("=" * 60)

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

# Count trainable params
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"GPU memory after LoRA: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print("CELL 4: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 6: MinesweeperGame
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 6: Defining MinesweeperGame...")
print("=" * 60)

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

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

    def _reveal_cell(self, row, col):
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

    def _flag_cell(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed:
            return False
        if (row, col) in self._flagged:
            self._flagged.remove((row, col))
        else:
            self._flagged.add((row, col))
        return True

    def do_action(self, action):
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

    def get_visible_board(self):
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

    def state(self):
        return self._state

    def pretty_print(self):
        visible = self.get_visible_board()
        lines = []
        header = "   " + " ".join(f"{i:2d}" for i in range(self.cols))
        lines.append(header)
        lines.append("  " + "─" * (self.cols * 3 + 1))
        for r, row in enumerate(visible):
            line = f"{r:2d}│ " + "  ".join(row)
            lines.append(line)
        return "\n".join(lines)

print("MinesweeperGame class defined.")
print("CELL 6: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Game Test
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 8: Testing game...")
print("=" * 60)

game = MinesweeperGame(rows=6, cols=6, num_mines=5)
print(game.pretty_print())
print(f"State: {game.state()}")

result = game.do_action({"type": "reveal", "row": 0, "col": 0})
print(f"\nAfter revealing (0,0) -> result: {result}")
print(game.pretty_print())
print(f"State: {game.state()}")
print(f"Cells revealed: {len(game._revealed)}")
print("CELL 8: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 10: format_state_for_llm + parse_llm_action
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 10: Defining prompt/parse functions...")
print("=" * 60)

def format_state_for_llm(game):
    state = {
        "board": game.get_visible_board(),
        "rows": game.rows,
        "cols": game.cols,
        "mines": game.num_mines,
        "flags_placed": len(game._flagged),
        "cells_revealed": len(game._revealed),
    }
    prompt = f"""You are playing Minesweeper. Analyze the game state and output your next move.

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
    return prompt

def parse_llm_action(response):
    best = None
    for match in re.finditer(r'\{[^{}]*\}', response):
        try:
            action = json.loads(match.group())
            if ("type" in action and "row" in action and "col" in action
                    and action["type"] in ["reveal", "flag"]):
                best = action
        except json.JSONDecodeError:
            continue
    return best

# Test
game = MinesweeperGame(rows=6, cols=6, num_mines=5)
prompt = format_state_for_llm(game)
print(f"Prompt length: {len(prompt)} chars")
token_count = len(tokenizer.encode(prompt))
print(f"Prompt token count: {token_count}")
print("CELL 10: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 12: Base model test (pre-training)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 12: Testing base model (pre-training)...")
print("=" * 60)

game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=42)
prompt = format_state_for_llm(game)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

# Single sample to conserve memory
import torch
torch.cuda.empty_cache()
with torch.no_grad():
    output = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=128,
        do_sample=True,
    )
response = tokenizer.decode(output[0], skip_special_tokens=True)
action = parse_llm_action(response)
print(f"  Parsed action: {action}")
print(f"  Response (last 300 chars): ...{response[-300:]}")
del output
torch.cuda.empty_cache()

print("CELL 12: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 14: Reward functions (WITH MISSING HELPERS FIXED)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 14: Defining reward functions (with helper fixes)...")
print("=" * 60)

def is_logically_safe(game, row, col):
    """Check if (row, col) can be deduced as safe from adjacent revealed numbers.
    A cell is logically safe if any neighboring revealed number cell has ALL its
    adjacent mines accounted for by flags."""
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
            # Count flags around this number cell
            flag_count = 0
            for dr2 in [-1, 0, 1]:
                for dc2 in [-1, 0, 1]:
                    if dr2 == 0 and dc2 == 0:
                        continue
                    nnr, nnc = nr + dr2, nc + dc2
                    if (0 <= nnr < game.rows and 0 <= nnc < game.cols
                            and (nnr, nnc) in game._flagged):
                        flag_count += 1
            if flag_count == cell_val:
                return True
    return False

def has_revealed_neighbor(game, row, col):
    """Check if (row, col) is adjacent to any revealed cell."""
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if (0 <= nr < game.rows and 0 <= nc < game.cols
                    and (nr, nc) in game._revealed):
                return True
    return False

def valid_json_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        action = parse_llm_action(response)
        if action is None:
            scores.append(-5.0)
        else:
            scores.append(1.0)
    return scores

def gameplay_scores(completions, **kwargs):
    scores = []
    seeds = kwargs.get("seed", [])
    move_histories = kwargs.get("move_history", [])

    for idx, completion in enumerate(completions):
        response = completion[0]["content"]
        action = parse_llm_action(response)

        if action is None:
            scores.append(-10.0)
            continue

        if idx >= len(seeds) or idx >= len(move_histories):
            scores.append(0.0)
            continue

        seed = seeds[idx]
        move_history_raw = move_histories[idx]
        if isinstance(move_history_raw, str):
            move_history = json.loads(move_history_raw)
        else:
            move_history = move_history_raw

        game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=seed)
        for prev_action in move_history:
            game.do_action(prev_action)

        try:
            row, col = int(action["row"]), int(action["col"])
        except (ValueError, TypeError):
            scores.append(-10.0)
            continue

        action_type = action["type"]

        if not (0 <= row < game.rows and 0 <= col < game.cols):
            scores.append(-15.0)
            continue

        if action_type == "reveal":
            if (row, col) in game._revealed:
                scores.append(-12.0)
                continue
            if (row, col) in game._flagged:
                scores.append(-8.0)
                continue
            if game._board[row][col] == -1:
                scores.append(-25.0)
                continue

            if has_revealed_neighbor(game, row, col):
                base = 3.0
            else:
                base = 1.0

            deduction_bonus = 5.0 if is_logically_safe(game, row, col) else 0.0

            revealed_before = len(game._revealed)
            game.do_action({"type": "reveal", "row": row, "col": col})
            revealed_after = len(game._revealed)
            new_reveals = revealed_after - revealed_before
            progress_bonus = min(new_reveals - 1, 7) if new_reveals > 1 else 0.0

            win_bonus = 100.0 if game.state() == "success" else 0.0

            score = base + deduction_bonus + progress_bonus + win_bonus
            scores.append(score)

        elif action_type == "flag":
            if (row, col) in game._revealed:
                scores.append(-8.0)
                continue
            if (row, col) in game._flagged:
                scores.append(-12.0)
                continue
            if game._board[row][col] == -1:
                flag_score = 15.0
            else:
                flag_score = -10.0
            if len(game._flagged) + 1 > game.num_mines:
                flag_score -= 10.0
            scores.append(flag_score)
        else:
            scores.append(-10.0)

    return scores

# Quick sanity test of reward functions
test_completions = [[{"content": '{"type": "reveal", "row": 0, "col": 0}'}]]
test_bad = [[{"content": 'I think we should reveal something'}]]
print(f"Valid JSON reward (good): {valid_json_reward(test_completions)}")
print(f"Valid JSON reward (bad):  {valid_json_reward(test_bad)}")

# Test gameplay_scores with a real game
test_game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=42)
# Find a safe cell
safe_cells = [(r,c) for r in range(6) for c in range(6) if test_game._board[r][c] != -1]
mine_cells = [(r,c) for r in range(6) for c in range(6) if test_game._board[r][c] == -1]
print(f"\nSeed=42 board: {len(safe_cells)} safe, {len(mine_cells)} mines")
print(f"Mine positions: {mine_cells}")
print(f"Safe cell (0,0) value: {test_game._board[0][0]}")

r, c = safe_cells[0]
test_comp = [[{"content": json.dumps({"type": "reveal", "row": r, "col": c})}]]
score = gameplay_scores(test_comp, seed=[42], move_history=["[]"])
print(f"Gameplay score for safe reveal ({r},{c}): {score}")

r, c = mine_cells[0]
test_comp = [[{"content": json.dumps({"type": "reveal", "row": r, "col": c})}]]
score = gameplay_scores(test_comp, seed=[42], move_history=["[]"])
print(f"Gameplay score for mine reveal ({r},{c}): {score}")

print("CELL 14: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 16: Dataset Generation
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 16: Generating training dataset...")
print("=" * 60)

from datasets import Dataset

def generate_game_states(num_samples=1000, rows=6, cols=6, num_mines=5, rng_seed=42):
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    dataset_items = []
    attempts = 0
    max_attempts = num_samples * 3

    while len(dataset_items) < num_samples and attempts < max_attempts:
        attempts += 1
        seed = int(np.random.randint(100000))
        game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
        num_moves = int(np.random.randint(0, 6))
        move_history = []
        for _ in range(num_moves):
            board = game.get_visible_board()
            unrevealed = []
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == '.':
                        unrevealed.append((r, c))
            if unrevealed and game.state() == "ongoing":
                r, c = random.choice(unrevealed)
                action = {"type": "reveal", "row": r, "col": c}
                game.do_action(action)
                move_history.append(action)
            else:
                break
        if game.state() == "ongoing":
            prompt_text = format_state_for_llm(game)
            dataset_items.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "seed": seed,
                "move_history": json.dumps(move_history),
            })

    return Dataset.from_list(dataset_items)

dataset = generate_game_states(num_samples=1000, rows=6, cols=6, num_mines=5)
print(f"Created {len(dataset)} training examples")

fresh_count = sum(1 for item in dataset if item["move_history"] == "[]")
print(f"  Fresh games: {fresh_count} ({fresh_count/len(dataset)*100:.1f}%)")
print(f"  Mid-game states: {len(dataset) - fresh_count} ({(len(dataset)-fresh_count)/len(dataset)*100:.1f}%)")

# Analyze prompt token lengths
prompt_lengths = []
for item in dataset:
    tokens = tokenizer.encode(item["prompt"][0]["content"])
    prompt_lengths.append(len(tokens))
print(f"\nPrompt token stats:")
print(f"  Min: {min(prompt_lengths)}, Max: {max(prompt_lengths)}, Mean: {np.mean(prompt_lengths):.0f}")
print(f"  Max prompt length setting: 600")
if max(prompt_lengths) > 600:
    print(f"  WARNING: {sum(1 for l in prompt_lengths if l > 600)} prompts exceed max_prompt_length!")
else:
    print(f"  All prompts fit within max_prompt_length limit.")

print("CELL 16: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 18: GRPO Config
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 18: Setting up GRPO config...")
print("=" * 60)

from trl import GRPOConfig, GRPOTrainer

max_prompt_length = 600
max_completion_length = max_seq_length - max_prompt_length

training_args = GRPOConfig(
    temperature=1.0,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=500,
    save_steps=100,
    report_to="none",
    output_dir="minesweeper_custom_outputs",
)

print(f"  Max steps: {training_args.max_steps}")
print(f"  Generations per state: {training_args.num_generations}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective completions per update: {training_args.num_generations * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Max prompt length: {max_prompt_length}")
print(f"  Max completion length: {max_completion_length}")
print(f"  Temperature: {training_args.temperature}")
print("CELL 18: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 19: Eval Callback
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 19: Creating eval callback...")
print("=" * 60)

from transformers import TrainerCallback

class MinesweeperEvalCallback(TrainerCallback):
    def __init__(self, eval_every_steps=50, num_games=5):
        self.eval_every_steps = eval_every_steps
        self.num_games = num_games

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if state.global_step % self.eval_every_steps != 0:
            return
        tokenizer = processing_class
        if tokenizer is None or model is None:
            return
        was_training = model.training
        model.eval()
        wins = 0
        for i in range(self.num_games):
            game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=10000 + i)
            moves = 0
            while game.state() == "ongoing" and moves < 50:
                prompt = format_state_for_llm(game)
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                output = model.generate(
                    **tokenizer(text, return_tensors="pt").to(model.device),
                    temperature=0.7,
                    max_new_tokens=128,
                    do_sample=True,
                )
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                action = parse_llm_action(response)
                if action is None:
                    break
                game.do_action(action)
                moves += 1
            if game.state() == "success":
                wins += 1
        win_rate = wins / self.num_games
        print(f"\n[Eval @ step {state.global_step}] Win rate: {wins}/{self.num_games} ({win_rate*100:.0f}%)\n")
        if was_training:
            model.train()

eval_callback = MinesweeperEvalCallback(eval_every_steps=50, num_games=5)
print("Eval callback: plays 5 games every 50 steps")
print("CELL 19: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 21: GRPOTrainer setup + train
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 21: Setting up trainer and starting training...")
print("=" * 60)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        valid_json_reward,
        gameplay_scores,
    ],
    args=training_args,
    train_dataset=dataset,
    callbacks=[eval_callback],
)

print("Trainer created. Starting training...")
sys.stdout.flush()

train_start = time.time()
trainer.train()
train_elapsed = time.time() - train_start
print(f"\nTraining completed in {train_elapsed/60:.1f} minutes")
print("CELL 21: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 25: Post-training evaluation (100 games)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 25: Post-training evaluation (100 games)...")
print("=" * 60)

def play_full_game(model, tokenizer, rows=6, cols=6, num_mines=5, seed=None, max_moves=50):
    game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
    moves = 0
    actions_taken = []
    end_reason = None
    while game.state() == "ongoing" and moves < max_moves:
        prompt = format_state_for_llm(game)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        output = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            temperature=0.7,
            max_new_tokens=128,
            do_sample=True,
        )
        response = tokenizer.decode(output[0])
        action = parse_llm_action(response)
        if action is None:
            end_reason = "parse_error"
            break
        result = game.do_action(action)
        actions_taken.append(action)
        if result in ("mine", "invalid_format", "out_of_bounds", "already_revealed", "flagged_cell", "invalid_flag"):
            end_reason = result
        moves += 1
    if game.state() == "success":
        end_reason = "win"
    elif end_reason is None:
        end_reason = "max_moves"
    return game, moves, end_reason, actions_taken

NUM_EVAL_GAMES = 100
print(f"Evaluating on {NUM_EVAL_GAMES} games...")
wins = 0
total_moves = 0
end_reasons = {}

for i in range(NUM_EVAL_GAMES):
    game, moves, end_reason, actions = play_full_game(model, tokenizer, seed=i)
    end_reasons[end_reason] = end_reasons.get(end_reason, 0) + 1
    if game.state() == "success":
        wins += 1
    if i < 10 or game.state() == "success":
        tag = "WIN" if game.state() == "success" else "LOSS"
        print(f"Game {i+1}: {tag} ({end_reason}) after {moves} moves")
    total_moves += moves

print(f"\n{'='*40}")
print(f"RESULTS ({NUM_EVAL_GAMES} games):")
print(f"  Win rate: {wins}/{NUM_EVAL_GAMES} ({wins/NUM_EVAL_GAMES*100:.1f}%)")
print(f"  Average moves: {total_moves/NUM_EVAL_GAMES:.1f}")
print(f"  End reasons: {json.dumps(end_reasons, indent=4)}")
print("CELL 25: DONE\n")

# ═══════════════════════════════════════════════════════════════
# CELL 27: Save model
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CELL 27: Saving model...")
print("=" * 60)

model.save_pretrained("my_minesweeper_model")
tokenizer.save_pretrained("my_minesweeper_model")
print("Model saved to: my_minesweeper_model/")
print("CELL 27: DONE\n")

print("=" * 60)
print("ALL CELLS COMPLETE")
print("=" * 60)
