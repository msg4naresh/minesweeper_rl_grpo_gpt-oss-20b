#!/usr/bin/env python3
import os, json, math, random, re, shutil
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from collections import Counter
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =============================
# Model loading + LoRA
# =============================
max_seq_length = 1024
lora_rank = 32

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-BF16",
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Patch chat template: force generation to start in the "final" channel.
# Without this, the model outputs <|channel|>analysis<|message|>...reasoning...
# which burns the entire 128-token completion budget before producing JSON.
_ct = tokenizer.chat_template
_ct = _ct.replace('{%- set reasoning_effort = "medium" %}',
                   '{%- set reasoning_effort = "none" %}')
_ct = _ct.replace('<|start|>assistant\n{%- endif %}',
                   '<|start|>assistant<|channel|>final<|message|>\n{%- endif %}')
tokenizer.chat_template = _ct

# =============================
# Game engine
# =============================
@dataclass
class MinesweeperGame:
    rows: int; cols: int; num_mines: int; seed: Optional[int]=None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _revealed: Set[Tuple[int,int]] = field(init=False, repr=False, default_factory=set)
    _flagged: Set[Tuple[int,int]] = field(init=False, repr=False, default_factory=set)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        if self.num_mines >= self.rows * self.cols: raise ValueError("Too many mines")
        self._rng = random.Random(self.seed)
        self._board = [[0]*self.cols for _ in range(self.rows)]
        self._place_mines(); self._calc_nums()

    def _place_mines(self):
        for r,c in self._rng.sample([(r,c) for r in range(self.rows) for c in range(self.cols)], self.num_mines):
            self._board[r][c] = -1

    def _calc_nums(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self._board[r][c] == -1: continue
                cnt = sum(self._board[nr][nc]==-1 for nr,nc in get_neighbors(r,c,self.rows,self.cols))
                self._board[r][c] = cnt

    def _reveal_cell(self, r,c):
        if not (0<=r<self.rows and 0<=c<self.cols): return False
        if (r,c) in self._revealed or (r,c) in self._flagged: return False
        stack=[(r,c)]
        while stack:
            rr,cc=stack.pop()
            if (rr,cc) in self._revealed: continue
            self._revealed.add((rr,cc))
            if self._board[rr][cc]==-1:
                self._state="failed"; return True
            if self._board[rr][cc]==0:
                for nr,nc in get_neighbors(rr,cc,self.rows,self.cols):
                    if (nr,nc) not in self._revealed and (nr,nc) not in self._flagged:
                        stack.append((nr,nc))
        return True

    def _flag_cell(self,r,c):
        if not (0<=r<self.rows and 0<=c<self.cols): return False
        if (r,c) in self._revealed: return False
        if (r,c) in self._flagged: self._flagged.remove((r,c))
        else: self._flagged.add((r,c))
        return True

    def do_action(self, action: dict) -> str:
        if self._state!="ongoing": return "game_over"
        if not isinstance(action, dict): self._state="failed"; return "invalid_format"
        t,r,c = action.get("type"), action.get("row"), action.get("col")
        try: r,c=int(r),int(c)
        except Exception: self._state="failed"; return "invalid_format"
        if t not in ["reveal","flag"] or not (0<=r<self.rows and 0<=c<self.cols):
            self._state="failed"; return "invalid_format"
        if t=="reveal":
            if (r,c) in self._revealed: self._state="failed"; return "already_revealed"
            if (r,c) in self._flagged: self._state="failed"; return "flagged_cell"
            ok=self._reveal_cell(r,c)
        else:
            if (r,c) in self._revealed: self._state="failed"; return "invalid_flag"
            ok=self._flag_cell(r,c)
        if not ok: self._state="failed"; return "invalid_format"
        self._check_win()
        if self._state=="failed": return "mine"
        if self._state=="success": return "win"
        return "ok"

    def _check_win(self):
        if len(self._revealed)== self.rows*self.cols - self.num_mines:
            self._state="success"

    def get_visible_board(self):
        vis=[]
        for r in range(self.rows):
            row=[]
            for c in range(self.cols):
                if (r,c) in self._flagged: row.append('F')
                elif (r,c) in self._revealed:
                    v=self._board[r][c]; row.append('*' if v==-1 else str(v))
                else: row.append('.')
            vis.append(row)
        return vis

    def state(self): return self._state

    def clone(self):
        g=object.__new__(MinesweeperGame)
        g.rows,g.cols,g.num_mines,g.seed=self.rows,self.cols,self.num_mines,self.seed
        g._rng=random.Random()
        g._board=[r[:] for r in self._board]
        g._revealed=set(self._revealed); g._flagged=set(self._flagged); g._state=self._state
        return g

# =============================
# Prompt + parser
# =============================
SYSTEM_PROMPT = "You output JSON actions for Minesweeper. No text, only JSON."

def build_prompt(game: MinesweeperGame):
    state={
        "board":game.get_visible_board(),
        "rows":game.rows,"cols":game.cols,"mines":game.num_mines,
        "flags_placed":len(game._flagged),"cells_revealed":len(game._revealed),
    }
    user=f"""You are playing Minesweeper. Analyze the game state and output your next move.

You must output ONLY a valid JSON object. No explanation, no analysis, no text.
Start your response immediately with {{ and end with }}.
Do NOT output cell which is already revealed or flagged.

Game state:
{json.dumps(state, indent=2)}

Output your next action as JSON:
{{"type": "reveal", "row": <row_index>, "col": <col_index>}}
or
{{"type": "flag", "row": <row_index>, "col": <col_index>}}

Your action:"""
    return user, SYSTEM_PROMPT

def parse_llm_action(resp: str) -> Optional[dict]:
    for m in re.finditer(r'\{[^{}]*\}', resp):
        try:
            a=json.loads(m.group())
            if set(["type","row","col"])<=a.keys() and a["type"] in ["reveal","flag"]:
                a["row"],a["col"]=int(a["row"]),int(a["col"])
                return a
        except Exception:
            continue
    t=re.search(r'"type"\s*:\s*"(reveal|flag)"', resp)
    r=re.search(r'"row"\s*:\s*(\d+)', resp)
    c=re.search(r'"col"\s*:\s*(\d+)', resp)
    if t and r and c:
        return {"type":t.group(1),"row":int(r.group(1)),"col":int(c.group(1))}
    return None

# =============================
# Deduction helpers
# =============================
def get_neighbors(r,c,rows,cols):
    out=[]
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            nr,nc=r+dr,c+dc
            if 0<=nr<rows and 0<=nc<cols: out.append((nr,nc))
    return out

def compute_deducible_cells(board: List[List[str]], rows: int, cols: int):
    safe, mines = set(), set()
    constraints=[]
    # pass 1
    for r in range(rows):
        for c in range(cols):
            if not board[r][c].isdigit(): continue
            num=int(board[r][c])
            neigh=get_neighbors(r,c,rows,cols)
            unrevealed=frozenset((nr,nc) for nr,nc in neigh if board[nr][nc]=='.')
            flagged_cnt=sum(1 for nr,nc in neigh if board[nr][nc]=='F')
            rem=num-flagged_cnt
            if unrevealed:
                constraints.append([set(unrevealed), rem])
                if rem==0: safe.update(unrevealed)
                elif rem==len(unrevealed): mines.update(unrevealed)
    # pass 2 subset
    changed=True
    while changed:
        changed=False
        constraints.sort(key=lambda x: len(x[0]))
        for i in range(len(constraints)):
            for j in range(i+1,len(constraints)):
                c1,r1=constraints[i]; c2,r2=constraints[j]
                if not c1 or not c2: continue
                if c1.issubset(c2):
                    diff=c2-c1; dm=r2-r1
                    if diff:
                        if dm==0 and not diff.issubset(safe):
                            safe.update(diff); changed=True
                        elif dm==len(diff) and not diff.issubset(mines):
                            mines.update(diff); changed=True
                        if changed:
                            constraints[j]=[diff, dm]
    return safe - mines, mines

def estimate_mine_probability(game: MinesweeperGame, row: int, col: int) -> float:
    max_prob=-1.0
    for nr,nc in get_neighbors(row,col,game.rows,game.cols):
        if (nr,nc) not in game._revealed: continue
        val=game._board[nr][nc]
        if val<=0: continue
        neigh=get_neighbors(nr,nc,game.rows,game.cols)
        flag_cnt=sum(1 for nnr,nnc in neigh if (nnr,nnc) in game._flagged)
        hid_cnt=sum(1 for nnr,nnc in neigh if (nnr,nnc) not in game._revealed and (nnr,nnc) not in game._flagged)
        rem=val-flag_cnt
        if hid_cnt>0 and rem>=0:
            max_prob=max(max_prob, rem/hid_cnt)
    if max_prob>=0: return min(max_prob,1.0)
    total_hidden=game.rows*game.cols - len(game._revealed) - len(game._flagged)
    rem_global=game.num_mines - len(game._flagged)
    return max(0.01, rem_global/total_hidden) if total_hidden>0 else 0.5

# =============================
# Combined reward with rollout
# =============================
def reward_combined(completions, **kw):
    scores=[]
    seeds=kw.get("seed",[])
    rowsL=kw.get("rows",[])
    colsL=kw.get("cols",[])
    minesL=kw.get("mines",[])
    histories=kw.get("move_history",[])
    for i,comp in enumerate(completions):
        resp=comp[0]["content"]
        if not resp or not resp.strip(): scores.append(-50.0); continue
        act=parse_llm_action(resp)
        if act is None: scores.append(-50.0); continue
        if i>=len(seeds): scores.append(0.0); continue
        rows,cols,mines = int(rowsL[i]), int(colsL[i]), int(minesL[i])
        hist = json.loads(histories[i]) if isinstance(histories[i], str) else histories[i]

        g=MinesweeperGame(rows, cols, mines, seed=int(seeds[i]))
        for a in hist: g.do_action(a)

        r,c,t = act.get("row"), act.get("col"), act.get("type")
        try: r,c=int(r),int(c)
        except Exception: scores.append(-50.0); continue
        if t not in ["reveal","flag"] or not (0<=r<rows and 0<=c<cols):
            scores.append(-50.0); continue
        if (r,c) in g._revealed: scores.append(-20.0); continue
        if t=="reveal" and (r,c) in g._flagged: scores.append(-15.0); continue
        if t=="flag" and (r,c) in g._flagged: scores.append(-10.0); continue

        base=3.0
        vis=g.get_visible_board()
        safe_cells, mine_cells = compute_deducible_cells(vis, rows, cols)

        g2=g.clone()
        outcome=g2.do_action(act)
        if outcome=="mine":
            scores.append(base-40.0); continue
        if g2.state()=="success":
            scores.append(base+100.0); continue

        move_score=base
        if t=="reveal":
            move_score+=10.0
            if (r,c) in safe_cells:
                move_score+=35.0
            else:
                chosen_p=estimate_mine_probability(g,r,c)
                min_p=1.0
                for rr in range(rows):
                    for cc in range(cols):
                        if (rr,cc) in g._revealed or (rr,cc) in g._flagged: continue
                        p=estimate_mine_probability(g, rr, cc)
                        min_p=min(min_p,p)
                if chosen_p - min_p > 0.15:
                    move_score-=15.0
            before=len(g._revealed); g.do_action(act)
            new_rev=len(g._revealed)-before
            move_score+= min(math.log2(new_rev+1)*2.0, 5.0) if new_rev>0 else 0.0
        else:
            if g._board[r][c]==-1: move_score+=15.0
            else: move_score-=8.0
            if (r,c) in mine_cells: move_score+=20.0
            if len(g._flagged)+1>g.num_mines: move_score-=8.0

        # rollout on g2
        rollout_bonus=0.0
        MAXR=10; survived=0; won=False
        for k in range(MAXR):
            if g2.state()!="ongoing": break
            vis2=g2.get_visible_board()
            safe2,mine2=compute_deducible_cells(vis2, g2.rows, g2.cols)
            h=None
            if safe2:
                sr,sc=next(iter(safe2)); h={"type":"reveal","row":sr,"col":sc}
            elif mine2:
                mr,mc=next(iter(mine2)); h={"type":"flag","row":mr,"col":mc}
            else:
                bestp, best = 1.1, None
                for rr in range(g2.rows):
                    for cc in range(g2.cols):
                        if (rr,cc) in g2._revealed or (rr,cc) in g2._flagged: continue
                        p=estimate_mine_probability(g2, rr, cc)
                        if p<bestp: bestp,p; best=(rr,cc)
                if best: h={"type":"reveal","row":best[0],"col":best[1]}
            if h is None: break
            g2.do_action(h)
            if g2.state()=="success": won=True; survived=k+1; break
            if g2.state()=="failed": survived=k; break
            survived=k+1
        if won: rollout_bonus=30.0
        elif survived>=MAXR: rollout_bonus=10.0
        else: rollout_bonus=float(survived)

        scores.append(move_score + rollout_bonus)
    return scores

# =============================
# Dataset generation
# =============================
def generate_dataset(configs, num_samples=1000, rng_seed=42):
    np.random.seed(rng_seed); random.seed(rng_seed)
    items=[]
    for rows, cols, mines, w in configs:
        target=int(num_samples*w)
        attempts=0
        while len([it for it in items if it["rows"]==rows and it["cols"]==cols and it["mines"]==mines])<target and attempts<target*6:
            attempts+=1
            seed=int(np.random.randint(100000))
            g=MinesweeperGame(rows, cols, mines, seed=seed)
            moves=int(np.random.randint(1,6))  # at least 1 move
            mh=[]
            for _ in range(moves):
                if g.state()!="ongoing": break
                board=g.get_visible_board()
                hidden=[(r,c) for r in range(rows) for c in range(cols) if board[r][c]=='.' and (r,c) not in g._flagged]
                if not hidden: break
                r,c=random.choice(hidden)
                a={"type":"reveal","row":r,"col":c}; g.do_action(a); mh.append(a)
                if g.state()=="ongoing" and random.random()<0.15:
                    hidden=[hc for hc in hidden if hc not in g._revealed and hc not in g._flagged]
                    if hidden:
                        fr,fc=random.choice(hidden)
                        fa={"type":"flag","row":fr,"col":fc}; g.do_action(fa); mh.append(fa)
            if g.state()!="ongoing": continue
            user, sys = build_prompt(g)
            items.append({
                "prompt":[{"role":"system","content":sys},{"role":"user","content":user}],
                "seed":seed,"rows":rows,"cols":cols,"mines":mines,
                "move_history":json.dumps(mh),
            })
    random.shuffle(items)
    return Dataset.from_list(items)

# =============================
# Curriculum callback
# =============================
class CurriculumCallback(TrainerCallback):
    def __init__(self, phases): self.phases=phases
    def on_step_begin(self, args, state, control, **kw):
        for start,end,ds in self.phases:
            if start <= state.global_step < end:
                control.train_dataset = ds
                break

# =============================
# Eval callback (best-of-3)
# =============================
class MinesweeperEvalCallback(TrainerCallback):
    def __init__(self, every=50, games=5, rows=4, cols=4, mines=2):
        self.every=every; self.games=games; self.rows=rows; self.cols=cols; self.mines=mines
    def on_step_end(self, args, state, control, model=None, processing_class=None, **kw):
        if state.global_step % self.every != 0: return
        tok=processing_class
        if tok is None or model is None: return
        was=model.training; model.eval()
        wins=0
        for i in range(self.games):
            g=MinesweeperGame(self.rows,self.cols,self.mines,seed=10000+i)
            moves=0
            while g.state()=="ongoing" and moves<50:
                user, sys = build_prompt(g)
                text=tok.apply_chat_template(
                    [{"role":"system","content":sys},{"role":"user","content":user}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                candidates=[]
                for _ in range(3):
                    out=model.generate(
                        **tok(text, return_tensors="pt").to(model.device),
                        temperature=0.55, max_new_tokens=112, do_sample=True,
                        top_p=0.9, repetition_penalty=1.1,
                    )
                    resp=tok.decode(out[0][len(tok(text, return_tensors="pt")["input_ids"][0]):], skip_special_tokens=True)
                    a=parse_llm_action(resp)
                    if a: candidates.append(a)
                best=None; bestp=2.0
                for a in candidates:
                    r,c=a["row"],a["col"]
                    if not (0<=r<g.rows and 0<=c<g.cols): continue
                    if (r,c) in g._revealed or (r,c) in g._flagged: continue
                    p=estimate_mine_probability(g,r,c)
                    if p<bestp: bestp=p; best=a
                if best is None: break
                res=g.do_action(best)
                if res not in ("ok","win"): break
                moves+=1
            if g.state()=="success": wins+=1
        print(f"\n[Eval step {state.global_step} {self.rows}x{self.cols}/{self.mines}m] Win {wins}/{self.games}\n")
        if was: model.train()

# =============================
# Training setup
# =============================
max_prompt_length=896
max_completion_length=112

PHASES=[
    {"range":(0,200),  "configs":[(4,4,2,0.6),(4,5,2,0.4)]},
    {"range":(200,400),"configs":[(5,5,3,0.5),(5,6,3,0.5)]},
    {"range":(400,600),"configs":[(6,6,5,0.6),(6,7,5,0.4)]},
]

datasets=[]
for idx,ph in enumerate(PHASES):
    ds=generate_dataset(ph["configs"], num_samples=1000, rng_seed=42+idx)
    datasets.append((ph["range"][0], ph["range"][1], ds))
    print(f"Phase {idx+1} dataset sizes:", Counter((r["rows"],r["cols"],r["mines"]) for r in ds))

training_args=GRPOConfig(
    temperature=0.8,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    bf16=True,
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=8,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=600,
    save_steps=200,
    report_to="none",
    output_dir="minesweeper_v3_outputs",
)

eval_cb=MinesweeperEvalCallback(every=50, games=5, rows=4, cols=4, mines=2)
curr_cb=CurriculumCallback(datasets)

trainer=GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_combined],
    args=training_args,
    train_dataset=datasets[0][2],
    callbacks=[curr_cb, eval_cb],
)

print("Starting training...")
trainer.train()
print("Training done.")

# =============================
# Quick post-training evals
# =============================
def play_eval(rows, cols, mines, games=20):
    wins=0; reasons=[]
    for i in range(games):
        g=MinesweeperGame(rows, cols, mines, seed=20000+i)
        moves=0; end="max_moves"
        while g.state()=="ongoing" and moves<50:
            user, sys = build_prompt(g)
            text=tokenizer.apply_chat_template(
                [{"role":"system","content":sys},{"role":"user","content":user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            out=model.generate(
                **tokenizer(text, return_tensors="pt").to(model.device),
                temperature=0.55, max_new_tokens=112, do_sample=True,
                top_p=0.9, repetition_penalty=1.1,
            )
            resp=tokenizer.decode(out[0][len(tokenizer(text, return_tensors="pt")["input_ids"][0]):], skip_special_tokens=True)
            a=parse_llm_action(resp)
            if a is None: end="parse_error"; break
            res=g.do_action(a)
            if res=="mine": end="mine"; break
            if res=="win": end="win"; break
            if res not in ("ok","win"): end=f"illegal:{res}"; break
            moves+=1
        if g.state()=="success": wins+=1
        reasons.append(end)
    print(f"{rows}x{cols}/{mines}m: win {wins}/{games} ({wins/games*100:.0f}%), reasons={Counter(reasons)}")

print("\nPost-training quick evals:")
play_eval(6,6,5)
play_eval(4,4,2)

# =============================
# Save
# =============================
print("\nSaving LoRA adapters...")
model.save_pretrained("/workspace/my_minesweeper_model_v3")
tokenizer.save_pretrained("/workspace/my_minesweeper_model_v3")
print("Saving merged model...")
model.save_pretrained_merged(
    "/workspace/your_finetuned_model",
    tokenizer,
    save_method="merged_16bit",
)
print("Done.")

