"""Paper 9 manuscript audit, small extra checks (CPU, 1 thread):
(1) v2 kappa under the role-based chance model (donor / recipient / other, per boundary B) used by the v1/v2 audit,
    and the conventional option-level kappa; (2) token counts of the immediate and delayed report tails with the cached
    Qwen3-1.7B tokenizer (local files only, no download); (3) Stage 0a's report prompt (v3 four-option or v4 two-option).
Writes audit/paper_audit/extra_checks_paper.json."""
import glob
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P9 = os.path.dirname(os.path.dirname(HERE))
OUT = {}

# (1) kappa
P = json.load(open(os.path.join(P9, "res_v2", "premise.json"), encoding="utf-8"))
scr = {i["city"]: i for i in json.load(open(os.path.join(P9, "res", "screen.json"), encoding="utf-8"))["items"] if i.get("ok")}
by = {}
for r in P["B"]:
    by.setdefault((r["kind"], r["B"]), []).append(r)


def role(r):
    return "donor" if r["pick"] == scr[r["y"]]["country"] else "recipient" if r["pick"] == scr[r["x"]]["country"] else "other"


agree, pe_role, pe_opt = [], [], []
for B in P["bounds"]:
    rep = {(r["x"], r["y"]): r for r in by[("report", B)]}
    ro = {(r["x"], r["y"]): r for r in by[("readout", B)]}
    ks = sorted(rep)
    agree += [rep[k]["pick"] == ro[k]["pick"] for k in ks]
    a = [role(rep[k]) for k in ks]
    b = [role(ro[k]) for k in ks]
    pe_role.append(sum(np.mean([x == c for x in a]) * np.mean([x == c for x in b]) for c in ("donor", "recipient", "other")))
    ia = [rep[k]["cands"].index(rep[k]["pick"]) for k in ks]
    ib = [rep[k]["cands"].index(ro[k]["pick"]) for k in ks]
    pe_opt.append(sum(np.mean([x == j for x in ia]) * np.mean([x == j for x in ib]) for j in range(4)))
po = float(np.mean(agree))
OUT["kappa"] = {"observed": po, "pe_role_mean": float(np.mean(pe_role)), "kappa_role": (po - np.mean(pe_role)) / (1 - np.mean(pe_role)),
                "pe_option_mean": float(np.mean(pe_opt)), "kappa_option": (po - np.mean(pe_opt)) / (1 - np.mean(pe_opt)),
                "same_cands_report_readout": all(r["cands"] == s["cands"] for B in P["bounds"]
                                                 for r, s in zip(sorted(by[("report", B)], key=lambda z: (z["x"], z["y"])),
                                                                 sorted(by[("readout", B)], key=lambda z: (z["x"], z["y"]))))}

# (2) token counts of the report tails (strings as in code_v4/v4lib.py and code/p9lib.py)
snap = glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--Qwen--Qwen3-1.7B", "snapshots", "*"))[0]
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
opts = "A) Very sure\nB) Less sure"
PREFILL, TAIL_END = "The answer is (", "\nAnswer with the letter."
q = "How sure were you of the second digit of your answer?" + "\n" + opts + TAIL_END
tail_turn2 = enc("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + PREFILL)
tail_del = enc("<|im_end|>\n") + tail_turn2
tail_imm = enc("\n\n" + "How sure am I of the second digit of this answer?" + "\n" + opts + "\n" + PREFILL)
OUT["tokens"] = {"imm_len": len(tail_imm), "del_len": len(tail_del), "im_end_nl_len": len(enc("<|im_end|>\n")),
                 "imm_tokens": [tok.decode([t]) for t in tail_imm],
                 "tailpos_tokens_match": [tok.decode([t]) for t in tail_imm] == json.load(open(os.path.join(P9, "stage0b", "tailpos.json"), encoding="utf-8"))["positions"][1:]}

# (3) Stage 0a's report prompt
src = open(os.path.join(P9, "stage0", "s0a_extract.py"), encoding="utf-8").read()
OUT["stage0a_tail_is_v4_two_option"] = "m.tail(v4lib.Q_REPORT, v4lib.rotate2(v4lib.BINS, 0))" in src
OUT["stage0a_swap_is_v3_unit_swap"] = "v3lib.unit_swap(" in src
json.dump(OUT, open(os.path.join(HERE, "extra_checks_paper.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != "tokens"}, indent=1))
print({k: v for k, v in OUT["tokens"].items() if k != "imm_tokens"})
