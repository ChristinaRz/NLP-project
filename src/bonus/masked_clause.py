"""
BONUS - Masked Clause Input (Greek legal text)

εκτελείται απο root:
python -m src.bonus.masked_clause
"""

from __future__ import annotations

import re
import pandas as pd
from transformers import pipeline
import stanza
import unicodedata

# masked inputs αρχικά
ARTICLE_1113_MASKED = (
    "Άρθρο 1113. Κοινό πράγμα. — Αν η κυριότητα του [MASK] ανήκει σε περισσότερους "
    "[MASK] αδιαιρέτου κατ΄ιδανικά [MASK], εφαρμόζονται οι διατάξεις για την κοινωνία."
)

ARTICLE_1114_MASKED = (
    "Άρθρο 1114. Πραγματική δουλεία σε [MASK] η υπέρ του κοινού ακινήτου. — Στο κοινό "
    "[MASK] μπορεί να συσταθεί πραγματική δουλεία υπέρ του [MASK] κύριου άλλου ακινήτου "
    "και αν ακόμη αυτός είναι [MASK] του ακινήτου που βαρύνεται με τη δουλεία. Το ίδιο ισχύει "
    "και για την [MASK] δουλεία πάνω σε ακίνητο υπέρ των εκάστοτε κυρίων κοινού ακινήτου, "
    "αν [MASK] από αυτούς είναι κύριος του [MASK] που βαρύνεται με τη δουλεία."
)

# ground truth
GT_1113 = ["πράγματος", "κυρίους", "μέρη"]
GT_1114 = ["βάρος", "ακίνητο", "εκάστοτε", "συγκύριος", "πραγματική", "κάποιος", "ακινήτου"]


# cache pipelines
_FM_CACHE: dict[str, object] = {}


def normalize_token(tok: str) -> str:
    # καθαρισμός περιττών spaces
    #σήμανση tokenizers
    tok = tok.strip()
    tok = tok.replace("##", "")
    return tok

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def get_fill_mask_pipe(model_name: str, top_k: int = 10):
    if model_name not in _FM_CACHE:
        _FM_CACHE[model_name] = pipeline("fill-mask", model=model_name, top_k=top_k)
    return _FM_CACHE[model_name]


def run_fill_mask(model_name: str, masked_text: str, top_k: int = 10) -> list[list[dict]]:
    fm = get_fill_mask_pipe(model_name, top_k=top_k)

    # εντοπισμός positions των [MASK]
    mask_positions = [m.start() for m in re.finditer(r"\[MASK\]", masked_text)]
    all_preds = []

    for _ in mask_positions:
        # pipeline("fill-mask") όταν έχει πολλά [MASK], συνήθως δουλεύει για το πρώτο.
        # κάθε φορά φτιάχνεται κείμενο που το πρώτο mask είναι αυτό που αξιολογείται
        # προσωρινή αντικατάσταση των προηγούμενων masks με placeholder που ΔΕΝ είναι [MASK]
        #μετακίνηση που πρώτου [MASK] στη θέση που θέλουμε

        # replace counter
        text = masked_text
        # όλα τα [MASK] -> [TMP]
        text = text.replace("[MASK]", "[TMP]")
        # επαναφορά ΜΟΝΟ το πρώτο [TMP] -> [MASK]
        text = text.replace("[TMP]", "[MASK]", 1)

        preds = fm(text)  # list of dicts
        all_preds.append(preds)

        # το πρώτο mask σε dummy token
        best = normalize_token(preds[0]["token_str"])
        masked_text = masked_text.replace("[MASK]", best, 1)

    #επιστροφή  predictions για ΚΑΘΕ mask θέση
    return all_preds


def evaluate(preds_per_mask: list[list[dict]], ground_truth: list[str], model_label: str, article_label: str) -> pd.DataFrame:
    rows = []
    for i, (preds, gt) in enumerate(zip(preds_per_mask, ground_truth), start=1):
        pred_tokens = [normalize_token(p["token_str"]) for p in preds]
        top1 = pred_tokens[0] if pred_tokens else ""
        top5 = pred_tokens[:5]
        top10 = pred_tokens[:10]

# !σύγκριση χωρίς τόνους (GreekBERT συχνά επιστρέφει άτονα)
        gt_n = strip_accents(gt.lower())
        top1_n = strip_accents(top1.lower())
        top5_n = [strip_accents(t.lower()) for t in top5]
        top10_n = [strip_accents(t.lower()) for t in top10]

        hit1 = int(top1_n == gt_n)
        hit5 = int(gt_n in top5_n)
        hit10 = int(gt_n in top10_n)


        # MRR
        rr = 0.0
        pred_tokens_n = [strip_accents(t.lower()) for t in pred_tokens]
        if gt_n in pred_tokens_n:
            rank = pred_tokens_n.index(gt_n) + 1
            rr = 1.0 / rank

        rows.append({
            "Article": article_label,
            "Model": model_label,
            "MaskIndex": i,
            "GroundTruth": gt,
            "Top1": top1,
            "Top5": ", ".join(top5),
            "Top10": ", ".join(top10),
            "Hit@1": hit1,
            "Hit@5": hit5,
            "Hit@10": hit10,
            "MRR": round(rr, 4),
        })
    return pd.DataFrame(rows)

# POS + dependency για ορισμενες κρίσιμες λέξεις
def syntax_check(text: str, nlp) -> list[dict]:
    doc = nlp(text)
    out = []
    for sent in doc.sentences:
        for w in sent.words:
            out.append({
                "text": w.text,
                "lemma": w.lemma,
                "upos": w.upos,
                "deprel": w.deprel,
                "head": w.head,
            })
    return out


def main():
    # Stanza Greek parser (POS+DEP)
    nlp = stanza.Pipeline("el", processors="tokenize,pos,lemma,depparse", tokenize_no_ssplit=False)

    models = [
        ("GreekBERT", "nlpaueb/bert-base-greek-uncased-v1"),
        ("mBERT", "bert-base-multilingual-cased"),
    ]

    results = []

    # Article 1113
    for label, name in models:
        preds = run_fill_mask(name, ARTICLE_1113_MASKED, top_k=10)
        df = evaluate(preds, GT_1113, label, "1113")
        results.append(df)

    # Article 1114
    for label, name in models:
        preds = run_fill_mask(name, ARTICLE_1114_MASKED, top_k=10)
        df = evaluate(preds, GT_1114, label, "1114")
        results.append(df)

    all_df = pd.concat(results, ignore_index=True)

    # metrics
    summary = (all_df.groupby(["Article", "Model"])[["Hit@1", "Hit@5", "Hit@10", "MRR"]]
               .mean()
               .reset_index())

    print("\n=== Detailed results (per mask) ===\n")
    print(all_df.to_string(index=False))

    print("\n=== Summary (mean over masks) ===\n")
    print(summary.to_string(index=False))

    # save
    import os
    os.makedirs("outputs", exist_ok=True)
    all_df.to_csv("outputs/bonus_mask_results.csv", index=False)
    summary.to_csv("outputs/bonus_mask_summary.csv", index=False)
    print("\nSaved: outputs/bonus_mask_results.csv")
    print("Saved: outputs/bonus_mask_summary.csv")

    ## Συντακτική ανάλυση##  parse για το ground truth
    #  πλήρες κείμενο ground truth αντικαθιστώντας το με τις σωστές λέξεις
    gt_1113_text = ARTICLE_1113_MASKED
    for w in GT_1113:
        gt_1113_text = gt_1113_text.replace("[MASK]", w, 1)

    print("\n=== Syntax snapshot (GT 1113) ===\n")
    syn = syntax_check(gt_1113_text, nlp)
    #περιορισμός μήκους
    for row in syn[:25]:
        print(row)


if __name__ == "__main__":
    main()
