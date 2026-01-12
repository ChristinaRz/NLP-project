"""
Παραδοτέο 1Β
Pipeline 2 — Neural rewriting με Flan-T5 (Transformers)

το script:
1) Κανονικοποιεί το κείμενο
2) instruction-style prompt
3) Tokenization (AutoTokenizer) και παραγωγή output (AutoModelForSeq2SeqLM).
4) Generation χωρίς sampling (do_sample=False)
5) Επιστρέφει αποτέλεσμα + γρήγορο έλεγχο

"""

from __future__ import annotations

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.data.texts import TEXT1, TEXT2


# Προεκπαιδευμένο text-to-text μοντέλο (encoder-decoder)
MODEL_NAME = "google/flan-t5-base"

# cache για φορτώνονται 1 φορά
_TOKENIZER = None
_MODEL = None


def normalize(text: str) -> str:
    #εφαρμογη strip και διαγραφεί περιττών spaces
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_t5():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()        #interfence
    return tokenizer, model


def reconstruct_with_t5(text: str, tokenizer, model) -> str:
    text = normalize(text)

    prompt = (
        "Rewrite the following text to be grammatically correct, clear, and well-structured. "
        "Keep the meaning and do not add new information:\n\n"
        f"{text}"
    )

    # μετατροπή prompt σε tensors (tokenization)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        #generate με beam search (χωρίς randomness)
        out_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=4,
            max_new_tokens=220,
            min_new_tokens=20,
            #  για να μειωθούν επαναλήψεις
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    # decode σε string
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return normalize(decoded)


#wrapper
def reconstruct(text: str) -> str:
    global _TOKENIZER, _MODEL

    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER, _MODEL = load_t5()

    return reconstruct_with_t5(text, _TOKENIZER, _MODEL)


def main() -> None:
    # seed για στθερότητα
    torch.manual_seed(0)

    out1 = reconstruct(TEXT1)
    print("\nTEXT 1:\n")
    print(out1)
    print("\n--- quick check ---")
    print("same_as_input:", out1.strip() == normalize(TEXT1))
    print("input_len:", len(TEXT1), "output_len:", len(out1))

    out2 = reconstruct(TEXT2)
    print("\n\nTEXT 2:\n")
    print(out2)
    print("\n--- quick check ---")
    print("same_as_input:", out2.strip() == normalize(TEXT2))
    print("input_len:", len(TEXT2), "output_len:", len(out2))


if __name__ == "__main__":
    main()
