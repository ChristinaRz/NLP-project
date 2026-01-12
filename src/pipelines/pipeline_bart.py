"""
Παραδοτέο 1Β
Pipeline 3 — Neural rewriting με BART (facebook/bart-large-cnn).

Το pipeline:
- λαμβάνει ολόκληρο το κείμενο
- παράγει νέο αναδομημένο κείμενο με έντονη αναδόμηση
- αφαιρέί λεπτομέρειες/ συμπυκνώσει πληροφορία

"""

from transformers import pipeline
from src.data.texts import TEXT1, TEXT2

# cache
_REWRITER = None


def load_bart():

    # summarization pipeline με BART large CNN.
    return pipeline(
        task="summarization",
        model="facebook/bart-large-cnn"
    )


def reconstruct(text: str) -> str:
    #εδώ το bart λειτουργέι ως rewriter
    global _REWRITER
    if _REWRITER is None:
        _REWRITER = load_bart()

    #ρυθμίσεις μήκους
    out = _REWRITER(
        text,
        max_length=160,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

    return out.strip()


if __name__ == "__main__":
    print("TEXT 1:\n")
    print(reconstruct(TEXT1))

    print("\n\nTEXT 2:\n")
    print(reconstruct(TEXT2))
