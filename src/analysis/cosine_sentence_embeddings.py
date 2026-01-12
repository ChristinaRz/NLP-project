"""
Παραδοτέο 2
Sentence embeddings + cosine similarity

Το script:
- Παίρνει τα original texts (TEXT1, TEXT2)
- Παίρνει reconstructed texts από τα 3 pipelines
- Υπολογίζει sentence embeddings (SBERT)
- Υπολογίζει cosine similarity original vs reconstructed
- Βγάζει πίνακα αποτελεσμάτων (pandas)

Εκτέλεση(από root):
python -m src.analysis.cosine_sentence_embeddings
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.data.texts import TEXT1, TEXT2

# import απο reconstruct των pipelines
from src.pipelines.pipeline_languagetool import reconstruct as lt_reconstruct
from src.pipelines.pipeline_t5 import reconstruct as t5_reconstruct
from src.pipelines.pipeline_bart import reconstruct as bart_reconstruct


def embed_texts(model: SentenceTransformer, texts: list[str]):
    # normalize embeddings
    return model.encode(texts, normalize_embeddings=True)


def cosine(a_vec, b_vec) -> float:
    # 2D arrays για cosine similarity
    return float(cosine_similarity([a_vec], [b_vec])[0][0])


def main():
    # sentence embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    #ground truth inputs
    originals = {
        "TEXT1": TEXT1,
        "TEXT2": TEXT2,
    }

    pipelines = {
        "LanguageTool": lt_reconstruct,
        "T5 (flan-t5-base)": t5_reconstruct,
        "BART (bart-large-cnn)": bart_reconstruct,
    }

    rows = []

    for text_id, original_text in originals.items():
        for pipe_name, fn in pipelines.items():
            reconstructed = fn(original_text)

            # embeddings
            orig_vec, rec_vec = embed_texts(model, [original_text, reconstructed])

            # cosine similarity
            score = cosine(orig_vec, rec_vec)

            rows.append({
                "Text": text_id,
                "Pipeline": pipe_name,
                "CosineSimilarity": round(score, 4),
                "OriginalLengthChars": len(original_text),
                "ReconstructedLengthChars": len(reconstructed),
            })
#αποτέλεσμα
    df = pd.DataFrame(rows).sort_values(["Text", "CosineSimilarity"], ascending=[True, False])

    print("\n=== Cosine Similarity (Sentence Embeddings) ===\n")
    print(df.to_string(index=False))

    # αποθηκέυει τα αποτελέσματα στον φάκελο outputs
    out_path = "outputs/cosine_sentence_results.csv"
    import os
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
