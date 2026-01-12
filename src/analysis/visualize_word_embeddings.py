"""
Παραδοτέο 2
Word Embeddings + PCA / t-SNE

Το script:
- Μαζεύει original + reconstructed texts
- Tokenize
- Εκπαιδεύει Word2Vec (custom embeddings)
- Παίρνει κοινές/συχνές λέξεις για οπτικοποίηση
- Τρέχει PCA & t-SNE
- Επιστρέφει εικόνες png + text με αποτελέσματα

Εκτέλεση (από root):
python -m src.analysis.visualize_word_embeddings
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data.texts import TEXT1, TEXT2

# import απο reconstruct των pipelines
from src.pipelines.pipeline_languagetool import reconstruct as lt_reconstruct
from src.pipelines.pipeline_t5 import reconstruct as t5_reconstruct
from src.pipelines.pipeline_bart import reconstruct as bart_reconstruct


"""


#Παίρνουμε τις πιο συχνές λέξεις για visualization.
def pick_words(tokenized_docs: list[list[str]], top_n: int = 30) -> list[str]:

    freq = Counter([t for doc in tokenized_docs for t in doc])
    # κόβουμε μικρές/αχρειαστες λέξεις
    blacklist = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "we", "i", "it", "is", "are", "was", "were"}
    words = [w for w, _ in freq.most_common(top_n * 2) if w not in blacklist]
    return words[:top_n]


def plot_2d(points: np.ndarray, labels: list[str], title: str, out_path: str):
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1])

    #labels
    for i, w in enumerate(labels):
        plt.annotate(w, (points[i, 0], points[i, 1]), fontsize=9, alpha=0.9)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    corpus = collect_corpus()
    tokenized_docs = [tokenize(t) for t in corpus.values()]

    w2v = train_word2vec(tokenized_docs)

    words = pick_words(tokenized_docs, top_n=30)
    vectors = np.array([w2v.wv[w] for w in words])

    # PCA (2D)
    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(vectors)
    plot_2d(
        pca_points,
        words,
        "Word Embeddings (Custom Word2Vec) - PCA (2D)",
        "outputs/word2vec_pca.png"
    )

    # t-SNE (2D) - θέλει προσοχή, βάζουμε fixed random_state giati?
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    tsne_points = tsne.fit_transform(vectors)
    plot_2d(
        tsne_points,
        words,
        "Word Embeddings (Custom Word2Vec) - t-SNE (2D)",
        "outputs/word2vec_tsne.png"
    )

    print("Saved plots:")
    print("- outputs/word2vec_pca.png")
    print("- outputs/word2vec_tsne.png")


if __name__ == "__main__":
    main()
"""
def tokenize(text: str) -> List[str]:
    #lowercase το κείμενο και κραταμε τις ακολουθίες a-z
    text = text.lower()
    return re.findall(r"[a-z]+", text)


def collect_corpus() -> Dict[str, str]:
    originals = {"TEXT1": TEXT1, "TEXT2": TEXT2}

    pipelines = {
        "ORIGINAL": lambda x: x,
        "LanguageTool": lt_reconstruct,
        "T5": t5_reconstruct,
        "BART": bart_reconstruct,
    }

# dict με όλα τα κείμενα που θα χρησιμοποιηθούν ως corpus
    corpus: Dict[str, str] = {}
    for text_id, original_text in originals.items():
        for pipe_name, fn in pipelines.items():
            # key
            key = f"{text_id}_{pipe_name}"
            corpus[key] = fn(original_text)
    return corpus


def train_word2vec(tokenized_docs: List[List[str]]) -> Word2Vec:

    #Word2Vec πάνω στο υπάρχον corpus
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,          # skip-gram
        workers=1,
        seed=42,       # reproducibility
        epochs=50

    )
    return model

#Παίρνουμε τις πιο συχνές λέξεις για visualization.
def pick_words(tokenized_docs: List[List[str]], top_n: int = 30) -> List[str]:

    freq = Counter(token for doc in tokenized_docs for token in doc)

#κόβουμε μικρές/αχρείαστες λέξεις
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "we", "i", "it",
        "is", "are", "was", "were", "be", "been", "as", "at", "by", "from", "that",
        "this", "with", "our", "you", "your", "they", "he", "she", "him", "her", "us"
    }

    candidates = []
    for w, _ in freq.most_common(top_n * 4):
        if w in stopwords:
            continue
        if len(w) <= 2:
            continue
        candidates.append(w)
        if len(candidates) == top_n:
            break

    return candidates

    # 2D  με labels.
def plot_2d(points: np.ndarray, labels: List[str], title: str, out_path: str) -> None:


    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1])

    for i, w in enumerate(labels):
        plt.annotate(w, (points[i, 0], points[i, 1]), fontsize=9, alpha=0.9)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    #  corpus
    corpus = collect_corpus()

    # tokenization
    tokenized_docs = [tokenize(text) for text in corpus.values()]

    #  εκπαίδευση Word2Vec
    w2v = train_word2vec(tokenized_docs)

    # οπτικοποίηση
    words = pick_words(tokenized_docs, top_n=30)

    # κρατάμε λέξεις που όντως υπάρχουν στο vocab του Word2Vec
    words = [w for w in words if w in w2v.wv]
    if len(words) < 5:
        raise ValueError("Βγήκαν πολύ λίγες λέξεις για plotting. Δοκίμασε μεγαλύτερο top_n ή άλλα stopwords.")

    vectors = np.array([w2v.wv[w] for w in words])

    # Αποθήκευση output
    with open("outputs/word2vec_words_used.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(words))

    # PCA σε 2D
    # PCA είναι ντετερμινιστικό, αλλά δίνουμε random_state για επαναληψιμότητα
    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(vectors)
    plot_2d(
        pca_points,
        words,
        "Word Embeddings (Custom Word2Vec) - PCA (2D)",
        "outputs/word2vec_pca.png",
    )

    # t-SNE σε 2D
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        init="pca",
        learning_rate="auto",
        #ίδιο plot σε κάθε εκτέλεση
        random_state=42
    )
    tsne_points = tsne.fit_transform(vectors)
    plot_2d(
        tsne_points,
        words,
        "Word Embeddings (Custom Word2Vec) - t-SNE (2D)",
        "outputs/word2vec_tsne.png",
    )

    print("Saved outputs:")
    print("- outputs/word2vec_pca.png")
    print("- outputs/word2vec_tsne.png")
    print("- outputs/word2vec_words_used.txt")


if __name__ == "__main__":
    main()
