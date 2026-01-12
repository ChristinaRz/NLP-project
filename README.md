# NLP-project


This repository contains an NLP project focused on **text reconstruction** and **semantic similarity analysis**, implemented as part of an academic assignment.

The project is structured in three main parts (Deliverables), along with an optional bonus task.

---

## Project Structure

NLP-project/
│
├── src/
│ ├── data/
│ │ └── texts.py # Input texts (TEXT1, TEXT2)
│ │
│ ├── pipelines/
│ │ ├── pipeline_languagetool.py # Grammar-based reconstruction
│ │ ├── pipeline_t5.py # Neural rewriting with Flan-T5
│ │ └── pipeline_bart.py # Neural rewriting with BART
│ │
│ ├── analysis/
│ │ ├── cosine_sentence_embeddings.py # Sentence embeddings + cosine similarity
│ │ └── visualize_word_embeddings.py # Word2Vec + PCA / t-SNE
│ │
│ └── bonus/
│ └── masked_clause.py # Bonus: masked clause prediction (Greek legal text)
│
├── outputs/ # Generated results (CSV, PNG)
│ └── (ignored by git)
│
├── README.md
└── .gitignore

All scripts are executed from the project root using python -m

The outputs/ folder is excluded from version control

Results are reproducible (fixed seeds where applicable)

Dependencies
Indicative libraries used:

transformers

torch

sentence-transformers

scikit-learn

gensim

language-tool-python

stanza

pandas

matplotlib
