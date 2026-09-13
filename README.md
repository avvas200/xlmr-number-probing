# Cross-Lingual Probing of Grammatical Number in XLM-R

A probing study investigating whether XLM-R encodes grammatical number
(singular vs. plural) in a shared, language-agnostic subspace, or whether
each language uses its own representation — and how this varies across
the model's layers.

## Research question

> Does `xlm-roberta-base` represent grammatical number in a language-agnostic
> way across typologically diverse languages? At which layer(s) is number
> information most accessible? Do probes trained on one language transfer
> to others?

## Method (planned)

- **Model:** [`xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) (12 layers, 278M params).
- **Languages:** English, Spanish, Arabic, Turkish, Chinese, Swahili — chosen for typological variety in how number is marked.
- **Data:** Universal Dependencies treebanks. Noun tokens with `Number=Sing` or `Number=Plur` in their morphological features.
- **Probes:** Linear classifiers (logistic regression) trained on frozen hidden states, one per (language, layer) pair.
- **Controls:** Selectivity via random-label control tasks (Hewitt & Liang, 2019).
- **Cross-lingual analysis:** Train a probe on language X, evaluate on language Y — a full 6×6 transfer matrix per layer.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # UD treebanks (gitignored)
│   └── processed/    # extracted noun CSVs
├── src/
│   └── extract_ud.py # Week 1: extract nouns + number labels from CoNLL-U
└── results/
    └── figures/      # plots (added Week 2+)
```

## Reproducing Week 1

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download UD treebanks

Grab treebanks from [Universal Dependencies](https://universaldependencies.org/)
into `data/raw/`. For English:

```bash
mkdir -p data/raw
cd data/raw
git clone --depth 1 https://github.com/UniversalDependencies/UD_English-EWT.git
cd ../..
```

Repeat for the other languages you want. Availability and treebank size
vary — check the UD site for what's available and pick reasonably sized
treebanks for each language.

### 3. Extract nouns

```bash
python src/extract_ud.py \
    --input data/raw/UD_English-EWT/en_ewt-ud-train.conllu \
    --output data/processed/english.csv \
    --language en
```

This produces a CSV with one row per singular/plural noun, including the
full sentence and the noun's position — everything Week 2 needs to
extract hidden states. and also the balance step

## Status

🚧 **Week 1 done:** data extraction pipeline built and tested on
English UD-EWT. Next: extend to all 6 languages, then Week 2 (hidden
state extraction and baseline probes).

🚧 **Week 2 In progress:** 
data pipeline complete

## References (to read / cite)

The following are the papers this project builds on. **Verify titles,
authors, and years before citing** — this list is from memory.

- Hewitt & Liang (2019). "Designing and Interpreting Probes with Control Tasks."
- Conneau et al. (2020). "Emerging Cross-lingual Structure in Pretrained Language Models."
- Pires, Schlinger & Garrette (2019). "How multilingual is Multilingual BERT?"
- Conneau et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale." (the XLM-R paper)
- Nivre et al. Universal Dependencies. https://universaldependencies.org/

## License

MIT
