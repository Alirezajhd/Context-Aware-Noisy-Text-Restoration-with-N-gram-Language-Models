# Context-Aware Noisy Text Restoration with N-gram Language Models

A noisy-text restoration system that decides **when to edit**, **what to edit**, and **when not to edit**. A bigram language model with Kneser-Ney smoothing scores correction candidates in context, and an abstention threshold keeps the system from touching tokens it is not sure about. It runs on an ordinary CPU, with no neural models.

Built for the Text Mining & NLP course (Assignment 1), Shiraz University, Spring 2026.

## Why

A spell checker looks at one word at a time. Noisy text is harder: the same corrupted token can have several valid corrections (`form` -> `from` / `for`), a token can carry several noise types at once (`Wrold` = transposition + case error), and some corruptions produce another valid word (`tern` instead of `term`). Context is the only way to choose between candidates, and the safest decision is often to leave the token alone. This project uses N-gram probabilities as the central scoring signal and treats over-correction as worse than a missed error.

## How it works

```
Noisy sentence
   -> tokenize (whitespace, punctuation split off, lowercase for the LM)
   -> suspiciousness check (OOV / rare / odd character patterns)
   -> candidate generation (case variants, ED1 + keyboard, confusion rules, repetition fix, word split)
   -> candidate scoring  (bigram context + unigram prior + edit penalty)
   -> abstain unless  Score(best) - Score(original) > tau
   -> restore original casing -> output + trace
```

**Language model.** Bigram, Kneser-Ney smoothing (d = 0.75), `<s>` / `</s>` boundary markers, everything lowercased in the LM and the original casing reapplied at output. UNK threshold is 1, so the vocabulary has 87,707 types.

**Candidate generation.** The original token is always a candidate. Further candidates come from case variants, edit distance 1 (QWERTY substitutions, deletions, insertions, transpositions), OCR-style confusion rules, repeated-character normalization and word splitting. Edit distance 2 is implemented but disabled in the final system because it hurt precision and F1. Candidates are generated only for tokens whose suspiciousness score is at least 0.3.

**Scoring.**

```
Score(c) = alpha * [ log P(c | w_{i-1}) + log P(w_{i+1} | c) ]
         + beta  * log P(c)
         + gamma * ( -lambda * EditDistance(c, original) )
```

Final weights: alpha = 0.50, beta = 0.20, gamma = 0.30, lambda = 0.10.

**Abstention.** A token is changed only if `Score(best) - Score(original) > tau`, with tau = 0.3 chosen on the dev set. Trading a little F1 for higher precision was a deliberate choice.

## Results

Dev set (2,500 sentences):

| Metric | No-change | Unigram baseline | Final system |
|---|---|---|---|
| Sentence accuracy | 0.1756 | 0.3572 | **0.5316** |
| Token accuracy | 0.8783 | 0.8858 | **0.9190** |
| WER | 0.0914 | 0.0853 | **0.0504** |
| CER | 0.0149 | 0.0184 | **0.0084** |
| Edit precision | 1.0000 | 0.6420 | 0.9313 |
| Edit recall | 0.0000 | 0.3292 | 0.3476 |
| Edit F1 | 0.0000 | 0.4352 | **0.5062** |
| Over-correction rate | 0.0000 | 0.3580 | 0.0687 |

Language-model perplexity on the dev set:

| Model | Smoothing | Perplexity |
|---|---|---|
| Bigram | add-k (k = 0.0045) | 335.81 |
| Bigram | Kneser-Ney | 466.65 |
| Trigram | Kneser-Ney | 738.44 |
| Trigram | add-k (k = 0.01) | 1372.18 |

Ablations (dev, change one thing versus the final system):

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| Final system | 0.9313 | 0.3476 | 0.5062 |
| No abstention (tau = 0) | 0.9177 | 0.3527 | 0.5096 |
| No edit penalty (gamma = 0) | 0.9267 | 0.3491 | 0.5072 |
| With edit distance 2 | 0.9064 | 0.3372 | 0.4916 |
| Trigram instead of bigram | 0.9341 | 0.3341 | 0.4922 |

Notes:

- Add-k gives lower perplexity, but Kneser-Ney gave the better restoration F1, so it was used in the final system.
- Trigrams are hurt by data sparsity: higher perplexity and lower recall than bigrams.
- Of the missed errors, about 47% never had the correct candidate generated, about 29% fell below the threshold, and about 24% were outranked by a wrong candidate.

## Installation

```bash
# Python 3.x; list any third-party packages you actually import here
pip install -r requirements.txt   # if you have one
```

## Data

The course-supplied data is not included in this repository. Arrange it like this:

```
HW1_data/
├── train_clean.txt     # clean text for N-gram training (55,000 sentences)
├── dev_clean.txt       # clean dev references (2,500 sentences)
├── dev_noisy.txt       # noisy dev set
└── test_noisy.txt      # blind test set (2,500 sentences)
```

All files are UTF-8, one sentence per line.

## Usage

```bash
python system.py \
  --train HW1_data/train_clean.txt \
  --dev-noisy HW1_data/dev_noisy.txt \
  --dev-clean HW1_data/dev_clean.txt \
  --test-noisy HW1_data/test_noisy.txt \
  --outdir results \
  --config config.json
```

Optional arguments: `--seed 42`, `--ngram-order`, `--threshold`.

## Output files

Everything is written to `--outdir`:

| File | Contents |
|---|---|
| `dev_predictions.csv` | `sentence_id, input_text, predicted_text, reference_text, edited` |
| `test_predictions.csv` | Same, without `reference_text` |
| `dev_metrics.json` | Sentence/token accuracy, WER, CER, edit precision/recall/F1, over-correction rate |
| `summary.json` | Final configuration (N-gram order, smoothing, UNK threshold, weights, seed) |
| `trace.jsonl` | Per-sentence decision log: candidates, scores, margins, edit/abstain reasons |
| `report.pdf` | Written report |
| `extra/` | Optional plots and ablation logs |

## Repository layout

```
system.py          # entry point: CLI + pipeline
ngram_model.py     # counting, Kneser-Ney / add-k smoothing, probabilities, perplexity
candidates.py      # candidate generation
scorer.py          # candidate scoring and abstention
evaluator.py       # metrics and output writers
config.json        # hyperparameters
```

## Known limitations

- **Real-word errors.** In-vocabulary tokens that are wrong in context (`tern` for `term`, `n` for `in`) take the fast path and are never examined.
- **Merged words.** `wersRoman` -> `were Roman` is missed because word-split candidates are only generated for some tokens.
- **Frequency bias.** Among valid candidates, a more frequent word can win over the true one (`hoisted` -> `hosted`, `lofe` -> `life` instead of `love`).
- **Punctuation.** Apostrophes and other punctuation are sometimes dropped during candidate generation (`governments'` -> `governments`).
- **Limited recall (0.35).** Mostly caused by gaps in candidate coverage, not by the threshold.

## Acknowledgements

Course: Text Mining & NLP, Dr. Mostafa Fakhrahmad, Shiraz University.
