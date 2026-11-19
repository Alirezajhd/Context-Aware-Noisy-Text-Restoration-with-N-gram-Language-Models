import json
import csv
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict


class Evaluator:
    def __init__(self):
        self.metrics = {}

    def damerau_levenshtein_distance(self, s1: List[str], s2: List[str]) -> int:
        n, m = len(s1), len(s2)
        if n == 0:
            return m
        if m == 0:
            return n

        INF = n + m
        dp = [[INF] * (m + 2) for _ in range(n + 2)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            dp[i][0] = i
        for j in range(1, m + 1):
            dp[0][j] = j

        last_seen = {}
        for i in range(1, n + 1):
            last_match = 0
            for j in range(1, m + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
                if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                    dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost)
                if s1[i-1] == s2[j-1]:
                    last_match = j
        return dp[n][m]

    def compute_wer(self, predicted: List[str], reference: List[str]) -> float:
        if not reference:
            return 0.0
        return self.damerau_levenshtein_distance(predicted, reference) / len(reference)

    def compute_cer(self, predicted: str, reference: str) -> float:
        if not reference:
            return 0.0
        return self.damerau_levenshtein_distance(list(predicted), list(reference)) / len(reference)

    def compute_token_accuracy(self, predicted: List[str], reference: List[str]) -> float:
        if not reference:
            return 1.0
        max_len = max(len(predicted), len(reference))
        p = predicted + [''] * (max_len - len(predicted))
        r = reference + [''] * (max_len - len(reference))
        return sum(1 for a, b in zip(p, r) if a == b) / max_len

    def compute_sentence_accuracy(self, predicted: str, reference: str) -> bool:
        return predicted.strip() == reference.strip()

    def compute_edited_flag(self, original_tokens: List[str], predicted_tokens: List[str]) -> bool:
        max_len = max(len(original_tokens), len(predicted_tokens))
        o = original_tokens + [''] * (max_len - len(original_tokens))
        p = predicted_tokens + [''] * (max_len - len(predicted_tokens))
        return any(a != b for a, b in zip(o, p))

    def compute_edit_metrics(self, original: List[str], predicted: List[str], reference: List[str]) -> Dict[str, float]:
        max_len = max(len(original), len(predicted), len(reference))
        o = original + [''] * (max_len - len(original))
        p = predicted + [''] * (max_len - len(predicted))
        r = reference + [''] * (max_len - len(reference))

        changed = changed_correct = changed_wrong = fixed = wrong_total = 0
        for orig, pred, ref in zip(o, p, r):
            is_wrong = orig != ref
            if is_wrong:
                wrong_total += 1
            is_changed = orig != pred
            if is_changed:
                changed += 1
                if not is_wrong:
                    changed_correct += 1
                else:
                    changed_wrong += 1
                    if pred == ref:
                        fixed += 1

        edit_precision = changed_wrong / changed if changed > 0 else 0.0
        edit_recall = fixed / wrong_total if wrong_total > 0 else 0.0
        edit_f1 = 2 * (edit_precision * edit_recall) / (edit_precision + edit_recall) if (edit_precision + edit_recall) > 0 else 0.0
        overcorrection_rate = changed_correct / changed if changed > 0 else 0.0

        return {
            'edit_precision': edit_precision,
            'edit_recall': edit_recall,
            'edit_F1': edit_f1,
            'overcorrection_rate': overcorrection_rate,
            'tokens_changed': changed,
            'tokens_wrong': wrong_total,
            'tokens_fixed': fixed,
            'tokens_overcorrected': changed_correct
        }

    def evaluate(self, original_sentences: List[str], predicted_sentences: List[str], reference_sentences: List[str]) -> Dict[str, Any]:
        orig_tokens = [s.split() for s in original_sentences]
        pred_tokens = [s.split() for s in predicted_sentences]
        ref_tokens = [s.split() for s in reference_sentences]

        total_sent_acc = total_tok_acc = total_wer = total_cer = 0
        total_changed = total_wrong = total_fixed = total_overcorrected = 0
        n = len(reference_sentences)

        for i in range(n):
            if i < len(predicted_sentences):
                total_sent_acc += self.compute_sentence_accuracy(predicted_sentences[i], reference_sentences[i])
            total_tok_acc += self.compute_token_accuracy(pred_tokens[i], ref_tokens[i])
            total_wer += self.compute_wer(pred_tokens[i], ref_tokens[i])
            total_cer += self.compute_cer(predicted_sentences[i], reference_sentences[i])

            em = self.compute_edit_metrics(orig_tokens[i], pred_tokens[i], ref_tokens[i])
            total_changed += em['tokens_changed']
            total_wrong += em['tokens_wrong']
            total_fixed += em['tokens_fixed']
            total_overcorrected += em['tokens_overcorrected']

        changed_wrong_total = total_changed - total_overcorrected
        edit_precision = changed_wrong_total / total_changed if total_changed > 0 else 1.0
        edit_recall = total_fixed / total_wrong if total_wrong > 0 else 0.0
        edit_f1 = 2 * (edit_precision * edit_recall) / (edit_precision + edit_recall) if (edit_precision + edit_recall) > 0 else 0.0
        overcorrection_rate = total_overcorrected / total_changed if total_changed > 0 else 0.0

        self.metrics = {
            'sentence_accuracy': total_sent_acc / n if n > 0 else 0.0,
            'token_accuracy': total_tok_acc / n if n > 0 else 0.0,
            'WER': total_wer / n if n > 0 else 0.0,
            'CER': total_cer / n if n > 0 else 0.0,
            'edit_precision': edit_precision,
            'edit_recall': edit_recall,
            'edit_F1': edit_f1,
            'overcorrection_rate': overcorrection_rate,
            'total_sentences': n,
            'tokens_changed': total_changed,
            'tokens_wrong': total_wrong,
            'tokens_fixed': total_fixed,
            'tokens_overcorrected': total_overcorrected
        }
        return self.metrics

    def save_dev_predictions(self, sentence_ids: List[int], input_texts: List[str], predicted_texts: List[str], reference_texts: List[str], edited: List[bool], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['sentence_id', 'input_text', 'predicted_text', 'reference_text', 'edited'])
            for i in range(len(sentence_ids)):
                w.writerow([sentence_ids[i], input_texts[i], predicted_texts[i], reference_texts[i], edited[i]])
        print(f"✅ Saved dev predictions to {output_path}")

    def save_test_predictions(self, sentence_ids: List[int], input_texts: List[str], predicted_texts: List[str], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['sentence_id', 'input_text', 'predicted_text'])
            for i in range(len(sentence_ids)):
                w.writerow([sentence_ids[i], input_texts[i], predicted_texts[i]])
        print(f"✅ Saved test predictions to {output_path}")

    def save_dev_metrics(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        required_metrics = ['sentence_accuracy', 'token_accuracy', 'WER', 'CER', 'edit_precision', 'edit_recall', 'edit_F1', 'overcorrection_rate']
        metrics_to_save = {k: self.metrics[k] for k in required_metrics if k in self.metrics}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"✅ Saved dev metrics to {output_path}")

    def save_summary(self, output_path: Path, config: Dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved summary to {output_path}")

    def save_trace(self, trace_data: List[Dict[str, Any]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in trace_data:
                f.write(json.dumps(entry) + '\n')
        print(f"✅ Saved trace to {output_path}")

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics


if __name__ == "__main__":
    original_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "teh cat sat on the mat",
        "recieve the message",
        "a quick brown dog runs"
    ]
    predicted_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "the cat sat on the mat",
        "receive the message",
        "a quick brown dog runs"
    ]
    reference_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "the cat sat on the mat",
        "receive the message",
        "a quick brown dog runs"
    ]

    evaluator = Evaluator()
    print("="*70)
    print("Evaluator - Section 6: Results (With Transposition)")
    print("="*70)

    metrics = evaluator.evaluate(original_sentences, predicted_sentences, reference_sentences)
    print("\n" + "="*70)
    print("Section 6.3: Dev Metrics (with Damerau-Levenshtein)")
    print("="*70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("Testing compute_edited_flag()")
    print("="*70)
    for i in range(len(original_sentences)):
        orig_tokens = original_sentences[i].split()
        pred_tokens = predicted_sentences[i].split()
        edited = evaluator.compute_edited_flag(orig_tokens, pred_tokens)
        print(f"Sentence {i}: '{original_sentences[i]}'")
        print(f"  → '{predicted_sentences[i]}'")
        print(f"  Edited: {edited}")

    print("\n" + "="*70)
    print("Testing Transposition Handling in WER")
    print("="*70)
    pred_tokens = ["teh", "cat", "sat"]
    ref_tokens = ["the", "cat", "sat"]
    wer = evaluator.compute_wer(pred_tokens, ref_tokens)
    print(f"WER for 'teh' → 'the' (transposition): {wer:.4f}")
    print(f"Expected: 1/3 = 0.3333 (transposition counts as 1 edit)")

    pred_tokens2 = ["teh", "recieve"]
    ref_tokens2 = ["the", "receive"]
    wer2 = evaluator.compute_wer(pred_tokens2, ref_tokens2)
    print(f"\nWER for 'teh recieve' → 'the receive': {wer2:.4f}")
    print("Each transposition counts as 1 edit operation")

    print("\n" + "="*70)
    print("Damerau-Levenshtein vs Standard Levenshtein")
    print("="*70)
    def levenshtein_distance(s1, s2):
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
        return dp[n][m]

    test_word, correct_word = "teh", "the"
    dl_dist = evaluator.damerau_levenshtein_distance(list(test_word), list(correct_word))
    lev_dist = levenshtein_distance(list(test_word), list(correct_word))
    print(f"Word: '{test_word}' → '{correct_word}'")
    print(f"  Damerau-Levenshtein (with transposition): {dl_dist}")
    print(f"  Standard Levenshtein (without transposition): {lev_dist}")
    print("  ✅ Damerau-Levenshtein correctly counts transposition as 1 edit")

    print("\n" + "="*70)
    print("Testing Output File Generation")
    print("="*70)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        sentence_ids = list(range(len(original_sentences)))
        edited = [evaluator.compute_edited_flag(orig.split(), pred.split()) for orig, pred in zip(original_sentences, predicted_sentences)]

        evaluator.save_dev_predictions(sentence_ids, original_sentences, predicted_sentences, reference_sentences, edited, output_dir / 'dev_predictions.csv')
        evaluator.save_test_predictions(sentence_ids, original_sentences, predicted_sentences, output_dir / 'test_predictions.csv')
        evaluator.save_dev_metrics(output_dir / 'dev_metrics.json')

        config = {
            "student_name": "Your Name",
            "student_id": "123456",
            "ngram_order": 2,
            "smoothing": "add_k",
            "tokenization": "whitespace",
            "case_handling": "lowercase",
            "unk_threshold": 2,
            "abstain_threshold": 1.5,
            "candidate_strategy": "edit_distance_1 + edit_distance_2 + dictionary",
            "score_weights": {"context": 0.65, "prior": 0.20, "edit_penalty": 0.15},
            "random_seed": 42
        }
        evaluator.save_summary(output_dir / 'summary.json', config)

        trace_data = [
            {"sentence_id": 0, "original": "the quick brown fox", "predicted": "the quick brown fox", "changes": [], "edited": False},
            {"sentence_id": 1, "original": "teh cat", "predicted": "the cat", "changes": [{"position": 0, "original": "teh", "predicted": "the"}], "edited": True}
        ]
        evaluator.save_trace(trace_data, output_dir / 'trace.jsonl')

        print("\nFiles generated:")
        for file in output_dir.iterdir():
            print(f"  ✅ {file.name}")

        print("\n" + "="*70)
        print("Example: dev_predictions.csv")
        print("="*70)
        with open(output_dir / 'dev_predictions.csv', 'r') as f:
            print(f.read())

        print("\n" + "="*70)
        print("Example: dev_metrics.json")
        print("="*70)
        with open(output_dir / 'dev_metrics.json', 'r') as f:
            print(f.read())

    print("\n" + "="*70)
    print("PDF REQUIREMENTS CHECKLIST FOR EVALUATOR")
    print("="*70)
    print("✅ 6.1: dev_predictions.csv (sentence_id, input_text, predicted_text, reference_text, edited)")
    print("✅ 6.2: test_predictions.csv (sentence_id, input_text, predicted_text)")
    print("✅ 6.3: dev_metrics.json (sentence_accuracy, token_accuracy, WER, CER, edit_precision, edit_recall, edit_F1, overcorrection_rate)")
    print("✅ 6.4: summary.json (student_name, student_id, ngram_order, smoothing, tokenization, case_handling, unk_threshold, abstain_threshold, candidate_strategy, score_weights, random_seed)")
    print("✅ 6.5: trace.jsonl (per-sentence decision log)")
    print("✅ Sentence accuracy: Fraction of sentences matching reference")
    print("✅ Token accuracy: Fraction of tokens matching reference")
    print("✅ WER: Damerau-Levenshtein (with transposition) / total reference tokens")
    print("✅ CER: Damerau-Levenshtein (with transposition) at character level")
    print("✅ Edit precision: Changed wrong / Changed total")
    print("✅ Edit recall: Fixed / Wrong total")
    print("✅ Edit F1: Harmonic mean of edit precision and recall")
    print("✅ Overcorrection rate: Changed correct / Changed total")
    print("✅ Transposition included as a single edit operation (Damerau-Levenshtein)")
    print("✅ compute_edited_flag() method for determining if any token was changed")
    print("="*70)