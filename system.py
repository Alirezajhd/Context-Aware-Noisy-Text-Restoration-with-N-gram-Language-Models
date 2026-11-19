"""
system.py - Noisy Text Restoration Pipeline Orchestrator
"""
import json, csv, argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set

from ngram_model import NGramModel, build_ngram_model
from candidates import CandidateGenerator
from scorer import CandidateScorer
from evaluator import Evaluator


class TextRestorationSystem:
    def __init__(
        self,
        ngram_model: NGramModel,
        vocabulary: set,
        threshold: float = 1.5,
        context_weight: float = 0.65,
        prior_weight: float = 0.20,
        edit_weight: float = 0.15,
        edit_lambda: float = 0.10,
        preserve_case: bool = True,
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        self.lm = ngram_model
        self.vocabulary = vocabulary
        self.threshold = threshold
        self.preserve_case = preserve_case
        self.verbose = verbose
        self.config = config or {}
        
        self.candidate_generator = CandidateGenerator(
            vocabulary=vocabulary,
            min_ed1_length=self.config.get('min_ed1_length', 1),
            min_ed2_length=self.config.get('min_ed2_length', 3),
            max_ed1_candidates=self.config.get('max_ed1_candidates', 50),
            max_ed2_candidates=self.config.get('max_ed2_candidates', 20),
            max_total_candidates=self.config.get('max_total_candidates', 100),
            use_keyboard=self.config.get('use_keyboard', True),
            use_transposition=self.config.get('use_transposition', True),
            use_confusion_rules=self.config.get('use_confusion_rules', True),
            suspicious_threshold=self.config.get('suspicious_threshold', 0.3),
            debug=True
        )
        
        self.scorer = CandidateScorer(
            ngram_model=ngram_model,
            context_weight=context_weight,
            prior_weight=prior_weight,
            edit_weight=edit_weight,
            edit_lambda=edit_lambda
        )
        
        self.total_tokens = 0
        self.tokens_changed = 0
        self.tokens_abstained = 0
        self.token_decisions = []
        self.sentence_traces = []
    
    def restore_sentence(self, sentence: str) -> Tuple[str, List[Dict[str, Any]]]:
        original_tokens = sentence.split()
        original_casing = [self._get_casing_pattern(t) for t in original_tokens]
        processed_tokens = []
        token_traces = []
        
        for i, token in enumerate(original_tokens):
            left_context = ' '.join(processed_tokens[-2:]) if processed_tokens else ''
            right_context = ' '.join(original_tokens[i+1:i+3]) if i + 1 < len(original_tokens) else ''
            
            unigram_freq = self._get_unigram_freq(token)
            result = self._process_token(token, left_context, right_context, i, unigram_freq)
            
            processed_tokens.append(result['output_token'])
            token_traces.append(result['trace'])
            
            self.total_tokens += 1
            if result['trace']['action'] == 'EDIT':
                self.tokens_changed += 1
            else:
                self.tokens_abstained += 1
        
        actions = [t['action'] for t in token_traces]
        if self.preserve_case:
            processed_tokens = self._restore_casing(processed_tokens, original_tokens, original_casing, actions)
        
        corrected_sentence = ' '.join(processed_tokens)
        trace_entry = self._build_trace_entry(len(self.sentence_traces), sentence, corrected_sentence, token_traces)
        self.sentence_traces.append(trace_entry)
        
        return corrected_sentence, token_traces
    
    def _get_unigram_freq(self, token: str) -> Optional[float]:
        token_lower = token.lower()
        if token_lower not in self.vocabulary:
            return None
        try:
            unigram_counts = self.lm.counts.get(tuple(), {})
            count = unigram_counts.get(token_lower, 0)
            return count / (self.lm.total_tokens or 1)
        except Exception:
            return None
    
    def _build_trace_entry(self, sentence_id: int, input_text: str, output_text: str,
                           token_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        edited = any(t['action'] == 'EDIT' for t in token_traces)
        
        total_score = total_original = total_improvement = num_scored = 0
        for trace in token_traces:
            if trace.get('best_score', 0.0) != 0.0:
                total_score += trace['best_score']
                total_original += trace.get('original_score', 0.0)
                total_improvement += trace.get('score_diff', 0.0)
                num_scored += 1
        
        avg_score = total_score / num_scored if num_scored > 0 else 0.0
        avg_original = total_original / num_scored if num_scored > 0 else 0.0
        avg_improvement = total_improvement / num_scored if num_scored > 0 else 0.0
        
        if edited:
            first_edit = next((t for t in token_traces if t['action'] == 'EDIT'), None)
            reason = f"token '{first_edit['original']}' replaced by '{first_edit['predicted']}' with margin {first_edit['score_diff']:.2f} > threshold {first_edit['threshold']}" if first_edit else "edit made"
        else:
            reason = "no token had margin above threshold"
        
        token_decisions = []
        for trace in token_traces:
            candidates = []
            if trace.get('candidates'):
                for rank, cand in enumerate(trace['candidates'], 1):
                    score = self._get_candidate_score(trace, cand)
                    candidates.append({"text": cand, "score": score, "rank": rank})
            else:
                candidates.append({"text": trace['original'], "score": trace.get('original_score', 0.0), "rank": 1})
            
            selected_score = trace.get('best_score', 0.0) if trace['action'] == 'EDIT' else trace.get('original_score', 0.0)
            token_decisions.append({
                "token_index": trace['position'],
                "original": trace['original'],
                "final": trace['predicted'] if trace['action'] == 'EDIT' else trace['original'],
                "changed": trace['action'] == 'EDIT',
                "candidates": candidates,
                "selected_candidate": trace['predicted'] if trace['action'] == 'EDIT' else trace['original'],
                "score_original": trace.get('original_score', 0.0),
                "score_selected": selected_score,
                "score_margin": trace.get('score_diff', 0.0),
                "threshold": trace.get('threshold', self.threshold),
                "decision": trace['action'].lower(),
                "decision_reason": self._build_decision_reason(trace)
            })
        
        return {
            "sentence_id": sentence_id,
            "input_text": input_text,
            "output_text": output_text,
            "edited": edited,
            "sentence_level": {
                "input_score": avg_original,
                "output_score": avg_score,
                "score_improvement": avg_improvement,
                "decision": "edit" if edited else "abstain",
                "reason": reason
            },
            "token_decisions": token_decisions,
            "summary": {
                "positions_considered": len(token_traces),
                "edits_made": sum(1 for t in token_traces if t['action'] == 'EDIT'),
                "abstentions": sum(1 for t in token_traces if t['action'] == 'ABSTAIN')
            }
        }
    
    def _get_candidate_score(self, trace: Dict, cand: str) -> float:
        if cand == trace.get('best_candidate', '') and 'best_score' in trace:
            return trace['best_score']
        if cand == trace['original'] and 'original_score' in trace:
            return trace['original_score']
        return 0.0
    
    def _build_decision_reason(self, trace: Dict) -> str:
        if trace['action'] == 'EDIT':
            return f"margin {trace['score_diff']:.2f} exceeds threshold {trace['threshold']}; best candidate '{trace['best_candidate']}' preferred"
        reasons = {
            'in_vocabulary_fast_path': "original token is in vocabulary and in-context score not suspicious",
            'in_vocabulary_no_alternatives': "original token is in vocabulary and no viable alternatives found",
            'score_diff_below_threshold': f"margin {trace['score_diff']:.2f} below threshold {trace['threshold']}; uncertainty favours original"
        }
        return reasons.get(trace.get('reason', ''), "original token retained due to insufficient evidence for change")
    
    # ==================== ADAPTIVE THRESHOLD ====================
    _ADAPTIVE_BASE = {'len_le_2': 1.0, 'len_eq_3': 0.5, 'len_eq_4': 0.25, 'digit': 1.5, 'common_word': 0.5, 'oov_bonus': -0.5}
    
    def _word_frequency(self, word: str) -> float:
        try:
            counts = self.lm.counts.get(tuple(), {})
            return counts.get(word, 0) / (self.lm.total_tokens or 1)
        except Exception:
            return 0.0
    
    def _compute_adaptive_threshold(self, token: str, is_oov: bool) -> float:
        if not self.config.get('enable_adaptive_threshold', False):
            return self.threshold
        
        scale = self.config.get('adaptive_scale', 1.0)
        adj = 0.0
        length = len(token)
        
        if length <= 2:
            adj += 1.0 * scale
        elif length == 3:
            adj += 0.5 * scale
        elif length == 4:
            adj += 0.25 * scale
        
        if any(c.isdigit() for c in token):
            adj += 1.5 * scale
        if is_oov:
            adj -= 0.5 * scale
        elif self._word_frequency(token.lower()) > self.config.get('adaptive_common_word_freq', 0.0008):
            adj += 0.5 * scale
        
        return max(self.config.get('adaptive_min_threshold', 0.2), self.threshold + adj)
    
    def _process_token(self, token: str, left: str, right: str, pos: int, unigram_freq: Optional[float] = None) -> Dict:
        token_lower = token.lower()
        is_oov = not self.candidate_generator.is_in_vocabulary(token_lower)
        
        # In-vocabulary token handling
        if not is_oov:
            if not self.candidate_generator.should_generate_candidates(token, unigram_freq):
                return self._make_abstain(token, pos, "in_vocabulary_fast_path")
            
            ctx_score = self.scorer.compute_context_score(token_lower, left, right)
            gate = self.config.get('real_word_context_gate', -15.0)
            min_len = self.config.get('real_word_min_length', 3)
            
            if not (len(token_lower) >= min_len and ctx_score < gate):
                return self._make_abstain(token, pos, "in_vocabulary_fast_path")
            
            # Generate and score candidates for real-word errors
            candidates = set(self.candidate_generator.generate_ed1_candidates(token_lower))
            candidates.update(self.candidate_generator.apply_confusion_rules(token))
            candidates.update(self.candidate_generator.generate_case_variants(token))
            candidates.add(token)
            
            # Prior guard
            prior_guard = self.config.get('real_word_prior_guard', 0.5)
            orig_prior = self.scorer.compute_prior_score(token_lower)
            candidates = [c for c in candidates if c == token_lower or self.scorer.compute_prior_score(c) >= orig_prior - prior_guard]
            if token not in candidates:
                candidates.append(token)
            
            if len(candidates) <= 1:
                return self._make_abstain(token, pos, "in_vocabulary_no_alternatives")
            
            scored = self.scorer.score_candidates(candidates, token, left, right)
            best, best_score = scored[0]
            orig_score = self.scorer.score_candidate(token, token, left, right)
            diff = best_score - orig_score
            
            real_threshold = self.config.get('real_word_abstain_threshold', 2.0)
            if diff > real_threshold and best != token_lower:
                return self._make_edit(best, token, pos, diff, real_threshold, scored, False, "real_word_or_case_error_corrected")
            return self._make_abstain(token, pos, "real_word_check_below_threshold", orig_score, diff, real_threshold)
        
        # OOV token: full candidate generation
        candidates = self.candidate_generator.generate_candidates(
            word=token, is_oov=True, unigram_frequency=unigram_freq,
            include_case_variants=self.config.get('include_case_variants', True),
            include_confusion_rules=self.config.get('include_confusion_rules', True),
            include_ed1=self.config.get('include_ed1', True),
            include_ed2=self.config.get('include_ed2', True)
        )
        if token not in candidates:
            candidates = [token] + candidates
        
        scored = self.scorer.score_candidates(candidates, token, left, right)
        best, best_score = scored[0] if scored else (token, float('-inf'))
        orig_score = self.scorer.score_candidate(token, token, left, right)
        diff = best_score - orig_score
        threshold = self._compute_adaptive_threshold(token, True)
        
        if self.verbose and is_oov:
            print(f"\n  🔍 Token: '{token}'")
            print(f"  🏆 Best: '{best}' (score: {best_score:.4f})")
            print(f"  📈 Score diff: {diff:.4f} vs threshold: {threshold}")
        
        if diff > threshold:
            return self._make_edit(best, token, pos, diff, threshold, scored, True, "score_diff_exceeds_threshold")
        return self._make_abstain(token, pos, "score_diff_below_threshold", orig_score, diff, threshold)
    
    def _make_abstain(self, token: str, pos: int, reason: str, orig_score: float = 0.0, diff: float = 0.0, threshold: float = None) -> Dict:
        return {
            'output_token': token,
            'trace': {
                'position': pos, 'original': token, 'predicted': token,
                'action': 'ABSTAIN', 'reason': reason, 'is_oov': False,
                'candidates': [], 'best_candidate': token, 'best_score': 0.0,
                'original_score': orig_score, 'score_diff': diff,
                'threshold': threshold or self.threshold
            }
        }
    
    def _make_edit(self, best: str, token: str, pos: int, diff: float, threshold: float,
                   scored: List, is_oov: bool, reason: str) -> Dict:
        return {
            'output_token': best,
            'trace': {
                'position': pos, 'original': token, 'predicted': best,
                'action': 'EDIT', 'reason': reason, 'is_oov': is_oov,
                'candidates': [c for c, _ in scored[:10]],
                'best_candidate': best, 'best_score': scored[0][1] if scored else 0.0,
                'original_score': self.scorer.score_candidate(token, token, '', ''),
                'score_diff': diff, 'threshold': threshold
            }
        }
    
    # ==================== CASE HANDLING ====================
    def _get_casing_pattern(self, token: str) -> str:
        if token.isupper(): return 'upper'
        if token.istitle(): return 'title'
        if token.islower(): return 'lower'
        return 'mixed'
    
    def _restore_casing(self, processed: List[str], original: List[str], casing: List[str], actions: Optional[List[str]] = None) -> List[str]:
        restored = []
        for i, (token, case) in enumerate(zip(processed, casing)):
            action = actions[i] if actions else None
            if action == 'ABSTAIN' or (action is None and token == original[i]):
                restored.append(original[i])
            elif case == 'upper':
                restored.append(token.upper())
            elif case == 'title':
                restored.append(token.capitalize())
            elif case == 'lower':
                restored.append(token.lower())
            else:
                restored.append(token)
        return restored
    
    # ==================== BATCH ====================
    def restore_sentences(self, sentences: List[str], show_progress: bool = True) -> Tuple[List[str], List[List[Dict]]]:
        corrected, traces = [], []
        total = len(sentences)
        for i, sent in enumerate(sentences):
            c, t = self.restore_sentence(sent)
            corrected.append(c)
            traces.append(t)
            if show_progress and total > 100 and (i + 1) % max(1, total // 10) == 0:
                print(f"   Processing sentence {i+1}/{total}")
        return corrected, traces
    
    # ==================== STATISTICS ====================
    def get_statistics(self) -> Dict:
        return {
            'total_tokens': self.total_tokens,
            'tokens_changed': self.tokens_changed,
            'tokens_abstained': self.tokens_abstained,
            'change_rate': self.tokens_changed / self.total_tokens if self.total_tokens > 0 else 0.0,
            'abstain_rate': self.tokens_abstained / self.total_tokens if self.total_tokens > 0 else 0.0,
            'threshold': self.threshold,
            'decisions': self.token_decisions
        }
    
    def reset_counters(self):
        self.total_tokens = 0
        self.tokens_changed = 0
        self.tokens_abstained = 0
        self.token_decisions = []
    
    def reset_statistics(self):
        self.reset_counters()
        self.sentence_traces = []
    
    # ==================== THRESHOLD TUNING ====================
    def tune_threshold(self, sentences: List[str], reference: List[str],
                       thresholds: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5]) -> Tuple[float, Dict]:
        evaluator = Evaluator()
        results = {}
        print("\n" + "="*70)
        print("Section 4.4: Threshold Tuning via Grid Search")
        print("="*70)
        print(f"{'Threshold':<12} {'Edits Made':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*60)
        
        for t in thresholds:
            old = self.threshold
            self.threshold = t
            self.reset_statistics()
            preds, _ = self.restore_sentences(sentences, False)
            metrics = evaluator.evaluate(sentences, preds, reference)
            results[t] = {k: metrics[k] for k in ['tokens_changed', 'edit_precision', 'edit_recall', 'edit_F1']}
            print(f"{t:<12.1f} {metrics['tokens_changed']:<12} {metrics['edit_precision']:<12.4f} {metrics['edit_recall']:<12.4f} {metrics['edit_F1']:<12.4f}")
            self.threshold = old
        
        best = max(results, key=lambda x: results[x]['edit_F1'])
        print("-"*60)
        print(f"Best threshold: {best} (F1 = {results[best]['edit_F1']:.4f})")
        print("="*70)
        return best, results
    
    # ==================== OUTPUT ====================
    def save_dev_predictions(self, ids: List[int], inputs: List[str], preds: List[str],
                             refs: List[str], edited: List[bool], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['sentence_id', 'input_text', 'predicted_text', 'reference_text', 'edited'])
            for i in range(len(ids)):
                w.writerow([ids[i], inputs[i], preds[i], refs[i], edited[i]])
    
    def save_test_predictions(self, ids: List[int], inputs: List[str], preds: List[str], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['sentence_id', 'input_text', 'predicted_text'])
            for i in range(len(ids)):
                w.writerow([ids[i], inputs[i], preds[i]])
    
    def save_dev_metrics(self, metrics: Dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ['sentence_accuracy', 'token_accuracy', 'WER', 'CER', 'edit_precision', 'edit_recall', 'edit_F1', 'overcorrection_rate']
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({k: metrics[k] for k in keys if k in metrics}, f, indent=2)
    
    def save_summary(self, config: Dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def save_trace(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for entry in self.sentence_traces:
                f.write(json.dumps(entry) + '\n')


# ==================== FACTORY FUNCTIONS ====================
def load_extra_wordlist(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        print(f"  ⚠ extra_wordlist '{path}' not found")
        return set()
    words = set()
    with open(p, encoding='utf-8', errors='ignore') as f:
        for line in f:
            w = line.strip().lower()
            if w and w.isalpha():
                words.add(w)
    print(f"  ✓ Loaded {len(words)} words from '{path}'")
    return words


def build_restoration_system(
    train_sentences: List[str],
    vocabulary: Optional[set] = None,
    ngram_order: int = 2,
    smoothing_k: float = 0.01,
    unk_threshold: int = 2,
    threshold: float = 1.5,
    context_weight: float = 0.65,
    prior_weight: float = 0.20,
    edit_weight: float = 0.15,
    edit_lambda: float = 0.10,
    lowercase: bool = True,
    use_boundaries: bool = True,
    smoothing: str = 'add_k',
    preserve_case: bool = True,
    verbose: bool = False,
    config: Optional[Dict] = None,
    extra_wordlist_path: Optional[str] = None
) -> TextRestorationSystem:
    print(f"\nTraining {ngram_order}-gram model with {smoothing} smoothing...")
    lm = build_ngram_model(train_sentences, ngram_order, smoothing_k, unk_threshold, lowercase, use_boundaries, smoothing)
    
    if vocabulary is None:
        vocabulary = lm.get_vocab()
        print(f"Vocabulary size: {len(vocabulary)}")
        extra = load_extra_wordlist(extra_wordlist_path)
        if extra:
            vocabulary = set(vocabulary) | extra
    
    cfg = config or {}
    cfg.setdefault('min_ed1_length', 1)
    cfg.setdefault('min_ed2_length', 3)
    cfg.setdefault('max_ed1_candidates', 50)
    cfg.setdefault('max_ed2_candidates', 20)
    cfg.setdefault('max_total_candidates', 100)
    cfg.setdefault('use_keyboard', True)
    cfg.setdefault('use_transposition', True)
    cfg.setdefault('use_confusion_rules', True)
    cfg.setdefault('include_case_variants', True)
    cfg.setdefault('include_ed1', True)
    cfg.setdefault('include_ed2', True)
    cfg.setdefault('suspicious_threshold', 0.3)
    
    return TextRestorationSystem(lm, vocabulary, threshold, context_weight, prior_weight,
                                 edit_weight, edit_lambda, preserve_case, verbose, cfg)


def load_data(path: Path) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# ==================== BASELINES ====================
def no_change_baseline(sentences: List[str]) -> List[str]:
    return sentences[:]


def unigram_baseline(sentences: List[str], unigram_probs: Dict[str, float]) -> List[str]:
    def edit_distance_1(word: str) -> List[str]:
        letters = 'abcdefghijklmnopqrstuvwxyz'
        candidates = set()
        for i in range(len(word) + 1):
            left, right = word[:i], word[i:]
            if right:
                candidates.add(left + right[1:])
            for c in letters:
                candidates.add(left + c + right)
                if right:
                    candidates.add(left + c + right[1:])
            if len(right) >= 2:
                candidates.add(left + right[1] + right[0] + right[2:])
        return list(candidates)
    
    vocab_set = set(unigram_probs.keys())
    corrected = []
    changed = total = 0
    for sent in sentences:
        tokens = sent.split()
        new_tokens = []
        for token in tokens:
            total += 1
            lower = token.lower()
            if lower in vocab_set:
                new_tokens.append(token)
            else:
                candidates = [c for c in edit_distance_1(lower) if c in vocab_set]
                if candidates:
                    best = max(candidates, key=lambda c: unigram_probs.get(c, 0.0))
                    new_tokens.append(best)
                    changed += 1
                else:
                    new_tokens.append(token)
        corrected.append(' '.join(new_tokens))
    print(f"  Unigram baseline changed {changed}/{total} tokens ({changed/total*100:.2f}%)")
    return corrected


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--train', help='Training data')
    parser.add_argument('--dev-noisy', help='Dev noisy')
    parser.add_argument('--dev-clean', help='Dev clean')
    parser.add_argument('--test-noisy', help='Test noisy')
    parser.add_argument('--outdir', default='results/')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ngram-order', type=int)
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()
    
    print("="*70)
    print("NLP Noisy Text Restoration System")
    print("="*70)
    
    try:
        with open(args.config) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config not found, using defaults")
        config = {}
    
    train_path = args.train or config.get('data', {}).get('train_data', 'train.txt')
    dev_noisy_path = args.dev_noisy or config.get('data', {}).get('dev_data', 'dev.txt')
    dev_clean_path = args.dev_clean or config.get('data', {}).get('dev_clean', 'dev_clean.txt')
    test_noisy_path = args.test_noisy or config.get('data', {}).get('test_data', 'test.txt')
    out_dir = Path(args.outdir)
    
    print("Loading data...")
    train = load_data(Path(train_path))
    dev_clean = load_data(Path(dev_clean_path))
    dev_noisy = load_data(Path(dev_noisy_path))
    test_noisy = load_data(Path(test_noisy_path))
    
    if args.quick:
        train, dev_clean, dev_noisy, test_noisy = train[:5000], dev_clean[:500], dev_noisy[:500], test_noisy[:500]
        print("⚠️ Quick mode")
    
    print(f"Loaded {len(train)} train, {len(dev_clean)} dev-clean, {len(dev_noisy)} dev-noisy, {len(test_noisy)} test")
    
    lm_cfg = config.get('language_model', {})
    score_cfg = config.get('scoring', {})
    sys_cfg = config.get('system', {})
    cand_cfg = config.get('candidate_generation', {})
    
    ngram_order = args.ngram_order or lm_cfg.get('ngram_order', 2)
    smoothing = lm_cfg.get('smoothing', 'add_k')
    smoothing_k = lm_cfg.get('smoothing_k', 0.01)
    unk_threshold = lm_cfg.get('unk_threshold', 2)
    threshold = args.threshold or sys_cfg.get('abstain_threshold', 1.5)
    
    print("\n" + "="*70)
    print("📋 Configuration")
    print("="*70)
    print(f"  Student: {config.get('student_name', 'N/A')} ({config.get('student_id', 'N/A')})")
    print(f"  N-gram order: {ngram_order}, Smoothing: {smoothing}")
    print(f"  Threshold: {threshold}")
    print("="*70)
    
    system = build_restoration_system(
        train, ngram_order=ngram_order, smoothing_k=smoothing_k,
        unk_threshold=unk_threshold, threshold=threshold,
        context_weight=score_cfg.get('context_weight', 0.65),
        prior_weight=score_cfg.get('prior_weight', 0.20),
        edit_weight=score_cfg.get('edit_weight', 0.15),
        edit_lambda=score_cfg.get('edit_lambda', 0.10),
        lowercase=lm_cfg.get('lowercase', True),
        use_boundaries=lm_cfg.get('use_boundaries', True),
        smoothing=smoothing,
        preserve_case=sys_cfg.get('preserve_case', True),
        verbose=args.verbose or sys_cfg.get('verbose', False),
        config=cand_cfg,
        extra_wordlist_path=config.get('data', {}).get('extra_wordlist')
    )
    
    if args.tune:
        print("\nTuning threshold...")
        best, _ = system.tune_threshold(dev_noisy[:500], dev_clean[:500])
        system.threshold = best
    
    print(f"\nPerplexity: {system.lm.get_perplexity(dev_clean[:1000]):.2f}")
    
    print("\nProcessing dev set...")
    system.reset_statistics()
    dev_preds, _ = system.restore_sentences(dev_noisy)
    stats = system.get_statistics()
    print(f"  Changed: {stats['tokens_changed']} ({stats['change_rate']*100:.2f}%)")
    
    print("\nProcessing test set...")
    system.reset_counters()
    test_preds, _ = system.restore_sentences(test_noisy)
    
    print("\nGenerating baselines...")
    vocab = system.lm.get_vocab()
    unigram_probs = {w: system.lm.get_probability(w, tuple()) for w in vocab}
    unigram_preds = unigram_baseline(dev_noisy, unigram_probs)
    
    print("Computing metrics...")
    evaluator = Evaluator()
    main_metrics = evaluator.evaluate(dev_noisy, dev_preds, dev_clean)
    no_change_metrics = evaluator.evaluate(dev_noisy, dev_noisy, dev_clean)
    unigram_metrics = evaluator.evaluate(dev_noisy, unigram_preds, dev_clean)
    
    print("\nWriting output files...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dev_ids = list(range(len(dev_noisy)))
    dev_edited = [d != p for d, p in zip(dev_noisy, dev_preds)]
    system.save_dev_predictions(dev_ids, dev_noisy, dev_preds, dev_clean, dev_edited, out_dir / 'dev_predictions.csv')
    system.save_test_predictions(list(range(len(test_noisy))), test_noisy, test_preds, out_dir / 'test_predictions.csv')
    system.save_dev_metrics(main_metrics, out_dir / 'dev_metrics.json')
    system.save_summary({
        'student_name': config.get('student_name', 'Alireza Jahedi'),
        'student_id': config.get('student_id', '40435338'),
        'ngram_order': ngram_order,
        'smoothing': smoothing,
        'tokenization': 'whitespace',
        'case_handling': 'lowercase' if lm_cfg.get('lowercase', True) else 'preserve',
        'unk_threshold': unk_threshold,
        'abstain_threshold': system.threshold,
        'candidate_strategy': 'edit_distance_1 + edit_distance_2 + dictionary + confusion_rules',
        'score_weights': {
            'context': score_cfg.get('context_weight', 0.65),
            'prior': score_cfg.get('prior_weight', 0.20),
            'edit_penalty': score_cfg.get('edit_weight', 0.15)
        },
        'random_seed': args.seed
    }, out_dir / 'summary.json')
    system.save_trace(out_dir / 'trace.jsonl')
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("\nMain System Metrics (Dev Set):")
    for k in ['CER', 'WER', 'edit_F1', 'edit_precision', 'edit_recall', 'overcorrection_rate', 'sentence_accuracy', 'token_accuracy']:
        if k in main_metrics:
            print(f"  {k}: {main_metrics[k]:.4f}")
    
    print("\nNo-Change Baseline:")
    for k in ['CER', 'WER', 'edit_F1', 'sentence_accuracy', 'token_accuracy']:
        if k in no_change_metrics:
            print(f"  {k}: {no_change_metrics[k]:.4f}")
    
    print("\nUnigram Baseline:")
    for k in ['CER', 'WER', 'edit_F1', 'sentence_accuracy', 'token_accuracy']:
        if k in unigram_metrics:
            print(f"  {k}: {unigram_metrics[k]:.4f}")
    
    stats = system.get_statistics()
    print(f"\nSystem: {stats['tokens_changed']} changed, threshold={system.threshold}")
    print(f"Outputs written to: {out_dir}")
    print("="*70)


if __name__ == "__main__":
    main()