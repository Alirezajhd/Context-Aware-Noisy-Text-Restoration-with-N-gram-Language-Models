"""
ngram_model.py - Complete N-gram Language Model
"""
import math, re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any


class NGramModel:
    def __init__(
        self,
        n: int = 2,
        k: float = 0.01,
        unk_threshold: int = 2,
        lowercase: bool = True,
        use_boundaries: bool = True,
        start_token: str = '<s>',
        end_token: str = '</s>',
        smoothing: str = 'add_k'
    ):
        self.n = n
        self.k = k
        self.unk_threshold = unk_threshold
        self.lowercase = lowercase
        self.use_boundaries = use_boundaries
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = '<UNK>'
        self.smoothing = smoothing
        self.discount = 0.75
        self.backoff_alpha = 0.4
        
        self.counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.context_totals: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.continuation_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.unique_continuations: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.total_continuations: int = 0
        
        self.vocab: set = set()
        self.vocab_size: int = 0
        self.total_tokens: int = 0
        self.is_trained = False
        self.prob_cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        self.training_stats = {}
    
    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split() if text else []
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        return tokens
    
    def add_boundaries(self, tokens: List[str]) -> List[str]:
        return [self.start_token] + tokens + [self.end_token] if self.use_boundaries else tokens
    
    def replace_rare_tokens(self, tokens: List[str], token_counts: Counter) -> List[str]:
        return [t if token_counts.get(t, 0) >= self.unk_threshold else self.unk_token for t in tokens]
    
    def build_counts(self, sentences: List[str]) -> None:
        print(f"Building {self._order_name()} model...")
        print(f"  k={self.k}, unk={self.unk_threshold}, smoothing={self.smoothing}")
        
        all_tokens = []
        for sent in sentences:
            all_tokens.extend(self.tokenize(sent))
        token_counts = Counter(all_tokens)
        
        rare = sum(1 for t, c in token_counts.items() if c < self.unk_threshold)
        print(f"  Rare tokens (< {self.unk_threshold}): {rare}/{len(token_counts)}")
        
        self.vocab = {t for t, c in token_counts.items() if c >= self.unk_threshold}
        self.vocab.add(self.unk_token)
        if self.use_boundaries:
            self.vocab.add(self.start_token)
            self.vocab.add(self.end_token)
        
        self.vocab_size = len(self.vocab)
        self.total_tokens = len(all_tokens)
        print(f"✓ Vocab: {self.vocab_size:,}, Tokens: {self.total_tokens:,}")
        
        print("  Counting unigrams...")
        for t, c in token_counts.items():
            self.counts[tuple()][t] += c
            self.context_totals[tuple()] += c
        
        processed = []
        for sent in sentences:
            tokens = self.tokenize(sent)
            tokens = self.replace_rare_tokens(tokens, token_counts)
            processed.append(self.add_boundaries(tokens))
        
        if self.n > 1:
            print(f"  Counting {self._order_name()}s...")
            for tokens in processed:
                pad = [self.start_token] * (self.n - 1) + tokens
                for i in range(self.n - 1, len(pad)):
                    ctx = tuple(pad[i - self.n + 1:i])
                    self.counts[ctx][pad[i]] += 1
                    self.context_totals[ctx] += 1
        
        print(f"✓ Unique contexts: {len(self.counts):,}")
        
        if self.smoothing == 'kneser_ney' and self.n > 1:
            print("  Building continuation counts...")
            self._build_continuation_counts(processed)
            print(f"✓ Total continuations: {self.total_continuations:,}")
        
        self.is_trained = True
        self.training_stats = {
            'vocab_size': self.vocab_size, 'total_tokens': self.total_tokens,
            'unique_contexts': len(self.counts), 'rare_tokens': rare,
            'smoothing': self.smoothing, 'k': self.k, 'unk_threshold': self.unk_threshold
        }
        
        print("\n  Top 10 unigrams:")
        for w, c in sorted(self.counts[tuple()].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {w}: {c}")
    
    def _build_continuation_counts(self, processed: List[List[str]]) -> None:
        self.continuation_counts.clear()
        self.unique_continuations.clear()
        
        word_contexts = defaultdict(set)
        for tokens in processed:
            pad = [self.start_token] + tokens
            for i in range(1, len(pad)):
                word_contexts[pad[i]].add(pad[i-1])
        
        for w, ctxs in word_contexts.items():
            self.continuation_counts[tuple()][w] = len(ctxs)
            self.unique_continuations[tuple()] += len(ctxs)
        
        for order in range(2, self.n + 1):
            followers = defaultdict(set)
            for tokens in processed:
                pad = [self.start_token] * (order - 1) + tokens
                for i in range(order - 1, len(pad)):
                    ctx = tuple(pad[i - order + 1:i])
                    followers[ctx].add(pad[i])
            
            for ctx, words in followers.items():
                for w in words:
                    self.continuation_counts[ctx][w] += 1
                self.unique_continuations[ctx] = len(words)
        
        self.total_continuations = self.unique_continuations.get(tuple(), 0)
        print(f"  'the' cont: {self.continuation_counts.get(tuple(), {}).get('the', 0)}")
    
    def _order_name(self) -> str:
        return {1: 'Unigram', 2: 'Bigram', 3: 'Trigram'}.get(self.n, f'{self.n}-gram')
    
    def get_probability(self, word: str, context: Tuple[str, ...] = tuple()) -> float:
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if word not in self.vocab:
            word = self.unk_token
        context = tuple(c if c in self.vocab else self.unk_token for c in context)
        
        key = (word, context)
        if key in self.prob_cache:
            return self.prob_cache[key]
        
        if self.smoothing == 'kneser_ney':
            prob = self._kn_prob(word, context)
        elif self.smoothing == 'stupid_backoff':
            prob = self._sb_prob(word, context)
        else:
            prob = self._add_k_prob(word, context)
        
        if prob <= 0 or math.isnan(prob):
            prob = 1e-10
        self.prob_cache[key] = prob
        return prob
    
    def _add_k_prob(self, word: str, context: Tuple[str, ...]) -> float:
        if not context:
            return (self.counts[tuple()].get(word, 0) + self.k) / (self.total_tokens + self.k * self.vocab_size)
        
        count = self.counts[context].get(word, 0) if context in self.counts else 0
        ctx_count = self.context_totals.get(context, 0)
        
        if ctx_count == 0:
            return self._add_k_prob(word, context[1:] if len(context) > 0 else tuple())
        return (count + self.k) / (ctx_count + self.k * self.vocab_size)
    
    def _kn_prob(self, word: str, context: Tuple[str, ...]) -> float:
        if not context:
            if word == self.unk_token:
                rare = self.vocab_size - len(self.counts[tuple()])
                return rare / (self.total_tokens + 1) if self.total_tokens > 0 else 1e-6
            
            cont = self.continuation_counts.get(tuple(), {}).get(word, 0)
            return cont / self.total_continuations if self.total_continuations > 0 else self._add_k_prob(word, context)
        
        count = self.counts[context].get(word, 0) if context in self.counts else 0
        ctx_count = self.context_totals.get(context, 0)
        
        p_high = max(count - self.discount, 0) / ctx_count if ctx_count > 0 else 0
        
        unique = self.unique_continuations.get(context, 0)
        lamb = (self.discount / ctx_count * unique) if ctx_count > 0 else 1.0
        lamb = min(1.0, lamb)
        
        p_low = self._kn_prob(word, context[1:] if len(context) > 1 else tuple())
        prob = p_high + lamb * p_low
        return prob if prob > 0 else 1e-10
    
    def _sb_prob(self, word: str, context: Tuple[str, ...]) -> float:
        if not context:
            return self.counts[tuple()].get(word, 0) / self.total_tokens if self.total_tokens > 0 else 1e-10
        
        count = self.counts[context].get(word, 0) if context in self.counts else 0
        ctx_count = self.context_totals.get(context, 0)
        
        if count > 0 and ctx_count > 0:
            return max(count / ctx_count, 1e-10)
        return max(self.backoff_alpha * self._sb_prob(word, context[1:] if len(context) > 1 else tuple()), 1e-10)
    
    def get_log_probability(self, word: str, context: Tuple[str, ...] = tuple()) -> float:
        prob = self.get_probability(word, context)
        return math.log(prob) if prob > 0 else -1000.0
    
    def get_perplexity(self, sentences: List[str]) -> float:
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.prob_cache = {}
        total_log, total_tokens = 0.0, 0
        
        for sent in sentences:
            tokens = self.tokenize(sent)
            tokens = [t if t in self.vocab else self.unk_token for t in tokens]
            if self.use_boundaries:
                tokens = [self.start_token] + tokens + [self.end_token]
            
            if self.n == 1:
                for w in tokens:
                    total_log += self.get_log_probability(w, tuple())
                    total_tokens += 1
            else:
                pad = [self.start_token] * (self.n - 1) + tokens
                for i in range(self.n - 1, len(pad)):
                    ctx = tuple(pad[i - self.n + 1:i])
                    total_log += self.get_log_probability(pad[i], ctx)
                    total_tokens += 1
        
        if total_tokens == 0:
            return float('inf')
        avg = total_log / total_tokens
        return float('inf') if avg < -100 else math.exp(-avg)
    
    def get_vocab(self) -> set:
        return self.vocab
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_total_tokens(self) -> int:
        return self.total_tokens
    
    def get_stats(self) -> Dict[str, Any]:
        return self.training_stats
    
    def clear_cache(self):
        self.prob_cache = {}


def build_ngram_model(
    sentences: List[str],
    n: int = 2,
    k: float = 0.01,
    unk_threshold: int = 2,
    lowercase: bool = True,
    use_boundaries: bool = True,
    smoothing: str = 'add_k'
) -> NGramModel:
    model = NGramModel(n, k, unk_threshold, lowercase, use_boundaries, '<s>', '</s>', smoothing)
    model.build_counts(sentences)
    return model


if __name__ == "__main__":
    train = ["the quick brown fox jumps over the lazy dog", "the dog barks at the cat"]
    model = build_ngram_model(train, n=2, smoothing='kneser_ney')
    print(f"\nSmoothing: {model.smoothing}")
    print(f"P('the'): {model.get_probability('the', tuple()):.6f}")
    print(f"P('<UNK>'): {model.get_probability('<UNK>', tuple()):.6f}")
    print(f"P('fox' | 'brown'): {model.get_probability('fox', ('brown',)):.6f}")