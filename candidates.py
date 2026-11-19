"""
candidates.py - Candidate Generation for Noisy Text Restoration
"""
import re, math
from typing import List, Set, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter


class CandidateGenerator:
    KEYBOARD = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wrdsf',
        'f': 'drgcxv', 'g': 'ftyhvcd', 'h': 'gyujnbg', 'i': 'uokjl',
        'j': 'huiklmn', 'k': 'ijolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm',
        'o': 'ipkl', 'p': 'olk', 'q': 'wa', 'r': 'etdf', 's': 'awedx',
        't': 'rygf', 'u': 'yijkh', 'v': 'cfgb', 'w': 'qesa', 'x': 'zscf',
        'y': 'tuhg', 'z': 'axs'
    }
    TOKEN_PATTERN = re.compile(r"^([^A-Za-z0-9]*)([A-Za-z0-9][A-Za-z0-9'-]*)([^A-Za-z0-9]*)$")
    
    def __init__(
        self,
        vocabulary: Set[str],
        max_ed1_candidates: int = 30,
        max_ed2_candidates: int = 10,
        max_total_candidates: int = 50,
        min_ed1_length: int = 2,
        min_ed2_length: int = 5,
        use_keyboard: bool = True,
        use_transposition: bool = True,
        use_confusion_rules: bool = True,
        debug: bool = False,
        suspicious_threshold: float = 0.3
    ):
        self.vocabulary = vocabulary
        self.vocab_lower = {w.lower() for w in vocabulary}
        self.max_ed1_candidates = max_ed1_candidates
        self.max_ed2_candidates = max_ed2_candidates
        self.max_total_candidates = max_total_candidates
        self.min_ed1_length = min_ed1_length
        self.min_ed2_length = min_ed2_length
        self.use_keyboard = use_keyboard
        self.use_transposition = use_transposition
        self.use_confusion_rules = use_confusion_rules
        self.debug = debug
        self.suspicious_threshold = suspicious_threshold
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.vowels = set('aeiouy')
        self.consonants = set('bcdfghjklmnpqrstvwxyz')
        self.rare_letters = set('jqxyzvw')
        
        self._keyboard_cache = self._build_keyboard_cache()
        self._char_freq = self._build_char_frequency()
        self.confusion_rules = self._build_confusion_rules()
        self.cache: Dict[str, List[str]] = {}
        self._substitution_map = self._build_substitution_map()
        self._length_index = self._build_length_index()
        self.stats = {'total_requests': 0, 'cache_hits': 0, 'total_candidates_generated': 0, 'avg_candidates': 0}
    
    # ==================== INDEX BUILDING ====================
    def _build_keyboard_cache(self) -> Dict[str, Set[str]]:
        cache = {}
        for c in self.alphabet:
            cache[c] = set(self.KEYBOARD.get(c, ''))
            cache[c.upper()] = {n.upper() for n in cache[c]}
        return cache
    
    def _build_char_frequency(self) -> Dict[str, float]:
        freq = defaultdict(int)
        total = sum(len(w) for w in self.vocab_lower)
        for w in self.vocab_lower:
            for c in w:
                freq[c] += 1
        return {c: freq[c] / total for c in freq} if total > 0 else {}
    
    def _build_substitution_map(self) -> Dict[str, Set[str]]:
        sub = defaultdict(set)
        for c, ns in self.KEYBOARD.items():
            for n in ns:
                sub[c].add(n)
                sub[c.upper()].add(n.upper())
        pairs = [('0','o'),('1','l'),('5','S'),('8','B'),('6','G'),('2','Z'),('3','E'),('4','A'),('7','T'),('9','g')]
        for a, b in pairs:
            sub[a].add(b); sub[b].add(a)
        return dict(sub)
    
    def _build_length_index(self) -> Dict[int, Set[str]]:
        idx = defaultdict(set)
        for w in self.vocab_lower:
            idx[len(w)].add(w)
        return dict(idx)
    
    def _build_confusion_rules(self) -> Dict[str, Set[str]]:
        rules = defaultdict(set)
        for a, b in [('rn','m'),('m','rn'),('cl','d'),('d','cl'),('vv','w'),('w','vv'),('0','o'),('1','l')]:
            rules[a].add(b); rules[b].add(a)
        return dict(rules)
    
    # ==================== TOKEN HANDLING ====================
    @staticmethod
    def split_token(token: str) -> Tuple[str, str, str]:
        m = CandidateGenerator.TOKEN_PATTERN.match(token)
        return (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")
    
    @staticmethod
    def apply_case(word: str, core: str) -> str:
        if not word:
            return word
        if core.isupper():
            return word.upper()
        if core.istitle():
            return word.capitalize()
        return word.lower()
    
    # ==================== SUSPICIOUSNESS ====================
    def is_in_vocabulary(self, word: str) -> bool:
        return word.lower() in self.vocab_lower
    
    def _has_repeated_chars(self, word: str) -> bool:
        if len(word) < 3:
            return False
        cnt = 1
        for i in range(1, len(word)):
            cnt = cnt + 1 if word[i] == word[i-1] else 1
            if cnt >= 3:
                return True
        return False
    
    def _has_unusual_length(self, word: str) -> bool:
        vc = sum(1 for c in word if c in self.vowels)
        cc = len(word) - vc
        if len(word) == 2 and vc == 0:
            return True
        if len(word) == 3 and (vc == 0 or (vc <= 1 and cc >= 2)):
            return True
        if vc == 0 and len(word) >= 4:
            return True
        if cc == 0 and len(word) >= 3:
            return True
        return False
    
    def _has_unusual_char_distribution(self, word: str) -> bool:
        max_cons = max_vow = c_cons = c_vow = 0
        for ch in word:
            if ch in self.consonants:
                c_cons += 1; max_cons = max(max_cons, c_cons); c_vow = 0
            elif ch in self.vowels:
                c_vow += 1; max_vow = max(max_vow, c_vow); c_cons = 0
            else:
                c_cons = c_vow = 0
        if max_cons >= 4 or max_vow >= 3:
            return True
        rare = sum(1 for c in word if c in self.rare_letters)
        if len(word) > 0 and rare / len(word) > 0.5:
            return True
        for bg in ['jj','qq','xx','zz','wx','xz','zx','qz','bj','cj','fj','gj','jx','kx','px','qx']:
            if bg in word:
                return True
        return False
    
    def _has_low_vocabulary_similarity(self, word: str) -> bool:
        if len(word) > 3:
            return False
        chars = 'abcdefghijklmnopqrstuvwxyz'
        for i in range(len(word)):
            for c in chars:
                if c == word[i]:
                    continue
                v = word[:i] + c + word[i+1:]
                if v in self.vocab_lower:
                    return False
        return True
    
    def _compute_entropy(self, word: str) -> float:
        if not word:
            return 0.0
        counts = Counter(word)
        return -sum((c / len(word)) * math.log2(c / len(word)) for c in counts.values() if c > 0)
    
    def get_suspiciousness_score(self, word: str, freq: Optional[float] = None) -> float:
        w = word.lower()
        score = 0.0
        if not self.is_in_vocabulary(word):
            score += 0.6
        if freq is not None:
            score += 0.3 if freq < 0.0001 else (0.15 if freq < 0.001 else 0)
        if any(c.isdigit() for c in word):
            score += 0.3
        if self._has_repeated_chars(w):
            score += 0.3
        if self._has_unusual_length(w):
            score += 0.2
        if self._has_unusual_char_distribution(w):
            score += 0.25
        if self._has_low_vocabulary_similarity(w):
            score += 0.2
        if self._compute_entropy(w) > 3.0:
            score += 0.15
        return min(score, 1.0)
    
    def should_generate_candidates(self, word: str, freq: Optional[float] = None) -> bool:
        if not self.is_in_vocabulary(word):
            return True
        return self.get_suspiciousness_score(word, freq) >= self.suspicious_threshold
    
    # ==================== CANDIDATE GENERATION ====================
    def apply_confusion_rules(self, word: str) -> List[str]:
        if not self.use_confusion_rules:
            return []
        candidates = set()
        w = word.lower()
        for pat, rep in [('rn','m'),('cl','d'),('vv','w')]:
            if pat in w:
                c = w.replace(pat, rep)
                if c in self.vocab_lower:
                    candidates.add(c)
        for i, ch in enumerate(w):
            if ch in self.confusion_rules:
                for rep in self.confusion_rules[ch]:
                    if len(rep) == 1:
                        c = w[:i] + rep + w[i+1:]
                        if c in self.vocab_lower:
                            candidates.add(c)
        lead, core, trail = self.split_token(word)
        return [f"{lead}{self.apply_case(c, core)}{trail}" for c in candidates]
    
    def generate_case_variants(self, word: str) -> List[str]:
        variants = {word, word.lower(), word.upper(), word.capitalize()}
        if len(word) <= 4:
            for i in range(1, len(word)):
                variants.add(word[:i].upper() + word[i:].lower())
                variants.add(word[:i].lower() + word[i:].upper())
        return [v for v in variants if v in self.vocabulary or v == word]
    
    def generate_ed1_candidates(self, word: str) -> List[str]:
        if len(word) < self.min_ed1_length:
            return []
        w = word.lower()
        candidates = set()
        # Substitutions
        for i, ch in enumerate(w):
            for sub in self._substitution_map.get(ch, set()):
                c = w[:i] + sub + w[i+1:]
                if c in self.vocab_lower and c != w:
                    candidates.add(c)
        # Deletions
        for i in range(len(w)):
            c = w[:i] + w[i+1:]
            if c in self.vocab_lower and len(c) >= 1 and c != w:
                candidates.add(c)
        # Insertions
        if len(w) <= 6:
            for i in range(len(w) + 1):
                for c in self.alphabet:
                    cand = w[:i] + c + w[i:]
                    if cand in self.vocab_lower and cand != w:
                        candidates.add(cand)
        # Transpositions
        if self.use_transposition and len(w) >= 2:
            for i in range(len(w) - 1):
                c = w[:i] + w[i+1] + w[i] + w[i+2:]
                if c in self.vocab_lower and c != w:
                    candidates.add(c)
        lead, core, trail = self.split_token(word)
        result = [f"{lead}{self.apply_case(c, core)}{trail}" for c in candidates]
        result.sort(key=lambda x: (abs(len(x) - len(word)), -sum(self._char_freq.get(c, 0) for c in x.lower() if c.isalpha())))
        return result[:self.max_ed1_candidates]
    
    def generate_ed2_candidates(self, word: str) -> List[str]:
        if len(word) < self.min_ed2_length:
            return []
        w = word.lower()
        candidates = set()
        for i, ch in enumerate(w):
            for sub in self._substitution_map.get(ch, set()):
                inter = w[:i] + sub + w[i+1:]
                # Second substitution
                for j, ch2 in enumerate(inter):
                    for sub2 in self._substitution_map.get(ch2, set()):
                        c = inter[:j] + sub2 + inter[j+1:]
                        if c in self.vocab_lower and c != w:
                            candidates.add(c)
                # Deletion on intermediate
                for j in range(len(inter)):
                    c = inter[:j] + inter[j+1:]
                    if c in self.vocab_lower and c != w:
                        candidates.add(c)
                # Insertion on intermediate
                if len(inter) <= 6:
                    for j in range(len(inter) + 1):
                        for c in self.alphabet:
                            cand = inter[:j] + c + inter[j:]
                            if cand in self.vocab_lower and cand != w:
                                candidates.add(cand)
        result = sorted(candidates, key=lambda x: (self._edit_distance(x, w), len(x)))
        lead, core, trail = self.split_token(word)
        return [f"{lead}{self.apply_case(c, core)}{trail}" for c in result[:self.max_ed2_candidates]]
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j+1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]
    
    def normalize_repeated_chars(self, word: str) -> List[str]:
        if len(word) < 3:
            return []
        candidates = set()
        w = word.lower()
        i = 0
        while i < len(w):
            j = i + 1
            while j < len(w) and w[j] == w[i]:
                j += 1
            cnt = j - i
            if cnt >= 2:
                c = w[:i] + w[i] + w[i+cnt:]
                if c in self.vocab_lower and c != w:
                    candidates.add(c)
                if cnt >= 3:
                    c = w[:i] + w[i]*2 + w[i+cnt:]
                    if c in self.vocab_lower and c != w:
                        candidates.add(c)
            i = j
        result, prev = [], ''
        for ch in w:
            if ch != prev:
                result.append(ch)
                prev = ch
        c = ''.join(result)
        if c in self.vocab_lower and c != w:
            candidates.add(c)
        lead, core, trail = self.split_token(word)
        return [f"{lead}{self.apply_case(c, core)}{trail}" for c in candidates]
    
    def generate_split_candidates(self, word: str) -> List[str]:
        w = word.lower()
        if len(w) < 6:
            return []
        candidates = []
        for i in range(2, len(w) - 1):
            l, r = w[:i], w[i:]
            if l in self.vocab_lower and r in self.vocab_lower:
                candidates.append(f"{l} {r}")
        return candidates
    
    def generate_candidates(
        self,
        word: str,
        is_oov: bool = True,
        unigram_frequency: Optional[float] = None,
        include_original: bool = True,
        include_case_variants: bool = True,
        include_confusion_rules: bool = True,
        include_ed1: bool = True,
        include_ed2: bool = True,
        include_repetition: bool = True,
        include_split: bool = True
    ) -> List[str]:
        self.stats['total_requests'] += 1
        if not self.should_generate_candidates(word, unigram_frequency):
            return [word]
        
        key = f"{word}_{is_oov}_{include_case_variants}_{include_confusion_rules}_{include_ed1}_{include_ed2}_{include_repetition}_{include_split}"
        if key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[key]
        
        candidates = set()
        if include_original:
            candidates.add(word)
        if include_case_variants:
            candidates.update(self.generate_case_variants(word))
        if include_confusion_rules and self.use_confusion_rules:
            candidates.update(self.apply_confusion_rules(word))
        if include_ed1 and len(word) >= self.min_ed1_length:
            candidates.update(self.generate_ed1_candidates(word))
        if include_ed2 and is_oov and len(word) >= self.min_ed2_length:
            candidates.update(self.generate_ed2_candidates(word))
        if include_repetition and len(word) >= 3:
            candidates.update(self.normalize_repeated_chars(word))
        if include_split:
            candidates.update(self.generate_split_candidates(word))
        
        result = self._prioritize_candidates(list(candidates), word)[:self.max_total_candidates]
        self.stats['total_candidates_generated'] += len(result)
        self.stats['avg_candidates'] = self.stats['total_candidates_generated'] / self.stats['total_requests']
        self.cache[key] = result
        return result
    
    def _prioritize_candidates(self, candidates: List[str], original: str) -> List[str]:
        def score(c: str) -> Tuple[int, int, int, str]:
            tier = 0 if c == original else (1 if self._is_confusion_variant(c, original) else (2 if c in self.vocabulary else 3))
            return (tier, self._edit_distance(c.lower(), original.lower()), abs(len(c) - len(original)), c)
        return sorted(candidates, key=score)
    
    def _is_confusion_variant(self, c: str, orig: str) -> bool:
        return self.use_confusion_rules and c in self.apply_confusion_rules(orig)
    
    def clear_cache(self):
        self.cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats
    
    def get_vocabulary_size(self) -> int:
        return len(self.vocabulary)