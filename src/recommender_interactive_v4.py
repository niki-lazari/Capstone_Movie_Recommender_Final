"""
Movie Recommender - Main Entry Point
Orchestrates the complete hybrid recommendation pipeline.

THIS IS IT - The module that finally uses all your 6 months of work properly!

Pipeline:
1. Parse natural language query → extract filters
2. Get candidate movies from data
3. Compute all 6 signals via signal fusion
4. Score and rank movies
5. Return top recommendations

Usage:
    recommender = MovieRecommender()
    results = recommender.recommend("kung fu movies from the 80s")
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Import our modules
try:
    from src.query_parser import QueryParser, ParsedQuery
    from src.signal_fusion import SignalFusion
    from src.scoring import Scorer, ScoredMovie
    from src.data_loader import DataLoader
    from src.zero_shot_integration import ZeroShotIntegrator
    from src.semantic_tag_expander import SemanticTagExpander
    from src.entity_extractor import EntityExtractor
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from query_parser import QueryParser, ParsedQuery
    from signal_fusion import SignalFusion
    from scoring import Scorer, ScoredMovie
    from data_loader import DataLoader
    from zero_shot_integration import ZeroShotIntegrator
    from semantic_tag_expander import SemanticTagExpander
    from entity_extractor import EntityExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SITUATION + OUTCOME QUERY DETECTION
# =============================================================================
# Detects queries like "My girlfriend broke up with me - I need something to cheer me up"
# Uses BERT to dynamically classify clauses as SITUATION vs OUTCOME (no hardcoding)

# Anchor phrases that define what "situation" and "outcome" mean semantically
SITUATION_ANCHORS = [
    'something bad happened to me',
    'I am going through a difficult time',
    'negative life event',
    'sad situation',
    'I am struggling',
    'dealing with hardship',
    'experiencing loss',
]

OUTCOME_ANCHORS = [
    'I want to feel better',
    'movie that will cheer me up',
    'something uplifting to watch',
    'positive movie experience',
    'make me happy',
    'entertainment I want',
    'type of movie I want to see',
]

# Cache for anchor embeddings (computed once, reused)
_situation_outcome_cache = {
    'situation_center': None,
    'outcome_center': None,
    'bert_model': None
}


def _get_situation_outcome_bert_model():
    """Get or load the BERT model for situation/outcome classification."""
    if _situation_outcome_cache['bert_model'] is None:
        from sentence_transformers import SentenceTransformer
        _situation_outcome_cache['bert_model'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Loaded BERT model for situation/outcome detection")
    return _situation_outcome_cache['bert_model']


def _get_anchor_centers():
    """Get or compute the anchor embedding centers."""
    if _situation_outcome_cache['situation_center'] is None:
        model = _get_situation_outcome_bert_model()

        # Compute anchor embeddings
        sit_emb = model.encode(SITUATION_ANCHORS)
        out_emb = model.encode(OUTCOME_ANCHORS)

        # Store centers (average of anchors)
        _situation_outcome_cache['situation_center'] = np.mean(sit_emb, axis=0)
        _situation_outcome_cache['outcome_center'] = np.mean(out_emb, axis=0)

    return (_situation_outcome_cache['situation_center'],
            _situation_outcome_cache['outcome_center'])


def _classify_clause(clause: str) -> dict:
    """
    Classify a single clause as SITUATION or OUTCOME using BERT similarity.

    Returns dict with:
        - classification: 'SITUATION' or 'OUTCOME'
        - situation_sim: similarity to situation anchors
        - outcome_sim: similarity to outcome anchors
        - confidence: absolute difference between similarities
    """
    import re
    model = _get_situation_outcome_bert_model()
    situation_center, outcome_center = _get_anchor_centers()

    # Encode the clause
    clause_emb = model.encode([clause])[0]

    # Compute cosine similarities
    sit_sim = np.dot(clause_emb, situation_center) / (
        np.linalg.norm(clause_emb) * np.linalg.norm(situation_center)
    )
    out_sim = np.dot(clause_emb, outcome_center) / (
        np.linalg.norm(clause_emb) * np.linalg.norm(outcome_center)
    )

    return {
        'clause': clause,
        'classification': 'SITUATION' if sit_sim > out_sim else 'OUTCOME',
        'situation_sim': float(sit_sim),
        'outcome_sim': float(out_sim),
        'confidence': abs(sit_sim - out_sim)
    }


def _split_query_into_clauses(query: str) -> List[str]:
    """
    Split a query into semantic clauses using common delimiters.

    Handles:
    - Dashes (—, –, -)
    - Semicolons (;)
    - Periods (.)
    - Colons (:)
    - Words: "but", "so"
    - Smart commas: only split on commas followed by a new clause (verb/pronoun)

    Uses spaCy for smart comma detection to avoid splitting adjective lists.
    """
    import re
    import spacy

    # Load spaCy model (cached after first load)
    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        # Fallback if spaCy not available - basic splitting only
        nlp = None

    q = query.lower()

    # Handle em-dash/en-dash/hyphen with spaces (but not hyphens in words)
    q = re.sub(r'\s*[—–]\s*', ' CLAUSE_SPLIT ', q)
    q = re.sub(r'\s+-\s+', ' CLAUSE_SPLIT ', q)

    # Split on semicolons
    q = re.sub(r'\s*;\s*', ' CLAUSE_SPLIT ', q)

    # Split on periods (but not decimals like "7.5")
    q = re.sub(r'\.(?!\d)\s*', ' CLAUSE_SPLIT ', q)

    # Split on colons
    q = re.sub(r'\s*:\s*', ' CLAUSE_SPLIT ', q)

    # Split on clause-connecting words
    q = re.sub(r'\s+but\s+', ' CLAUSE_SPLIT ', q)
    q = re.sub(r'\s+so\s+', ' CLAUSE_SPLIT ', q)

    # Smart comma handling - only split if followed by a new clause
    if nlp and ',' in q:
        # Process with spaCy to check what follows each comma
        doc = nlp(q)

        # Find comma positions and check what follows
        new_q = ""
        i = 0
        tokens = list(doc)

        while i < len(tokens):
            token = tokens[i]
            new_q += token.text_with_ws if hasattr(token, 'text_with_ws') else token.text + " "

            # If this token is a comma, check what follows
            if token.text == ',' and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                # Split if next token is a verb, pronoun, or clause-starting word
                # VERB, AUX (auxiliary verb), PRON (pronoun like "I", "my", "we")
                if next_token.pos_ in ('VERB', 'AUX', 'PRON'):
                    new_q = new_q.rstrip(', ') + ' CLAUSE_SPLIT '
            i += 1
        q = new_q

    # Split and clean
    clauses = q.split('CLAUSE_SPLIT')
    clauses = [c.strip() for c in clauses if c.strip()]

    return clauses


def _filter_keywords_by_clause_similarity(
    keywords: List[str],
    situation_clause: str,
    outcome_clause: str,
    keep_outcome: bool = True
) -> List[str]:
    """
    Filter keywords based on similarity to the ACTUAL query clauses (dynamic anchors).

    Instead of hardcoded positive/negative word lists, we use the user's own
    situation and outcome clauses as reference points.

    Args:
        keywords: List of keywords to filter
        situation_clause: The user's actual situation clause (e.g., "my girlfriend broke up with me")
        outcome_clause: The user's actual outcome clause (e.g., "I need something to cheer me up")
        keep_outcome: If True, keep keywords closer to outcome clause
                      If False, keep keywords closer to situation clause

    Returns:
        Filtered list of keywords
    """
    if not keywords:
        return []

    if not situation_clause or not outcome_clause:
        # Can't filter without both clauses - return all keywords
        return keywords

    model = _get_situation_outcome_bert_model()

    # Encode the actual clauses as dynamic anchors
    situation_emb = model.encode([situation_clause])[0]
    outcome_emb = model.encode([outcome_clause])[0]

    filtered = []
    for kw in keywords:
        # Encode the keyword
        kw_emb = model.encode([kw])[0]

        # Compute similarities to ACTUAL clause embeddings (not hardcoded anchors)
        sit_sim = np.dot(kw_emb, situation_emb) / (
            np.linalg.norm(kw_emb) * np.linalg.norm(situation_emb)
        )
        out_sim = np.dot(kw_emb, outcome_emb) / (
            np.linalg.norm(kw_emb) * np.linalg.norm(outcome_emb)
        )

        # Keep based on which clause it's closer to
        is_outcome_like = out_sim > sit_sim

        if keep_outcome and is_outcome_like:
            filtered.append(kw)
        elif not keep_outcome and not is_outcome_like:
            filtered.append(kw)

    return filtered


def detect_situation_outcome_query(query: str) -> dict:
    """
    Detect if a query is a "situation + outcome" pattern.

    A situation+outcome query has:
    - A SITUATION clause: describes what happened to the user (e.g., "my girlfriend broke up with me")
    - An OUTCOME clause: describes what movie experience they want (e.g., "I need something to cheer me up")

    Args:
        query: The raw query string

    Returns:
        dict with:
            - is_situation_outcome: True if both situation and outcome detected
            - situations: list of situation clauses
            - outcomes: list of outcome clauses
            - clause_details: detailed classification for each clause
    """
    clauses = _split_query_into_clauses(query)

    situations = []
    outcomes = []
    clause_details = []

    for clause in clauses:
        result = _classify_clause(clause)
        clause_details.append(result)

        # Only count if confidence is above threshold
        if result['confidence'] > 0.02:
            if result['classification'] == 'SITUATION':
                situations.append(clause)
            else:
                outcomes.append(clause)

    # It's a situation+outcome query if we have BOTH
    is_sit_out = len(situations) > 0 and len(outcomes) > 0

    return {
        'is_situation_outcome': is_sit_out,
        'situations': situations,
        'outcomes': outcomes,
        'clause_details': clause_details
    }


def _extract_key_terms_from_clause(clause: str) -> Dict[str, List[str]]:
    """
    Extract meaningful terms from a clause, separating phrasal verbs from context nouns.

    Example: "my girlfriend broke up with me" -> {
        'phrasal_verbs': ['broke up'],
        'context_nouns': ['girlfriend'],
        'adjectives': []
    }
    Example: "i need something to cheer me up" -> {
        'phrasal_verbs': ['cheer up'],
        'context_nouns': [],
        'adjectives': []
    }
    Example: "my cat ran away" -> {
        'phrasal_verbs': ['ran away'],
        'context_nouns': ['cat'],
        'adjectives': []
    }

    Uses spaCy for DYNAMIC phrase detection (no hardcoded patterns).

    Returns a dict with:
    - phrasal_verbs: verb + particle phrases (the PRIMARY indicators for situation/outcome)
    - context_nouns: descriptor entities (background context, less important)
    - adjectives: emotional descriptors like "depressed", "terrible"

    The phrasal_verbs are what drive candidate selection. Context nouns are secondary.
    """
    import spacy

    result = {
        'phrasal_verbs': [],
        'context_nouns': [],
        'adjectives': []
    }

    # Load spaCy model (cached after first load)
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        logger.warning("spaCy model not found, falling back to simple extraction")
        import re
        words = re.findall(r'\b[a-z]+\b', clause.lower())
        stopwords = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'a', 'an', 'the',
                     'is', 'am', 'are', 'was', 'were', 'be', 'to', 'of', 'in', 'for',
                     'and', 'or', 'but', 'so', 'just', 'really', 'very', 'something',
                     'need', 'want', 'looking', 'give', 'get', 'got'}
        result['context_nouns'] = [w for w in words if w not in stopwords and len(w) > 2]
        return result

    doc = nlp(clause.lower())
    used_indices = set()

    # 1. Extract phrasal verbs (verb + particle) - THE PRIMARY INDICATORS
    # Examples: "broke up", "ran away", "cheer up", "feel good", "passed away"
    PHRASAL_PARTICLES = {'up', 'away', 'out', 'off', 'down', 'over', 'through', 'back', 'in', 'on'}

    for token in doc:
        if token.pos_ == 'VERB':
            particles = []
            for child in token.children:
                # Particle dependency (e.g., "broke up")
                if child.dep_ == 'prt':
                    particles.append(child)
                # Adverb modifier that's actually a phrasal particle (e.g., "ran away")
                elif child.dep_ == 'advmod' and child.text.lower() in PHRASAL_PARTICLES:
                    particles.append(child)
                # Preposition as part of phrasal verb (e.g., "going through")
                elif child.dep_ == 'prep' and child.text.lower() in PHRASAL_PARTICLES:
                    particles.append(child)
                # Adjective complements like "feel good", "feel better"
                elif child.dep_ in ['acomp', 'advmod'] and child.pos_ in ['ADJ', 'ADV']:
                    if child.text in ['good', 'better', 'bad', 'worse', 'great', 'terrible']:
                        particles.append(child)

            if particles:
                phrase_tokens = [token] + particles
                phrase_tokens.sort(key=lambda t: t.i)
                phrase = ' '.join([t.text for t in phrase_tokens])
                result['phrasal_verbs'].append(phrase)
                used_indices.update([t.i for t in phrase_tokens])

    # 1b. Catch "feel good" pattern when spaCy parses "feel" as noun (e.g., "feel good movie")
    # Look for "feel good" or "feel bad" patterns in the text
    import re
    feel_good_match = re.search(r'\bfeel[\s-]?(good|bad|better|worse|great)\b', clause.lower())
    if feel_good_match:
        phrase = feel_good_match.group(0).replace('-', ' ')
        if phrase not in result['phrasal_verbs']:
            result['phrasal_verbs'].append(phrase)
            # Mark these indices as used (approximate - find the tokens)
            for token in doc:
                if token.text in ['feel', 'good', 'bad', 'better', 'worse', 'great']:
                    used_indices.add(token.i)

    # 2. Extract standalone meaningful verbs (not stopwords like "need", "want", "get")
    VERB_STOPWORDS = {'need', 'want', 'get', 'got', 'give', 'make', 'let', 'have', 'do', 'be'}
    for token in doc:
        if token.pos_ == 'VERB' and token.i not in used_indices:
            if token.lemma_ not in VERB_STOPWORDS and len(token.text) > 2:
                result['phrasal_verbs'].append(token.lemma_)
                used_indices.add(token.i)

    # 3. Extract adjectives - emotional descriptors
    for token in doc:
        if token.pos_ == 'ADJ' and token.i not in used_indices:
            if not token.is_stop and len(token.text) > 2:
                result['adjectives'].append(token.text)
                used_indices.add(token.i)

    # 4. Extract context nouns (secondary - background info)
    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.i not in used_indices:
            if not head.is_stop and len(head.text) > 2:
                # Skip generic nouns
                if head.text not in ['something', 'anything', 'nothing', 'thing', 'movie', 'film']:
                    result['context_nouns'].append(head.text)
                    used_indices.add(head.i)

    # Dedupe each category
    result['phrasal_verbs'] = list(set(result['phrasal_verbs']))
    result['context_nouns'] = list(set(result['context_nouns']))
    result['adjectives'] = list(set(result['adjectives']))

    return result


def expand_situation_outcome_terms(situations: List[str], outcomes: List[str],
                                    semantic_expander) -> dict:
    """
    Expand situation and outcome phrases into searchable movie keywords/tags.

    Separates:
    - PRIMARY terms (phrasal verbs): drive candidate selection
    - SECONDARY terms (context nouns): background info, lower weight
    - ADJECTIVES: emotional descriptors

    Args:
        situations: List of situation clauses (e.g., ["my girlfriend broke up with me"])
        outcomes: List of outcome clauses (e.g., ["i need something to cheer me up"])
        semantic_expander: The SemanticTagExpander instance

    Returns:
        dict with:
            - situation_primary: phrasal verbs from situations (e.g., ["broke up"])
            - situation_secondary: context nouns from situations (e.g., ["girlfriend"])
            - situation_adjectives: adjectives from situations
            - outcome_primary: phrasal verbs from outcomes (e.g., ["cheer up", "feel good"])
            - outcome_secondary: context nouns from outcomes
            - outcome_adjectives: adjectives from outcomes
            - situation_expansions: dict mapping primary situation terms to expanded keywords
            - outcome_expansions: dict mapping primary outcome terms to expanded keywords
    """
    # Extract structured terms from each clause
    situation_primary = []
    situation_secondary = []
    situation_adjectives = []

    for clause in situations:
        extracted = _extract_key_terms_from_clause(clause)
        situation_primary.extend(extracted['phrasal_verbs'])
        situation_secondary.extend(extracted['context_nouns'])
        situation_adjectives.extend(extracted['adjectives'])

    # Dedupe
    situation_primary = list(set(situation_primary))
    situation_secondary = list(set(situation_secondary))
    situation_adjectives = list(set(situation_adjectives))

    outcome_primary = []
    outcome_secondary = []
    outcome_adjectives = []

    for clause in outcomes:
        extracted = _extract_key_terms_from_clause(clause)
        outcome_primary.extend(extracted['phrasal_verbs'])
        outcome_secondary.extend(extracted['context_nouns'])
        outcome_adjectives.extend(extracted['adjectives'])

    # Dedupe
    outcome_primary = list(set(outcome_primary))
    outcome_secondary = list(set(outcome_secondary))
    outcome_adjectives = list(set(outcome_adjectives))

    # Expand PRIMARY terms only (these drive candidate selection)
    situation_expansions = {}
    outcome_expansions = {}

    if semantic_expander:
        # Expand situation primary terms (phrasal verbs like "broke up", "ran away")
        for term in situation_primary:
            expanded = semantic_expander.expand_term(term, method='hybrid', search_keywords=True)
            if expanded:
                situation_expansions[term] = list(expanded)

        # Also expand adjectives (like "depressed", "sad") - they indicate mood
        for term in situation_adjectives:
            expanded = semantic_expander.expand_term(term, method='hybrid', search_keywords=True)
            if expanded:
                situation_expansions[term] = list(expanded)

        # Expand outcome primary terms (phrasal verbs like "cheer up", "feel good")
        for term in outcome_primary:
            expanded = semantic_expander.expand_term(term, method='hybrid', search_keywords=True)
            if expanded:
                outcome_expansions[term] = list(expanded)

        # Also expand outcome adjectives
        for term in outcome_adjectives:
            expanded = semantic_expander.expand_term(term, method='hybrid', search_keywords=True)
            if expanded:
                outcome_expansions[term] = list(expanded)

    return {
        'situation_primary': situation_primary,
        'situation_secondary': situation_secondary,
        'situation_adjectives': situation_adjectives,
        'outcome_primary': outcome_primary,
        'outcome_secondary': outcome_secondary,
        'outcome_adjectives': outcome_adjectives,
        'situation_expansions': situation_expansions,
        'outcome_expansions': outcome_expansions
    }


def get_situation_outcome_candidates(movies_df: pd.DataFrame,
                                      sit_out_expansions: dict,
                                      situation_clause: str,
                                      outcome_clause: str,
                                      semantic_expander,
                                      zero_shot_tags_df,
                                      max_candidates: int = 2000) -> List[str]:
    """
    Get candidate movies for situation+outcome queries.

    STRATEGY - Use DIFFERENT data sources for situation vs outcome:
    1. SITUATION: Expand terms to TMDB keywords, match against movie keywords/overview
       (Keywords describe what the movie is ABOUT - content/plot)
    2. OUTCOME: Expand terms to tags/genres, match against movie genre + zero-shot tags
       (Genre/tags describe the movie EXPERIENCE - comedy, feel-good, uplifting)
    3. Movie must match BOTH situation AND outcome to be a candidate

    This approach works dynamically for ANY query because:
    - Situation terms naturally expand to content keywords (breakup -> ex-girlfriend, divorce, dumped)
    - Outcome terms naturally expand to mood/genre tags (cheer up -> comedy, feel-good, uplifting)

    Args:
        movies_df: Full movies dataframe
        sit_out_expansions: Dict from expand_situation_outcome_terms()
        situation_clause: The user's actual situation clause
        outcome_clause: The user's actual outcome clause
        semantic_expander: SemanticTagExpander for expansion
        zero_shot_tags_df: DataFrame with zero-shot mood tags
        max_candidates: Maximum candidates to return

    Returns:
        List of candidate movie titles
    """
    # =================================================================
    # INFLECTION GENERATOR: Generate verb/noun variants for overview matching
    # =================================================================
    # Instead of lemmatizing every movie overview (slow), we generate inflected
    # variants of our situation keywords (fast) and do simple string matching.
    # This catches "dumps" in overview when we have "dump" as a keyword.

    def generate_inflections(word: str) -> set:
        """
        Generate common inflected forms of a word for matching.
        Uses simple rules to cover most English verb/noun variations.

        Examples:
            dump -> {dump, dumps, dumped, dumping}
            break -> {break, breaks, broke, broken, breaking}
            divorce -> {divorce, divorces, divorced, divorcing}
        """
        word = word.lower().strip()
        if len(word) < 3:
            return {word}

        variants = {word}

        # Common irregular verbs (situation-relevant)
        IRREGULARS = {
            'break': ['break', 'breaks', 'broke', 'broken', 'breaking'],
            'leave': ['leave', 'leaves', 'left', 'leaving'],
            'lose': ['lose', 'loses', 'lost', 'losing'],
            'die': ['die', 'dies', 'died', 'dying', 'dead', 'death'],
            'split': ['split', 'splits', 'splitting'],
            'hurt': ['hurt', 'hurts', 'hurting'],
            'feel': ['feel', 'feels', 'felt', 'feeling'],
            'go': ['go', 'goes', 'went', 'gone', 'going'],
            'get': ['get', 'gets', 'got', 'gotten', 'getting'],
            'fall': ['fall', 'falls', 'fell', 'fallen', 'falling'],
            'give': ['give', 'gives', 'gave', 'given', 'giving'],
            'take': ['take', 'takes', 'took', 'taken', 'taking'],
            'come': ['come', 'comes', 'came', 'coming'],
            'know': ['know', 'knows', 'knew', 'known', 'knowing'],
            'see': ['see', 'sees', 'saw', 'seen', 'seeing'],
            'tell': ['tell', 'tells', 'told', 'telling'],
            'find': ['find', 'finds', 'found', 'finding'],
            'think': ['think', 'thinks', 'thought', 'thinking'],
            'make': ['make', 'makes', 'made', 'making'],
            'say': ['say', 'says', 'said', 'saying'],
            'let': ['let', 'lets', 'letting'],
            'put': ['put', 'puts', 'putting'],
            'keep': ['keep', 'keeps', 'kept', 'keeping'],
            'begin': ['begin', 'begins', 'began', 'begun', 'beginning'],
            'run': ['run', 'runs', 'ran', 'running'],
            'write': ['write', 'writes', 'wrote', 'written', 'writing'],
            'meet': ['meet', 'meets', 'met', 'meeting'],
            'pay': ['pay', 'pays', 'paid', 'paying'],
            'sit': ['sit', 'sits', 'sat', 'sitting'],
            'stand': ['stand', 'stands', 'stood', 'standing'],
            'hear': ['hear', 'hears', 'heard', 'hearing'],
            'bring': ['bring', 'brings', 'brought', 'bringing'],
            'hold': ['hold', 'holds', 'held', 'holding'],
            'catch': ['catch', 'catches', 'caught', 'catching'],
            'fight': ['fight', 'fights', 'fought', 'fighting'],
            'win': ['win', 'wins', 'won', 'winning'],
            'spend': ['spend', 'spends', 'spent', 'spending'],
            'send': ['send', 'sends', 'sent', 'sending'],
            'build': ['build', 'builds', 'built', 'building'],
            'deal': ['deal', 'deals', 'dealt', 'dealing'],
            'sell': ['sell', 'sells', 'sold', 'selling'],
            'buy': ['buy', 'buys', 'bought', 'buying'],
            'lead': ['lead', 'leads', 'led', 'leading'],
            'understand': ['understand', 'understands', 'understood', 'understanding'],
            'draw': ['draw', 'draws', 'drew', 'drawn', 'drawing'],
            'grow': ['grow', 'grows', 'grew', 'grown', 'growing'],
            'throw': ['throw', 'throws', 'threw', 'thrown', 'throwing'],
            'show': ['show', 'shows', 'showed', 'shown', 'showing'],
            'fly': ['fly', 'flies', 'flew', 'flown', 'flying'],
            'drive': ['drive', 'drives', 'drove', 'driven', 'driving'],
            'speak': ['speak', 'speaks', 'spoke', 'spoken', 'speaking'],
            'rise': ['rise', 'rises', 'rose', 'risen', 'rising'],
            'wake': ['wake', 'wakes', 'woke', 'woken', 'waking'],
            'wear': ['wear', 'wears', 'wore', 'worn', 'wearing'],
            'lie': ['lie', 'lies', 'lay', 'lain', 'lying', 'lied'],  # both meanings
            'lay': ['lay', 'lays', 'laid', 'laying'],
            'steal': ['steal', 'steals', 'stole', 'stolen', 'stealing'],
            'choose': ['choose', 'chooses', 'chose', 'chosen', 'choosing'],
            'forget': ['forget', 'forgets', 'forgot', 'forgotten', 'forgetting'],
            'forgive': ['forgive', 'forgives', 'forgave', 'forgiven', 'forgiving'],
            'hide': ['hide', 'hides', 'hid', 'hidden', 'hiding'],
            'beat': ['beat', 'beats', 'beaten', 'beating'],
            'bite': ['bite', 'bites', 'bit', 'bitten', 'biting'],
            'blow': ['blow', 'blows', 'blew', 'blown', 'blowing'],
            'tear': ['tear', 'tears', 'tore', 'torn', 'tearing'],
            'strike': ['strike', 'strikes', 'struck', 'stricken', 'striking'],
            'shake': ['shake', 'shakes', 'shook', 'shaken', 'shaking'],
            'seek': ['seek', 'seeks', 'sought', 'seeking'],
            'shoot': ['shoot', 'shoots', 'shot', 'shooting'],
            'sing': ['sing', 'sings', 'sang', 'sung', 'singing'],
            'sink': ['sink', 'sinks', 'sank', 'sunk', 'sinking'],
            'swim': ['swim', 'swims', 'swam', 'swum', 'swimming'],
            'swing': ['swing', 'swings', 'swung', 'swinging'],
            'teach': ['teach', 'teaches', 'taught', 'teaching'],
        }

        # Check if word is an irregular verb
        for base, forms in IRREGULARS.items():
            if word in forms:
                variants.update(forms)
                return variants

        # Regular verb rules
        # Base: dump -> dumps, dumped, dumping
        if word.endswith('e'):
            # dance -> dances, danced, dancing
            variants.add(word + 's')
            variants.add(word + 'd')
            variants.add(word[:-1] + 'ing')
        elif word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
            # cry -> cries, cried, crying
            variants.add(word[:-1] + 'ies')
            variants.add(word[:-1] + 'ied')
            variants.add(word + 'ing')
        elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            # pass -> passes, passed, passing
            variants.add(word + 'es')
            variants.add(word + 'ed')
            variants.add(word + 'ing')
        elif len(word) > 2 and word[-1] not in 'aeiouwy' and word[-2] in 'aeiou' and word[-3] not in 'aeiou':
            # Single vowel + consonant: stop -> stops, stopped, stopping
            variants.add(word + 's')
            variants.add(word + word[-1] + 'ed')
            variants.add(word + word[-1] + 'ing')
        else:
            # Default: dump -> dumps, dumped, dumping
            variants.add(word + 's')
            variants.add(word + 'ed')
            variants.add(word + 'ing')

        # Noun plurals (for words that might be nouns)
        if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
            variants.add(word[:-1] + 'ies')
        elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            variants.add(word + 'es')
        else:
            variants.add(word + 's')

        return variants

    # TMDB genre ID to name mapping
    GENRE_ID_TO_NAME = {
        28: 'action', 12: 'adventure', 16: 'animation', 35: 'comedy',
        80: 'crime', 99: 'documentary', 18: 'drama', 10751: 'family',
        14: 'fantasy', 36: 'history', 27: 'horror', 10402: 'music',
        9648: 'mystery', 10749: 'romance', 878: 'science fiction',
        10770: 'tv movie', 53: 'thriller', 10752: 'war', 37: 'western'
    }

    # =================================================================
    # STEP 1: SITUATION PATH - Expand to TMDB keywords (content)
    # =================================================================
    situation_seeds = (
        sit_out_expansions.get('situation_primary', []) +
        sit_out_expansions.get('situation_adjectives', [])
        # Secondary terms (context nouns like "girlfriend") excluded - too generic for filtering
    )

    # Expand situation terms to TMDB KEYWORDS (search_keywords=True)
    situation_keywords = set()
    for term in situation_seeds:
        situation_keywords.add(term.lower())
        if semantic_expander:
            expanded = semantic_expander.expand_term(
                term, method='embedding', top_k=20,
                min_similarity=0.30, search_keywords=True  # TMDB keywords
            )
            situation_keywords.update([kw.lower() for kw in expanded])

    # Generate ALL inflected variants of situation keywords for flexible overview matching
    # e.g., "dump" -> {"dump", "dumps", "dumped", "dumping"}
    # This catches "dumps" in "...suddenly dumps him" when we have "dump" as keyword
    situation_variants = set()
    for kw in situation_keywords:
        # Only generate inflections for single words (not phrases)
        if ' ' not in kw:
            situation_variants.update(generate_inflections(kw))
        else:
            situation_variants.add(kw)  # Keep phrases as-is

    logger.info(f"   [SIT+OUT] Situation seeds: {situation_seeds}")
    logger.info(f"   [SIT+OUT] Situation keywords ({len(situation_keywords)}): {list(situation_keywords)[:15]}")
    logger.info(f"   [SIT+OUT] Situation variants ({len(situation_variants)}): {list(situation_variants)[:20]}")

    # =================================================================
    # STEP 2: OUTCOME PATH - Expand to tags/genres (experience)
    # =================================================================
    outcome_seeds = (
        sit_out_expansions.get('outcome_primary', []) +
        sit_out_expansions.get('outcome_adjectives', [])
    )

    # Expand outcome terms to ZERO-SHOT TAGS (search_keywords=False)
    outcome_tags = set()
    outcome_genres = set()

    for term in outcome_seeds:
        term_lower = term.lower()
        outcome_tags.add(term_lower)

        if semantic_expander:
            # Expand to zero-shot tags (mood descriptors)
            expanded_tags = semantic_expander.expand_term(
                term, method='embedding', top_k=20,
                min_similarity=0.30, search_keywords=False  # Zero-shot tags
            )
            outcome_tags.update([t.lower() for t in expanded_tags])

            # Also expand to TMDB keywords to catch genre-like terms
            expanded_kw = semantic_expander.expand_term(
                term, method='embedding', top_k=10,
                min_similarity=0.40, search_keywords=True
            )
            outcome_tags.update([t.lower() for t in expanded_kw])

    # Map common outcome terms to genres
    OUTCOME_TO_GENRE = {
        'cheer': ['comedy'], 'cheer up': ['comedy'], 'funny': ['comedy'],
        'laugh': ['comedy'], 'happy': ['comedy'], 'fun': ['comedy'],
        'feel good': ['comedy', 'family'], 'feel-good': ['comedy', 'family'],
        'uplifting': ['comedy', 'drama'], 'inspiring': ['drama'],
        'scary': ['horror'], 'frightening': ['horror'], 'terrifying': ['horror'],
        'thrilling': ['thriller'], 'suspense': ['thriller'],
        'romantic': ['romance'], 'love': ['romance'],
        'action': ['action'], 'exciting': ['action', 'adventure'],
        'adventure': ['adventure'], 'epic': ['adventure'],
        'sad': ['drama'], 'cry': ['drama'], 'emotional': ['drama'],
    }

    for term in outcome_seeds:
        term_lower = term.lower()
        for key, genres in OUTCOME_TO_GENRE.items():
            if key in term_lower:
                outcome_genres.update(genres)

    logger.info(f"   [SIT+OUT] Outcome seeds: {outcome_seeds}")
    logger.info(f"   [SIT+OUT] Outcome tags ({len(outcome_tags)}): {list(outcome_tags)[:15]}")
    logger.info(f"   [SIT+OUT] Outcome genres: {outcome_genres}")

    if not situation_keywords:
        logger.warning("   [SIT+OUT] No situation keywords found")
        return []

    if not outcome_tags and not outcome_genres:
        logger.warning("   [SIT+OUT] No outcome tags or genres found")
        return []

    # =================================================================
    # STEP 3: Filter movies - must match BOTH situation AND outcome
    # =================================================================
    def movie_matches_situation(row):
        """Check if movie matches situation via keywords or overview (with inflection variants)."""
        movie_keywords = row.get('keywords', None)
        movie_overview = row.get('overview', '')

        # Parse movie keywords
        movie_kw_lower = []
        if movie_keywords is not None:
            if isinstance(movie_keywords, float) and np.isnan(movie_keywords):
                pass
            elif isinstance(movie_keywords, (list, np.ndarray)) and len(movie_keywords) > 0:
                movie_kw_lower = [str(k).strip().lower() for k in movie_keywords]
            elif isinstance(movie_keywords, str):
                movie_kw_lower = [movie_keywords.strip().lower()]

        overview_lower = str(movie_overview).lower() if movie_overview else ''

        # Check situation keywords against movie keywords (exact/substring match)
        for sit_kw in situation_keywords:
            for movie_kw in movie_kw_lower:
                if sit_kw in movie_kw or movie_kw in sit_kw:
                    return True
            # Check overview for exact match (only for longer keywords to avoid false positives)
            if len(sit_kw) > 3 and sit_kw in overview_lower:
                return True

        # INFLECTION MATCHING: Check if any situation variant appears in overview
        # This catches "dumps" in overview when we have "dump" as keyword
        # Much faster than lemmatizing every overview - we just do string matching
        if situation_variants and overview_lower:
            # Split overview into words for word-boundary matching
            # This prevents "dump" from matching "dumpster" etc.
            import re
            overview_words = set(re.findall(r'\b\w+\b', overview_lower))
            if situation_variants & overview_words:  # Set intersection
                return True

        return False

    def movie_matches_outcome(row, movie_title):
        """Check if movie matches outcome via genre or zero-shot tags."""
        # Check genre
        genre_ids = row.get('genre_ids', None)
        if genre_ids is not None and not (isinstance(genre_ids, float) and np.isnan(genre_ids)):
            if isinstance(genre_ids, (list, np.ndarray)):
                movie_genres = [GENRE_ID_TO_NAME.get(int(gid), '').lower() for gid in genre_ids]
                for og in outcome_genres:
                    if og.lower() in movie_genres:
                        return True

        # Check zero-shot tags
        if zero_shot_tags_df is not None and len(outcome_tags) > 0:
            # Normalize title for lookup
            title_norm = movie_title.lower().strip()
            movie_row = zero_shot_tags_df[zero_shot_tags_df['title_norm'] == title_norm]

            if not movie_row.empty:
                for tag in outcome_tags:
                    tag_col = tag.replace('-', ' ').replace('_', ' ')
                    if tag_col in movie_row.columns:
                        if movie_row[tag_col].values[0] > 0:
                            return True

        # Fallback: check if any outcome tag appears in overview
        movie_overview = row.get('overview', '')
        overview_lower = str(movie_overview).lower() if movie_overview else ''
        for tag in outcome_tags:
            if len(tag) > 4 and tag in overview_lower:
                return True

        return False

    # Apply filtering - check ALL movies first, then limit by quality
    matching_movies = []
    for idx, row in movies_df.iterrows():
        movie_title = row.get('title', row.get('tmdb_title', ''))
        if not movie_title:
            continue

        # Must match BOTH situation AND outcome
        if movie_matches_situation(row) and movie_matches_outcome(row, movie_title):
            # Store title with popularity for sorting
            popularity = row.get('popularity', 0) or 0
            vote_avg = row.get('vote_average', 0) or 0
            matching_movies.append((movie_title, popularity, vote_avg))

    logger.info(f"   [SIT+OUT] Found {len(matching_movies)} movies matching BOTH situation AND outcome")

    # Sort by popularity (descending), then limit
    matching_movies.sort(key=lambda x: (x[1], x[2]), reverse=True)

    candidates = [m[0] for m in matching_movies[:max_candidates]]

    if len(matching_movies) > max_candidates:
        logger.info(f"   [SIT+OUT] Limited to top {max_candidates} by popularity")

    return candidates


@dataclass
class RecommendationResult:
    """Container for recommendation results."""
    query: str
    parsed_query: ParsedQuery
    recommendations: List[ScoredMovie]
    num_candidates: int
    entity_track: Optional[List[ScoredMovie]] = None  # Dual-track: entity-focused results
    mood_track: Optional[List[ScoredMovie]] = None    # Dual-track: mood-focused results
    dual_track_mode: bool = False                      # Whether dual-track was used

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'query': self.query,
            'parsed_filters': {
                'genres': self.parsed_query.genres,
                'decades': self.parsed_query.decades,
                'years': self.parsed_query.years,
                'moods': self.parsed_query.moods,
                'themes': self.parsed_query.themes,
                'keywords': self.parsed_query.keywords
            },
            'num_candidates': self.num_candidates,
            'recommendations': [r.to_dict() for r in self.recommendations]
        }

        if self.dual_track_mode:
            result['dual_track_mode'] = True
            result['entity_track'] = [r.to_dict() for r in self.entity_track] if self.entity_track else []
            result['mood_track'] = [r.to_dict() for r in self.mood_track] if self.mood_track else []

        return result
    
    def print_summary(self, top_n: int = 10):
        """Print human-readable summary."""
        print("\n" + "="*60)
        print(f"RECOMMENDATIONS FOR: '{self.query}'")
        print("="*60)

        print(f"\n📋 Parsed Query:")
        if self.parsed_query.genres:
            print(f"   Genres: {', '.join(self.parsed_query.genres)}")
        if self.parsed_query.decades:
            print(f"   Decades: {', '.join([f'{d}s' for d in self.parsed_query.decades])}")
        if self.parsed_query.years:
            print(f"   Years: {', '.join(map(str, self.parsed_query.years))}")
        if self.parsed_query.moods:
            print(f"   Moods: {', '.join(self.parsed_query.moods)}")
        if self.parsed_query.themes:
            print(f"   Themes: {', '.join(self.parsed_query.themes)}")
        if self.parsed_query.keywords:
            print(f"   Keywords: {', '.join(self.parsed_query.keywords)}")

        print(f"\n🎬 Evaluated {self.num_candidates} candidates")

        # DUAL-TRACK DISPLAY: Show both entity and mood tracks
        if self.dual_track_mode and self.entity_track and self.mood_track:
            print(f"\nDUAL-TRACK MODE: Showing both interpretations")
            print("="*60)

            # ENTITY TRACK
            print(f"\nENTITY TRACK ({len(self.entity_track)} results):")
            print("-" * 60)
            for i, movie in enumerate(self.entity_track, 1):
                print(f"{i}. {movie.movie_title} (Score: {movie.final_score:.3f})")

            # MOOD TRACK
            print(f"\nMOOD TRACK ({len(self.mood_track)} results):")
            print("-" * 60)
            for i, movie in enumerate(self.mood_track, 1):
                print(f"{i}. {movie.movie_title} (Score: {movie.final_score:.3f})")

        # SINGLE-TRACK DISPLAY: Standard recommendation list
        else:
            print(f"\n🏆 Top {min(top_n, len(self.recommendations))} Recommendations:")
            print("-" * 60)

            for i, movie in enumerate(self.recommendations[:top_n], 1):
                print(f"\n{i}. {movie.movie_title}")
                print(f"   Score: {movie.final_score:.3f}")
                print(f"   Signals: CF={movie.cf_score:.2f} Content={movie.content_score:.2f} "
                      f"Theme={movie.theme_score:.2f} Sentiment={movie.sentiment_score:.2f} "
                      f"Tag={movie.tag_score:.2f} Query={movie.query_score:.2f}")


class MovieRecommenderInteractiveV4:
    """
    Interactive hybrid movie recommendation system V4 - WITH CONCEPT COVERAGE

    This version adds concept coverage scoring to prevent single-concept matches
    from ranking too high. For queries like "ancient history revenge", movies must
    match MULTIPLE concepts (ancient/history AND revenge) to rank highly.
    """

    def __init__(self,
                 data_dir: str = "data/raw",
                 models_dir: str = "models",
                 config_path: Optional[str] = "src/config.yaml",
                 use_semantic_expansion: bool = True):
        """
        Initialize the complete recommendation system V2.

        Args:
            data_dir: Directory containing processed data
            models_dir: Directory containing models
            config_path: Path to config file
            use_semantic_expansion: Enable semantic tag matching (default: True)
        """
        logger.info("Initializing MovieRecommenderInteractiveV2 with semantic expansion...")

        # Store configuration
        self.use_semantic_expansion = use_semantic_expansion

        # Initialize components
        self.data_loader = DataLoader(data_dir, models_dir)
        self.query_parser = QueryParser()
        self.signal_fusion = SignalFusion(models_dir, data_dir)
        self.scorer = Scorer(config_path)
        self.zero_shot_integrator = ZeroShotIntegrator(data_dir)
        self.entity_extractor = EntityExtractor(f"{data_dir}/tmdb_fully_enriched.parquet")

        # Initialize BERT model for semantic expansion (if enabled)
        if self.use_semantic_expansion:
            logger.info("Loading BERT model for semantic expansion...")
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model.to(self.device)
            logger.info(f"   BERT model loaded on {self.device}")
        
        # Load data
        logger.info("Loading data...")
        self.movies = self.data_loader.get_movies()

        # Deduplicate by tmdb_id (some movies appear from multiple sources)
        if 'tmdb_id' in self.movies.columns:
            before_count = len(self.movies)
            self.movies = self.movies.drop_duplicates(subset=['tmdb_id'], keep='first')
            after_count = len(self.movies)
            if before_count > after_count:
                logger.info(f"   Deduplicated movies: {before_count} -> {after_count} (removed {before_count - after_count} duplicates)")

        # Fix data types and column names
        if not self.movies.empty:
            # Ensure 'year' is numeric
            if 'year' in self.movies.columns:
                self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')

            # Add Bayesian weighted rating to prevent low-vote bias
            if 'vote_average' in self.movies.columns and 'vote_count' in self.movies.columns:
                # Calculate global mean for Bayesian formula
                global_mean = self.movies['vote_average'].mean()
                if pd.isna(global_mean):
                    global_mean = 4.89  # TMDB typical mean

                # OPTIMIZATION: Vectorized Bayesian weighting (10-50x faster than apply+lambda)
                # Formula: WR = (v/(v+m)) * R + (m/(v+m)) * C
                # Where: v=vote_count, R=vote_average, m=min_votes, C=global_mean
                min_votes = 1000
                vote_avg = self.movies['vote_average'].fillna(global_mean)
                vote_cnt = self.movies['vote_count'].fillna(0)
                self.movies['weighted_rating'] = (
                    (vote_cnt / (vote_cnt + min_votes)) * vote_avg +
                    (min_votes / (vote_cnt + min_votes)) * global_mean
                )
                logger.info(f"   Added Bayesian weighted_rating column (mean: {global_mean:.2f}) [vectorized]")
            else:
                logger.warning("   vote_average or vote_count missing - skipping Bayesian weighting")

            # Create a 'title' column if it doesn't exist
            if 'title' not in self.movies.columns:
                if 'tmdb_title' in self.movies.columns:
                    self.movies['title'] = self.movies['tmdb_title']
                elif 'original_title' in self.movies.columns:
                    self.movies['title'] = self.movies['original_title']

            # OPTIMIZATION: Create title->metadata lookup for fast signal computation
            # Using to_dict('records') instead of iterrows() for 5-10x speedup
            self.title_to_metadata = {}
            records = self.movies.to_dict('records')
            for row in records:
                title = row.get('title', '')
                if title:  # Only add movies with valid titles
                    self.title_to_metadata[title] = {
                        'cast': row.get('cast', []),
                        'director': row.get('director', ''),
                        'keywords': row.get('keywords', []),
                        'genre_ids': row.get('genre_ids', []),
                        'vote_average': row.get('vote_average', 0),
                        'vote_count': row.get('vote_count', 0),
                        'weighted_rating': row.get('weighted_rating', row.get('vote_average', 0)),  # Bayesian weighted
                        'popularity': row.get('popularity', 0),
                        'overview': row.get('overview', '')
                    }
            logger.info(f"   Created metadata lookup for {len(self.title_to_metadata)} movies [optimized]")

        # Get rare tags for boosting
        self.rare_tags = self.zero_shot_integrator.get_rare_tags(threshold=100)

        # Initialize semantic tag expander for query expansion (with TMDB keywords!)
        unified_tags = self.zero_shot_integrator.merge_tags()
        if unified_tags is not None and not unified_tags.empty:
            all_tags = [col for col in unified_tags.columns if col != 'title_norm']

            # Load TMDB keywords for expanded coverage (43k movies vs 17k with tags)
            all_keywords = None
            try:
                import pickle
                from pathlib import Path
                keyword_list_path = Path(models_dir) / "keyword_list.pkl"
                if keyword_list_path.exists():
                    with open(keyword_list_path, 'rb') as f:
                        all_keywords = pickle.load(f)
                    logger.info(f"   Loaded {len(all_keywords)} TMDB keywords for semantic expansion")
                else:
                    logger.warning(f"   TMDB keywords not found at {keyword_list_path}")
                    logger.warning("   Run generate_keyword_embeddings.py for full 43k movie coverage")
            except Exception as e:
                logger.warning(f"   Could not load TMDB keywords: {e}")

            self.semantic_expander = SemanticTagExpander(all_tags, all_keywords, models_dir=models_dir)
            logger.info(f"   Initialized semantic tag expander with {len(all_tags)} tags" +
                       (f" and {len(all_keywords)} keywords" if all_keywords else ""))
        else:
            self.semantic_expander = None
            logger.warning("   Could not initialize semantic tag expander (no tags available)")

        logger.info("✅ MovieRecommender ready!")
        logger.info(f"   Loaded {len(self.movies)} movies")
        logger.info(f"   Identified {len(self.rare_tags)} rare tags for boosting")

    def _should_use_dual_track(self, parsed_query: ParsedQuery) -> str:
        """
        Detect if query is complex and should use dual-track recommendations.
        Now returns a MODE string instead of boolean to support multi-theme mode.

        LINE OF DEMARCATION:
        - SINGLE-TRACK (None): "Tom Cruise action movies from the 90s" (actor + genre + year)
        - ENTITY_MOOD: "Tom Cruise Drama movies from the 90s about lawyers" (actor + genre + year + theme)
        - MULTI_THEME: "Movies about trauma and recovery through humor" (no entity, multiple theme groups)

        Returns:
            None - single track mode
            "entity_mood" - entity + mood/theme dual track (current behavior)
            "multi_theme" - theme vs theme dual track (NEW)
        """
        # Check if has entities (actors, directors, studios)
        has_entities = (len(parsed_query.actors) > 0 or
                       len(parsed_query.directors) > 0 or
                       len(parsed_query.studios) > 0)

        # KEY ENTITY NOUNS that indicate entity_mood mode, not multi_theme
        # These are concrete nouns that should be treated as entity constraints
        KEY_ENTITY_NOUNS = {
            'female', 'male', 'woman', 'man', 'girl', 'boy',
            'dog', 'cat', 'animal', 'pet',
            'lawyer', 'doctor', 'cop', 'detective', 'soldier', 'teacher',
            'mother', 'father', 'daughter', 'son', 'wife', 'husband',
            'girlfriend', 'boyfriend', 'friend', 'family',
            'lead', 'protagonist', 'hero', 'heroine'
        }

        # Check if any content nouns are entity nouns
        content_nouns = getattr(parsed_query, 'content_nouns', [])
        has_entity_noun = any(noun.lower() in KEY_ENTITY_NOUNS for noun in content_nouns)

        # Check for multi-theme mode (no entities but 2+ theme groups detected)
        # BUT only if no entity nouns are present (otherwise use entity_mood)
        has_multi_theme = (not has_entities and
                          not has_entity_noun and
                          hasattr(parsed_query, 'theme_groups') and
                          len(parsed_query.theme_groups) >= 2)

        if has_multi_theme:
            # MULTI-THEME MODE: No entity, but multiple distinct theme groups
            logger.info(f"\n   MULTI-THEME MODE TRIGGERED:")
            logger.info(f"   - Theme Groups: {parsed_query.theme_groups}")
            logger.info(f"   - Content Nouns: {parsed_query.content_nouns}")
            return "multi_theme"

        # If we have entity nouns (like "lead", "female"), use entity_mood mode
        if has_entity_noun and not has_entities:
            logger.info(f"\n   ENTITY_MOOD MODE TRIGGERED (entity nouns detected):")
            logger.info(f"   - Entity Nouns: {[n for n in content_nouns if n.lower() in KEY_ENTITY_NOUNS]}")
            logger.info(f"   - Other Content Nouns: {[n for n in content_nouns if n.lower() not in KEY_ENTITY_NOUNS]}")
            return "entity_mood"

        # Get all entity name words (to exclude from descriptor count)
        entity_words = set()
        for actor in parsed_query.actors:
            entity_words.update(word.lower() for word in actor.split())
        for director in parsed_query.directors:
            entity_words.update(word.lower() for word in director.split())
        for studio in parsed_query.studios:
            entity_words.update(word.lower() for word in studio.split())

        # Count descriptive dimensions (excluding entity name words)
        descriptive_keywords = [kw for kw in parsed_query.keywords if kw.lower() not in entity_words]
        num_descriptors = len(descriptive_keywords) + len(parsed_query.themes) + len(parsed_query.moods)

        # TRIGGER 1: Has entities AND descriptors → entity + mood combination
        if has_entities and num_descriptors > 0:
            logger.info(f"\n   ENTITY_MOOD MODE TRIGGERED:")
            logger.info(f"   - Actors/Directors/Studios: {parsed_query.actors + parsed_query.directors + parsed_query.studios}")
            logger.info(f"   - Keywords: {parsed_query.keywords}")
            logger.info(f"   - Themes: {parsed_query.themes}")
            logger.info(f"   - Moods: {parsed_query.moods}")
            return "entity_mood"

        # TRIGGER 2: Multiple descriptors (>2) without entities → still use entity_mood
        # (backwards compatibility for complex theme queries that don't have 2+ groups)
        # EXCEPTION: Situational queries (adult_context=True) should use single-track
        # e.g., "my girlfriend broke up with me" - "girlfriend" is context, not a search term
        if num_descriptors > 2:
            # Check for situational query (breakup, relationship context)
            if hasattr(parsed_query, 'adult_context') and parsed_query.adult_context:
                logger.info(f"\n   SINGLE-TRACK MODE (situational query):")
                logger.info(f"   - Adult context detected (breakup/relationship situation)")
                logger.info(f"   - Using mood/theme matching only, not entity search")
                return None  # Single track - don't search for "girlfriend" as entity

            logger.info(f"\n   ENTITY_MOOD MODE TRIGGERED (complex descriptors):")
            logger.info(f"   - {num_descriptors} descriptors detected")
            logger.info(f"   - Keywords: {parsed_query.keywords}")
            logger.info(f"   - Themes: {parsed_query.themes}")
            return "entity_mood"

        # Single track mode
        return None

    def _classify_entity_keywords(self, keywords: List[str]) -> tuple:
        """
        Classify keywords into entity keywords (concrete nouns) vs. mood/context keywords.
        Uses spaCy POS tagging for universal classification (works for ANY query).

        Entity keywords: NOUNS (concrete, matchable things: dog, lawyer, football, girlfriend)
        Mood keywords: VERBS, ADJECTIVES, PRONOUNS, ADVERBS (actions, states, qualities)

        Args:
            keywords: List of keywords from parsed query

        Returns:
            (entity_keywords, mood_keywords) tuple
        """
        # Blacklist: words that should ALWAYS be mood (even if POS says noun)
        # These are common stop words, request words, and abstract concepts
        MOOD_BLACKLIST = {
            # Pronouns
            'you', 'me', 'my', 'your', 'his', 'her', 'him', 'them', 'their', 'our', 'we', 'i',
            # Request verbs
            'recommend', 'help', 'need', 'want', 'looking', 'find', 'show', 'give', 'suggest',
            # States/emotions
            'died', 'dying', 'dead', 'sick', 'depressed', 'sad', 'happy', 'angry', 'scared',
            # Qualities
            'good', 'bad', 'best', 'worst', 'great', 'terrible', 'big', 'small', 'new', 'old',
            # Temporal
            'today', 'tomorrow', 'yesterday', 'tonight', 'morning', 'afternoon', 'evening', 'night',
            # Verbs (common ones that might slip through)
            'feeling', 'watching', 'seeing', 'making', 'doing', 'going', 'coming',
            'forget', 'remember', 'think', 'know', 'believe', 'hope', 'wish',
            'watch', 'make', 'have', 'get', 'take', 'give', 'tell',
            # Abstract concepts
            'time', 'day', 'week', 'month', 'year', 'thing', 'stuff', 'way', 'place'
        }

        entity_keywords = []
        mood_keywords = []

        # KEY NOUNS that should ALWAYS be treated as entity (for NOUN ENTITY BOOST)
        # These are concepts where movie keywords MUST contain them
        KEY_ENTITY_NOUNS = {
            'female', 'male', 'woman', 'man', 'girl', 'boy',
            'dog', 'cat', 'animal', 'pet',
            'lawyer', 'doctor', 'cop', 'detective', 'soldier', 'teacher',
            'mother', 'father', 'daughter', 'son', 'wife', 'husband',
            'girlfriend', 'boyfriend', 'friend', 'family',
            'lead', 'protagonist', 'hero', 'heroine'
        }

        # Use spaCy for POS tagging
        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Check blacklist first
            if keyword_lower in MOOD_BLACKLIST:
                mood_keywords.append(keyword)
                continue

            # SPECIAL: For multi-word phrases, extract KEY NOUNS
            # "strong female lead" → extract 'female' and 'lead' as entity nouns
            if ' ' in keyword_lower:
                words = keyword_lower.split()
                has_key_noun = any(word in KEY_ENTITY_NOUNS for word in words)
                if has_key_noun:
                    # Extract the key nouns from the phrase
                    for word in words:
                        if word in KEY_ENTITY_NOUNS:
                            entity_keywords.append(word)
                    continue  # Don't add the full phrase, just the nouns
                else:
                    # Multi-word without key noun → mood
                    mood_keywords.append(keyword)
                    continue

            # Check if single word is a key entity noun
            if keyword_lower in KEY_ENTITY_NOUNS:
                entity_keywords.append(keyword)
                continue

            # Use spaCy POS tagging for other words
            try:
                doc = self.entity_extractor.nlp(keyword)
                if len(doc) == 0:
                    # Empty doc, default to mood
                    mood_keywords.append(keyword)
                    continue

                # Get the main token's POS tag
                main_token = doc[0]
                pos = main_token.pos_

                # NOUN or PROPN (proper noun) → entity
                if pos in ['NOUN', 'PROPN']:
                    entity_keywords.append(keyword)
                # VERB, ADJ, ADV, PRON, DET → mood
                elif pos in ['VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', 'SCONJ', 'AUX']:
                    mood_keywords.append(keyword)
                # NUM, X, SYM → mood (numbers, unknown, symbols)
                elif pos in ['NUM', 'X', 'SYM']:
                    mood_keywords.append(keyword)
                else:
                    # Fallback: default to entity (be conservative)
                    entity_keywords.append(keyword)

            except Exception as e:
                # If POS tagging fails, use length heuristic
                logger.warning(f"POS tagging failed for '{keyword}': {e}")
                if len(keyword_lower) >= 4:
                    entity_keywords.append(keyword)
                else:
                    mood_keywords.append(keyword)

        logger.info(f"   [ENTITY CLASSIFICATION - POS-BASED]")
        logger.info(f"      Entity keywords (NOUNS): {entity_keywords}")
        logger.info(f"      Mood keywords (VERBS/ADJ/etc): {mood_keywords}")

        return entity_keywords, mood_keywords

    def _compute_mood_theme_score(self, movie_title: str, mood_themes: List[str]) -> float:
        """
        Compute a score based on how many mood themes (e.g., 'uplifting', 'comforting')
        match the movie's zero-shot tags.

        Args:
            movie_title: Movie title to check
            mood_themes: List of mood theme keywords (e.g., ['uplifting', 'comforting', 'feel-good'])

        Returns:
            Score 0-1 based on proportion of mood themes matched
        """
        if not mood_themes:
            return 0.0

        # Access zero-shot tags DataFrame
        if not hasattr(self.signal_fusion, 'zero_shot_tags') or self.signal_fusion.zero_shot_tags is None:
            return 0.0

        try:
            # Find movie row - try title normalization variations
            movie_row = self.signal_fusion.zero_shot_tags[
                self.signal_fusion.zero_shot_tags['title_norm'] == movie_title.lower()
            ]

            # If not found, try without special characters
            if len(movie_row) == 0:
                normalized = movie_title.lower().replace("'", "").replace(":", "").replace("-", " ")
                movie_row = self.signal_fusion.zero_shot_tags[
                    self.signal_fusion.zero_shot_tags['title_norm'] == normalized
                ]

            if len(movie_row) == 0:
                # Movie not in zero-shot tags - return 0
                return 0.0

            # Count how many mood themes this movie matches
            matches = 0
            for theme in mood_themes:
                # Normalize theme for matching (replace hyphens with spaces)
                theme_normalized = theme.replace("-", " ")

                # Check if this theme is a column in zero-shot tags
                if theme_normalized in movie_row.columns and movie_row[theme_normalized].values[0] > 0:
                    matches += 1

            # Normalize by number of mood themes
            score = matches / len(mood_themes)
            return float(score)

        except Exception as e:
            logger.debug(f"Mood theme score error for {movie_title}: {e}")
            return 0.0

    def _compute_theme_group_score(self, movie_title: str, theme_group: List[str]) -> float:
        """
        Compute a score based on how well a movie matches a theme group.
        Used for MULTI-THEME MODE to score movies against specific theme concepts.

        Uses multiple signals:
        1. Zero-shot tags (direct theme matching)
        2. TMDB keywords (semantic matching)
        3. Overview text (keyword presence)

        Args:
            movie_title: Movie title to check
            theme_group: List of theme keywords (e.g., ['trauma', 'recovery'] or ['humor'])

        Returns:
            Score 0-1+ based on theme relevance (can exceed 1.0 for strong matches)
        """
        if not theme_group:
            return 0.0

        score = 0.0
        theme_lower = [t.lower() for t in theme_group]

        # 1. Check zero-shot tags
        if hasattr(self.signal_fusion, 'zero_shot_tags') and self.signal_fusion.zero_shot_tags is not None:
            try:
                movie_row = self.signal_fusion.zero_shot_tags[
                    self.signal_fusion.zero_shot_tags['title_norm'] == movie_title.lower()
                ]
                if len(movie_row) == 0:
                    normalized = movie_title.lower().replace("'", "").replace(":", "").replace("-", " ")
                    movie_row = self.signal_fusion.zero_shot_tags[
                        self.signal_fusion.zero_shot_tags['title_norm'] == normalized
                    ]

                if len(movie_row) > 0:
                    # Check each theme in the group
                    for theme in theme_lower:
                        # Check direct column match
                        if theme in movie_row.columns and movie_row[theme].values[0] > 0:
                            score += 0.4  # Strong signal from zero-shot tag

                        # Check partial matches (e.g., "trauma" might match "traumatic")
                        for col in movie_row.columns:
                            if theme in col.lower() and movie_row[col].values[0] > 0:
                                score += 0.2
                                break
            except Exception as e:
                logger.debug(f"Zero-shot check error for {movie_title}: {e}")

        # 2. Check TMDB keywords
        try:
            movie_data = self.movies[self.movies['title'] == movie_title]
            if len(movie_data) > 0:
                keywords = movie_data.iloc[0].get('keywords', None)
                if keywords is not None:
                    if isinstance(keywords, str):
                        keywords_lower = keywords.lower()
                    elif isinstance(keywords, (list, np.ndarray)):
                        keywords_lower = ' '.join(str(k).lower() for k in keywords)
                    else:
                        keywords_lower = ''

                    for theme in theme_lower:
                        if theme in keywords_lower:
                            score += 0.3  # Good signal from TMDB keyword

                # 2b. Check genre_ids (handles plural/singular like "comedies" -> genre_id 35)
                CONCEPT_TO_GENRE_IDS = {
                    'thriller': [53], 'thrillers': [53],
                    'horror': [27], 'comedy': [35], 'comedies': [35],
                    'drama': [18], 'dramas': [18], 'action': [28],
                    'romance': [10749], 'romantic': [10749], 'romances': [10749],
                    'sci-fi': [878], 'science fiction': [878],
                    'fantasy': [14], 'mystery': [9648], 'crime': [80],
                    'war': [10752], 'western': [37], 'animation': [16],
                    'animated': [16], 'adventure': [12], 'documentary': [99],
                    'family': [10751], 'history': [36], 'historical': [36],
                }
                genre_ids = movie_data.iloc[0].get('genre_ids', None)
                if genre_ids is not None:
                    if isinstance(genre_ids, (list, np.ndarray)):
                        genre_ids_list = [int(gid) for gid in genre_ids]
                    else:
                        genre_ids_list = []

                    for theme in theme_lower:
                        if theme in CONCEPT_TO_GENRE_IDS:
                            expected_gids = CONCEPT_TO_GENRE_IDS[theme]
                            if any(gid in genre_ids_list for gid in expected_gids):
                                score += 0.5  # Strong signal from genre match

                # 3. Check overview text
                overview = movie_data.iloc[0].get('overview', '')
                if overview and isinstance(overview, str):
                    overview_lower = overview.lower()
                    for theme in theme_lower:
                        if theme in overview_lower:
                            score += 0.15  # Moderate signal from overview
        except Exception as e:
            logger.debug(f"TMDB/overview check error for {movie_title}: {e}")

        # Normalize by theme group size (more themes = harder to match all)
        # But don't penalize too heavily - partial matches are still valuable
        normalized_score = score / (1 + 0.2 * len(theme_group))

        return normalized_score

    def _apply_concept_coverage_penalty(self, scored_movies: List[ScoredMovie],
                                       concept_groups: Dict[str, List[str]],
                                       df: pd.DataFrame) -> List[ScoredMovie]:
        """
        Apply concept coverage multiplier to query_score.

        For multi-concept queries, movies must match MULTIPLE concepts to rank high.
        - "Sudden Impact" (revenge only) = 1/3 concepts → multiplier = 0.14
        - "Gladiator" (ancient + history + revenge) = 3/3 concepts → multiplier = 1.0

        Args:
            scored_movies: Movies with scores from signal fusion
            concept_groups: Dict mapping concept to expanded keywords
            df: Full movie dataframe for metadata lookup

        Returns:
            Movies with adjusted query_scores
        """
        if not concept_groups or len(concept_groups) <= 1:
            # Single concept query or no concepts - skip penalty
            return scored_movies

        total_concepts = len(concept_groups)
        power = 1.4  # Penalty strength (1.4 = balanced, 1.75 = strong, 2.0 = very strong)

        logger.info(f"\n[V4 CONCEPT COVERAGE] Applying penalty for {total_concepts}-concept query")
        logger.info(f"   Power: {power}, Concepts: {list(concept_groups.keys())}")

        for movie in scored_movies:
            # Find movie in dataframe
            movie_row = df[df['tmdb_title'] == movie.movie_title]
            if movie_row.empty:
                continue

            movie_data = movie_row.iloc[0]

            # Check which concepts this movie matches
            concepts_matched = 0
            matched_concept_names = []

            for concept_name, concept_keywords in concept_groups.items():
                # Check if ANY keyword from this concept matches the movie
                movie_matches_concept = False

                # Check TMDB keywords
                movie_keywords = movie_data.get('keywords', [])
                if isinstance(movie_keywords, (list, np.ndarray)) and len(movie_keywords) > 0:
                    keywords_str = " ".join([str(kw).lower() for kw in movie_keywords])
                    for kw in concept_keywords:
                        if kw.lower() in keywords_str:
                            movie_matches_concept = True
                            break

                # Check overview if no keyword match
                if not movie_matches_concept:
                    overview = movie_data.get('overview', '')
                    if overview and isinstance(overview, str):
                        overview_lower = overview.lower()
                        for kw in concept_keywords:
                            if kw.lower() in overview_lower:
                                movie_matches_concept = True
                                break

                if movie_matches_concept:
                    concepts_matched += 1
                    matched_concept_names.append(concept_name)

            # Calculate coverage multiplier
            coverage_ratio = concepts_matched / total_concepts
            multiplier = coverage_ratio ** power

            # Apply multiplier to query_score
            original_query_score = movie.query_score
            movie.query_score = original_query_score * multiplier

            # Recalculate final_score (using accuracy mode weights)
            movie.final_score = (
                0.60 * movie.query_score +
                0.15 * movie.tag_score +
                0.10 * movie.theme_score +
                0.08 * movie.content_score +
                0.05 * movie.cf_score +
                0.02 * movie.sentiment_score
            )

            # Log significant penalties
            if multiplier < 0.8:
                logger.info(f"   [PENALTY] {movie.movie_title}: {concepts_matched}/{total_concepts} concepts " +
                          f"({matched_concept_names}) → multiplier={multiplier:.2f}, " +
                          f"query_score {original_query_score:.3f}→{movie.query_score:.3f}")

        # Re-sort by new final_score
        scored_movies.sort(key=lambda x: x.final_score, reverse=True)

        return scored_movies

    def _get_candidate_movies(self, parsed_query: ParsedQuery,
                              max_candidates: int = 2000):
        """
        Get candidate movies using INVERTED FUNNEL architecture.

        NEW STRATEGY (Production-Ready):
        1. Start with BROAD TMDB-based filtering (genre, year, actor, etc.)
        2. Return 2000-5000 candidates (much larger pool than before)
        3. Let ALL 6 signals vote on these candidates
        4. Zero-shot tags used as BOOST signal, not FILTER

        This ensures movies without zero-shot tags can still rank well via other signals.

        Args:
            parsed_query: Parsed query object
            max_candidates: Maximum number of candidates (default: 2000)

        Returns:
            Tuple of (candidate movie titles, actor_filter_applied flag, curated_list_used flag, expanded_tags, concept_groups, keyword_search_used)
        """
        candidates = self.movies.copy()
        actor_filter_applied = False
        concept_groups = None
        keyword_search_used = False  # Track if we found discriminating keywords

        # STEP 1: Create concept_groups for scoring (if query has themes/keywords)
        if parsed_query.themes or parsed_query.keywords:
            query_tags = parsed_query.themes + parsed_query.keywords

            if self.semantic_expander:
                # Get original concepts for grouping
                if hasattr(parsed_query, 'expanded_categories') and hasattr(parsed_query, 'original_themes'):
                    original_concepts = list(parsed_query.expanded_categories)
                    original_concepts.extend(parsed_query.original_themes)
                    original_concepts.extend(parsed_query.original_keywords)
                    original_concepts = list(dict.fromkeys(original_concepts))
                    concept_terms = original_concepts
                else:
                    concept_terms = query_tags

                # CRITICAL FIX: Exclude genre terms from concept groups
                # Genres are already filtered in candidate selection (OR logic)
                # Including them in concept coverage creates unfair AND-logic penalties
                # E.g., "thrillers OR action about animals" shouldn't penalize non-thrillers
                genre_terms = {'action', 'adventure', 'animation', 'comedy', 'crime',
                              'drama', 'fantasy', 'horror', 'romance', 'sci-fi', 'science fiction',
                              'thriller', 'thrillers', 'western', 'war', 'mystery', 'mysteries'}

                content_concepts = [term for term in concept_terms if term.lower() not in genre_terms]

                if len(content_concepts) != len(concept_terms):
                    filtered_genres = [term for term in concept_terms if term.lower() in genre_terms]
                    logger.info(f"   Excluded genre terms from concept groups: {filtered_genres}")
                    logger.info(f"   Content concepts for AND-logic scoring: {content_concepts}")

                concept_terms = content_concepts if content_concepts else concept_terms

                # FILTER OUT USELESS WORDS from concept terms (stop words, filler words)
                useless_words = {'just', 'need', 'what', 'you', 'me', 'my', 'and', 'the', 'a', 'an',
                               'to', 'from', 'with', 'about', 'for', 'of', 'in', 'on', 'at', 'by',
                               'make', 'recommend', 'would', 'can', 'could', 'should', 'movie', 'movies', 'film', 'films'}
                filtered_concept_terms = [term for term in concept_terms if term.lower() not in useless_words and len(term) > 2]

                if not filtered_concept_terms:
                    # All terms were filtered out - use original
                    filtered_concept_terms = concept_terms

                # Create concept groups for scoring (only content concepts, not genres)
                concept_groups = self.semantic_expander.expand_query_terms_grouped(
                    filtered_concept_terms,
                    method='hybrid'
                )
                logger.info(f"   Created concept groups for scoring: {list(concept_groups.keys())}")

        # CONTEXT-AWARE KEYWORD FILTERING
        # Remove negative context words when query has positive themes/moods
        positive_themes = {'uplifting', 'feel-good', 'comforting', 'heartwarming', 'inspiring',
                          'hope', 'happy', 'joy', 'cheerful', 'funny', 'lighthearted', 'optimistic'}
        negative_context_words = {'died', 'dead', 'death', 'dying', 'broke', 'lost', 'losing',
                                 'broken', 'sad', 'depressed', 'upset'}

        has_positive_themes = any(theme.lower() in positive_themes for theme in parsed_query.themes)

        if has_positive_themes and parsed_query.keywords:
            # Filter out negative context words AND stop words from keywords
            stop_words = {'you', 'me', 'my', 'recommend', 'would', 'can', 'could', 'should',
                         'what', 'make', 'help', 'need', 'want', 'show', 'give'}

            original_keywords = parsed_query.keywords.copy()
            parsed_query.keywords = [
                kw for kw in parsed_query.keywords
                if kw.lower() not in negative_context_words and kw.lower() not in stop_words
            ]

            if len(parsed_query.keywords) < len(original_keywords):
                filtered_out = set(original_keywords) - set(parsed_query.keywords)
                logger.info(f"   [CONTEXT FILTER] Removed negative/stop words from keywords: {filtered_out}")
                logger.info(f"   [CONTEXT FILTER] Remaining keywords: {parsed_query.keywords}")

        # STEP 1.5: DISCRIMINATING TERM SEARCH (NEW - prioritizes rare terms)
        # For queries with rare/specific themes (political, biography, etc.), search TMDB metadata FIRST
        # This ensures movies with perfect keyword matches aren't filtered out by genre-only filtering
        discriminating_candidates = None
        expanded_keywords_for_scoring = None  # Will store expanded keywords to pass to scoring
        if parsed_query.themes or parsed_query.keywords:
            # Identify rare/discriminating terms (exclude common genres)
            common_genres = {'action', 'comedy', 'drama', 'horror', 'thriller', 'romance',
                           'sci-fi', 'fantasy', 'western', 'crime', 'mystery', 'war', 'adventure'}
            # PHASE 3: Collect entity name words to exclude from semantic expansion
            # This prevents "Bill Murray" from expanding to "billionaire", "bill clinton", "hillbilly"
            entity_words = set()
            if parsed_query.actors:
                for actor in parsed_query.actors:
                    entity_words.update(word.lower() for word in actor.split())
                logger.info(f"   [PHASE 3] Excluding actor name words from expansion: {entity_words}")
            if parsed_query.directors:
                for director in parsed_query.directors:
                    entity_words.update(word.lower() for word in director.split())
                logger.info(f"   [PHASE 3] Excluding director name words from expansion: {entity_words}")
            if parsed_query.studios:
                for studio in parsed_query.studios:
                    entity_words.update(word.lower() for word in studio.split())
                logger.info(f"   [PHASE 3] Excluding studio name words from expansion: {entity_words}")

            # PRIORITY FIX: Process keywords FIRST (concrete entity terms), then themes (abstract mood terms)
            # This ensures entity keywords like 'dog' get semantic expansion priority over mood keywords like 'uplifting'
            discriminating_terms = []
            seen_terms = set()  # Track what we've already added

            # EXTRACT ORIGINAL CONCEPTS from raw query (before query parser expansion)
            # This ensures we score based on user's actual concepts, not expanded mood terms
            FILLER_WORDS = {'a', 'an', 'the', 'with', 'from', 'to', 'for', 'of', 'in', 'on', 'at',
                          'by', 'and', 'or', 'strong', 'good', 'best', 'great', 'really',
                          'want', 'like', 'love', 'need', 'looking', 'recommend', 'show',
                          'movie', 'movies', 'film', 'films'}

            # COMPOUND PHRASES: Keep these as single concepts (don't split on space)
            # E.g., "female lead" is ONE concept, not two separate concepts
            COMPOUND_PHRASES = {
                # ORIGINAL - Character/Role
                'female lead': 'female lead',
                'strong female': 'female lead',
                'female protagonist': 'female lead',
                'woman lead': 'female lead',
                'male lead': 'male lead',
                'strong male': 'male lead',

                # ORIGINAL - Story Type
                'true story': 'true story',
                'based on true': 'true story',
                'real events': 'true story',
                'true events': 'true story',
                'world war': 'world war',
                'serial killer': 'serial killer',
                'time travel': 'time travel',
                'sci fi': 'sci-fi',
                'science fiction': 'sci-fi',

                # EMOTIONAL / PSYCHOLOGICAL
                'broke up': 'broke up',
                'break up': 'broke up',
                'broken up': 'broke up',
                'breaking up': 'broke up',
                'broken home': 'broken home',
                'identity crisis': 'identity crisis',
                'emotional breakdown': 'emotional breakdown',
                'midlife crisis': 'midlife crisis',
                'healing journey': 'healing journey',
                'coming to terms': 'coming to terms',
                'overcoming grief': 'overcoming grief',
                'inner demons': 'inner demons',
                'buried trauma': 'buried trauma',
                'seeking redemption': 'seeking redemption',
                'hitting rock bottom': 'rock bottom',
                'rock bottom': 'rock bottom',
                'letting go': 'letting go',
                'cheer up': 'cheer up',
                'cheer me up': 'cheer up',
                'feel better': 'feel better',

                # RELATIONSHIP & SOCIAL
                'toxic relationship': 'toxic relationship',
                'dysfunctional family': 'dysfunctional family',
                'found family': 'found family',
                'strained marriage': 'strained marriage',
                'troubled friendship': 'troubled friendship',
                'unlikely friendship': 'unlikely friendship',
                'unrequited love': 'unrequited love',
                'forbidden love': 'forbidden love',
                'love triangle': 'love triangle',
                'falling out': 'falling out',
                'custody battle': 'custody battle',
                'chosen family': 'found family',

                # PLOT / CONFLICT
                'on the run': 'on the run',
                'wrongfully accused': 'wrongfully accused',
                'personal vendetta': 'personal vendetta',
                'political intrigue': 'political intrigue',
                'government conspiracy': 'government conspiracy',
                'moral dilemma': 'moral dilemma',
                'family secret': 'family secret',
                'hidden agenda': 'hidden agenda',
                'long buried past': 'buried past',
                'buried past': 'buried past',
                'high stakes': 'high stakes',
                'high stakes gamble': 'high stakes',
                'catastrophic event': 'catastrophic event',
                'unraveling mystery': 'unraveling mystery',
                'small town mystery': 'small town mystery',
                'small town': 'small town',
                'revenge mission': 'revenge mission',

                # ACTION / THRILLER
                'undercover mission': 'undercover mission',
                'heist gone wrong': 'heist gone wrong',
                'prison break': 'prison break',
                'hostage situation': 'hostage situation',
                'survival story': 'survival story',
                'lone survivor': 'lone survivor',
                'post apocalyptic': 'post-apocalyptic',
                'post-apocalyptic': 'post-apocalyptic',
                'last man standing': 'last man standing',
                'parallel universe': 'parallel universe',
                'alien invasion': 'alien invasion',

                # HUMOR / VIBE / ARCHETYPES
                'fish out of water': 'fish out of water',
                'coming of age': 'coming of age',
                'dark comedy': 'dark comedy',
                'feel good': 'feel-good',
                'feel-good': 'feel-good',
                'over the top': 'over the top',
                'odd couple': 'odd couple',
                'lovable loser': 'lovable loser',
                'reluctant hero': 'reluctant hero',

                # VIEWER INTENT / EXPERIENCE
                'edge of seat': 'edge of seat',
                'edge of my seat': 'edge of seat',
                'slow burn': 'slow burn',
                'comfort movie': 'comfort movie',
                'mind bending': 'mind-bending',
                'mind-bending': 'mind-bending',
                'plot twist': 'plot twist',
                'gut punch': 'gut punch',
                'heart warming': 'heartwarming',
                'heartwarming': 'heartwarming',
                'tear jerker': 'tear jerker',
            }

            # REDUNDANT CONCEPTS: Map multiple words to the same concept
            # This prevents "female" and "lead" from counting as separate concepts
            REDUNDANT_CONCEPTS = {
                'lead': 'female lead',  # If we have "female lead", "lead" alone is redundant
                'protagonist': 'female lead',
                'heroine': 'female lead',
            }

            # Use raw_query which has the original query text
            raw_query = getattr(parsed_query, 'raw_query', None) or ''
            raw_query_lower = raw_query.lower()

            # STEP 1: Extract compound phrases first
            # Use word boundary matching to avoid "female" matching "male lead"
            import re
            extracted_compounds = set()
            for phrase, canonical in COMPOUND_PHRASES.items():
                # Use word boundary regex to avoid partial matches
                # E.g., "male lead" should NOT match inside "female lead"
                pattern = r'\b' + re.escape(phrase) + r'\b'
                if re.search(pattern, raw_query_lower):
                    extracted_compounds.add(canonical)
                    logger.info(f"   [COMPOUND PHRASE] Found '{phrase}' -> canonical '{canonical}'")

            # STEP 2: Extract remaining single words (excluding words from compound phrases)
            compound_words = set()
            for phrase in extracted_compounds:
                compound_words.update(phrase.split())
            # Also exclude words that would be redundant
            for word, canonical in REDUNDANT_CONCEPTS.items():
                if canonical in extracted_compounds:
                    compound_words.add(word)

            original_query_words = []
            for w in raw_query.split():
                w_lower = w.lower()
                if (w_lower not in FILLER_WORDS and
                    len(w) >= 3 and
                    w_lower not in compound_words):
                    original_query_words.append(w_lower)

            # STEP 3: Combine compound phrases + remaining words
            original_query_words = list(extracted_compounds) + original_query_words
            # Remove duplicates while preserving order
            seen = set()
            original_query_words = [x for x in original_query_words if not (x in seen or seen.add(x))]

            # MULTI-THEME OVERRIDE: Use content_nouns instead of raw keywords
            # This removes junk words like "that", "deal", "through" for theme-only queries
            # ONLY applies to multi-theme queries (no entities, 2+ theme groups)
            is_multi_theme = (
                len(parsed_query.actors) == 0 and
                len(parsed_query.directors) == 0 and
                len(parsed_query.studios) == 0 and
                hasattr(parsed_query, 'theme_groups') and
                len(parsed_query.theme_groups) >= 2
            )
            if is_multi_theme and hasattr(parsed_query, 'content_nouns') and parsed_query.content_nouns:
                logger.info(f"   [MULTI-THEME] Overriding query words with content_nouns")
                logger.info(f"   [MULTI-THEME] Before: {original_query_words}")
                original_query_words = [n.lower() for n in parsed_query.content_nouns]
                logger.info(f"   [MULTI-THEME] After: {original_query_words}")

            logger.info(f"   [CONCEPT SCORING] Original query concepts: {original_query_words}")

            concept_to_expansions = {}  # Maps concept -> list of expansions (for concept-based scoring)

            # PRIORITY 1: Keywords (concrete, entity-related terms from the query)
            # MULTI-THEME FIX: Use content_nouns instead of raw keywords for multi-theme queries
            # This prevents junk words like "that", "deal", "through" from polluting candidate filtering
            keywords_source = parsed_query.keywords
            if is_multi_theme and hasattr(parsed_query, 'content_nouns') and parsed_query.content_nouns:
                keywords_source = parsed_query.content_nouns
                logger.info(f"   [MULTI-THEME] Using content_nouns for candidate filtering: {keywords_source}")

            keyword_discriminating = []
            for term in keywords_source:
                term_lower = term.lower()
                # Skip entity name words (e.g., "bill", "murray" from "Bill Murray")
                if term_lower in entity_words:
                    logger.info(f"   [PHASE 3] Skipping entity word '{term_lower}' from expansion")
                    continue
                # Skip common genres, short words, and filler words
                if (term_lower not in common_genres and
                    len(term) >= 3 and
                    term_lower not in {'movie', 'film', 'movies', 'films', 'based', 'true', 'real', 'good', 'best'}):
                    if term_lower not in seen_terms:
                        keyword_discriminating.append(term_lower)
                        seen_terms.add(term_lower)

            # PRIORITY 2: Themes (mood, tone, genre-related terms)
            # NOTE: We NO LONGER filter out common_genres because we now ADD genres to themes
            # in query_parser.py (lines 383-386), so filtering them here would undo that fix
            theme_discriminating = []
            for term in parsed_query.themes:
                term_lower = term.lower()
                # Skip entity name words
                if term_lower in entity_words:
                    continue
                # Skip short words and filler words (but NOT genres!)
                if (len(term) >= 3 and
                    term_lower not in {'movie', 'film', 'movies', 'films', 'based', 'true', 'real', 'good', 'best'}):
                    if term_lower not in seen_terms:
                        theme_discriminating.append(term_lower)
                        seen_terms.add(term_lower)

            # COMBINE: Keywords first, then themes
            discriminating_terms = keyword_discriminating + theme_discriminating

            # PRIORITY 3: Multi-word phrases (always discriminating, append at end)
            all_query_terms = parsed_query.keywords + parsed_query.themes
            multi_word_terms = [t for t in all_query_terms if ' ' in t]
            for term in multi_word_terms:
                term_lower = term.lower()
                if term_lower not in seen_terms:
                    discriminating_terms.append(term_lower)
                    seen_terms.add(term_lower)

            logger.info(f"   [PRIORITY] {len(keyword_discriminating)} keywords prioritized, {len(theme_discriminating)} themes added")

            if discriminating_terms and len(discriminating_terms) >= 1:
                logger.info(f"   [KEYWORD SEARCH] Found {len(discriminating_terms)} discriminating terms: {discriminating_terms[:5]}")

                # DUAL-TRACK KEYWORD SEPARATION
                # For dual-track queries, separate entity keywords from mood keywords
                # This ensures entity track focuses on concrete nouns (dog, lawyer)
                # while mood track uses abstract context (died, recommend, uplifting)
                entity_discriminating = None
                mood_discriminating = None

                # Determine if dual-track should be used (returns None, "entity_mood", or "multi_theme")
                dual_track_mode = self._should_use_dual_track(parsed_query)

                if dual_track_mode and keyword_discriminating:
                    # Classify keywords into entity vs. mood
                    entity_kws, mood_kws = self._classify_entity_keywords(keyword_discriminating)

                    # PRIORITIZE: Sort themes so noun-entity themes come first
                    # This ensures "strong female lead", "female protagonist" get expanded
                    # before abstract mood terms like "bittersweet", "depression"
                    KEY_ENTITY_NOUNS = {
                        'female', 'male', 'woman', 'man', 'girl', 'boy',
                        'dog', 'cat', 'animal', 'pet',
                        'lawyer', 'doctor', 'cop', 'detective', 'soldier', 'teacher',
                        'mother', 'father', 'daughter', 'son', 'wife', 'husband',
                        'girlfriend', 'boyfriend', 'friend', 'family',
                        'lead', 'protagonist', 'hero', 'heroine'
                    }

                    def has_entity_noun(term):
                        words = term.lower().split()
                        return any(w in KEY_ENTITY_NOUNS for w in words)

                    noun_themes = [t for t in theme_discriminating if has_entity_noun(t)]
                    other_themes = [t for t in theme_discriminating if not has_entity_noun(t)]
                    prioritized_themes = noun_themes + other_themes

                    # For dual-track:
                    # - Entity track: entity keywords + themes (genres are concrete subjects!)
                    # - Mood track: mood keywords + themes (for atmospheric matching)
                    entity_discriminating = (entity_kws + prioritized_themes) if entity_kws else prioritized_themes
                    mood_discriminating = mood_kws + theme_discriminating if mood_kws else theme_discriminating

                    logger.info(f"   [DUAL-TRACK SEPARATION]")
                    logger.info(f"      Entity track will use: {entity_discriminating}")
                    logger.info(f"      Mood track will use: {mood_discriminating}")

                # SEMANTIC EXPANSION: Expand discriminating terms to related keywords
                # For dual-track: expand ONLY entity keywords (not mood keywords)
                expanded_keywords_set = set()
                if self.semantic_expander and hasattr(self.semantic_expander, 'all_keywords') and self.semantic_expander.all_keywords:
                    # Determine which terms to expand based on mode
                    if dual_track_mode and entity_discriminating is not None:
                        # DUAL-TRACK: Expand only entity keywords
                        # Limit increased to 15 (from 10) to ensure noun-entity terms get processed
                        terms_to_expand = entity_discriminating[:15]
                        logger.info(f"   [SEMANTIC EXPANSION - DUAL TRACK] Expanding {len(terms_to_expand)} ENTITY keywords only")
                    else:
                        # SINGLE-TRACK: Expand all discriminating terms
                        terms_to_expand = discriminating_terms[:15]
                        logger.info(f"   [SEMANTIC EXPANSION] Expanding {len(terms_to_expand)} terms to find related TMDB keywords...")

                    # Build concept-to-expansions mapping for concept-based scoring
                    # IMPORTANT: Use ORIGINAL QUERY WORDS as concepts, not expanded themes
                    # This ensures "dark psychological thriller female" = 4 concepts max
                    # Not "dark + bleak + gritty + noir + tragic..." = 12+ concepts

                    for term in terms_to_expand:  # Expand all terms for candidate search
                        # Expand using BERT similarity on TMDB keywords
                        expanded = self.semantic_expander.expand_term(
                            term,
                            method='embedding',
                            top_k=5,  # CAPPED: 5 keywords per term
                            min_similarity=0.35,
                            search_keywords=True
                        )
                        expanded_with_original = set(expanded)
                        expanded_with_original.add(term.lower())
                        expanded_keywords_set.update(expanded_with_original)

                    # Build concept_to_expansions using ONLY original query words
                    # This is the key change: score based on original query concepts
                    # IMPORTANT: Use 'hybrid' method to include KEYWORD_MAPPINGS
                    # which has curated synonyms like 'female' -> 'wife', 'woman', 'girl'
                    for concept in original_query_words:
                        expanded = self.semantic_expander.expand_term(
                            concept,
                            method='hybrid',  # Changed from 'embedding' to include manual mappings
                            top_k=10,  # Increased from 5 to get more coverage
                            min_similarity=0.30,  # Lowered threshold for better recall
                            search_keywords=True
                        )
                        expanded_with_original = set(expanded)
                        expanded_with_original.add(concept.lower())
                        concept_to_expansions[concept.lower()] = list(expanded_with_original)

                    logger.info(f"      Expanded to {len(expanded_keywords_set)} total keywords across {len(concept_to_expansions)} concepts")

                    logger.info(f"   [SEMANTIC EXPANSION] Total expanded keywords: {len(expanded_keywords_set)}")
                    # Store for scoring phase
                    expanded_keywords_for_scoring = list(expanded_keywords_set)
                    # Store concept_to_expansions for use in recommend()
                    self._concept_to_expansions = concept_to_expansions
                    # Use expanded keywords for search
                    search_terms = list(expanded_keywords_set)
                else:
                    # Fallback: just use original discriminating terms
                    logger.info(f"   [KEYWORD SEARCH] Semantic expansion not available, using exact terms")
                    search_terms = discriminating_terms
                    # Each term is its own concept with no expansions
                    concept_to_expansions = {t.lower(): [t.lower()] for t in discriminating_terms}
                    self._concept_to_expansions = concept_to_expansions

                # =============================================================
                # AUTO-PASS: Movies with entity word/variant in TITLE
                # =============================================================
                # If the movie title contains the entity noun (or variant),
                # it automatically passes into the candidate pool.
                # E.g., "Single White Female" contains "Female" → AUTO-PASS
                # E.g., "Gone Girl" contains "Girl" (female variant) → AUTO-PASS

                # KEY_ENTITY_NOUNS - entity nouns that trigger title matching
                KEY_ENTITY_NOUNS_FOR_TITLE = {
                    'female', 'male', 'woman', 'man', 'girl', 'boy',
                    'dog', 'cat', 'animal', 'pet',
                    'lawyer', 'doctor', 'cop', 'detective', 'soldier', 'teacher',
                    'mother', 'father', 'daughter', 'son', 'wife', 'husband',
                    'girlfriend', 'boyfriend', 'friend', 'family',
                    'lead', 'protagonist', 'hero', 'heroine',
                    'war', 'soldier', 'military'
                }

                # NOUN_VARIANTS maps entity nouns to related keywords for title matching
                NOUN_VARIANTS_FOR_TITLE = {
                    'female': ['female', 'woman', 'girl', 'wife', 'mother', 'daughter', 'sister', 'heroine', 'femme'],
                    'male': ['male', 'man', 'boy', 'husband', 'father', 'son', 'brother', 'hero'],
                    'dog': ['dog', 'puppy', 'canine', 'hound'],
                    'cat': ['cat', 'kitten', 'feline'],
                    'lawyer': ['lawyer', 'attorney', 'legal', 'counsel'],
                    'doctor': ['doctor', 'physician', 'medical', 'nurse'],
                    'cop': ['cop', 'police', 'detective', 'officer', 'sheriff'],
                    'soldier': ['soldier', 'military', 'army', 'marine', 'veteran', 'platoon', 'battalion'],
                    'war': ['war', 'battle', 'combat', 'warfare'],
                }

                # Find entity nouns in original query words
                query_entity_nouns = [w for w in original_query_words if w in KEY_ENTITY_NOUNS_FOR_TITLE]

                # Collect all entity variants to search for in titles
                title_entity_variants = set()
                for entity_noun in query_entity_nouns:
                    # Add the entity noun itself
                    title_entity_variants.add(entity_noun)
                    # Add all variants
                    if entity_noun in NOUN_VARIANTS_FOR_TITLE:
                        title_entity_variants.update(NOUN_VARIANTS_FOR_TITLE[entity_noun])

                # Find movies where title contains any entity variant
                title_auto_pass = pd.DataFrame()
                if title_entity_variants and 'title' in candidates.columns:
                    def title_contains_entity(title):
                        if title is None or (isinstance(title, float) and np.isnan(title)):
                            return False
                        title_lower = str(title).lower()
                        # Check if any entity variant is a word in the title
                        title_words = title_lower.replace('-', ' ').replace(':', ' ').split()
                        return any(variant in title_words for variant in title_entity_variants)

                    title_mask = candidates['title'].apply(title_contains_entity)
                    title_auto_pass = candidates[title_mask]

                    if len(title_auto_pass) > 0:
                        logger.info(f"   [AUTO-PASS] {len(title_auto_pass)} movies with entity word in title (variants: {list(title_entity_variants)[:5]}...)")

                # Search TMDB metadata for these keywords (exact or expanded)
                # =============================================================
                # CANDIDATE FILTER: STRICT GENRE + ENTITY MATCHING
                # =============================================================
                # NEW LOGIC:
                # 1. If query has genre (from parsed_query.genres) -> REQUIRE genre match (via genre_id)
                # 2. If query has entity (from extracted_compounds like "female lead") -> REQUIRE entity match (with expansions)
                # 3. If neither genre nor entity in query -> fall back to 2+ matches, 1+ exact
                #
                # This ensures queries like "dark psychological thrillers with female lead"
                # require BOTH thriller genre_id AND female lead keyword match

                # Determine query requirements
                query_genres = set(g.lower() for g in parsed_query.genres) if parsed_query.genres else set()
                query_entities = extracted_compounds if extracted_compounds else set()

                # ENTITY EXPANSIONS: Build expansion map for entity concepts
                # E.g., "female lead" -> ["female lead", "femme fatale", "woman", "girl", "mother", ...]
                from semantic_tag_expander import SemanticTagExpander
                ENTITY_EXPANSIONS = {}
                for entity in query_entities:
                    entity_lower = entity.lower()
                    # Get expansions from KEYWORD_MAPPINGS
                    if entity_lower in SemanticTagExpander.KEYWORD_MAPPINGS:
                        ENTITY_EXPANSIONS[entity_lower] = [exp.lower() for exp in SemanticTagExpander.KEYWORD_MAPPINGS[entity_lower]]
                    else:
                        ENTITY_EXPANSIONS[entity_lower] = [entity_lower]

                has_genre_requirement = len(query_genres) > 0
                has_entity_requirement = len(query_entities) > 0

                # MULTI-THEME MODE: Check if we have 2+ theme groups AND NO entities
                # This ensures entity+mood queries (like "Tom Cruise thrillers about lawyers")
                # are NOT affected by multi-theme filtering logic
                has_multi_theme_requirement = (
                    not has_entity_requirement and  # CRITICAL: Only for queries with NO entities
                    hasattr(parsed_query, 'theme_groups') and
                    len(parsed_query.theme_groups) >= 2
                )
                multi_theme_groups = parsed_query.theme_groups if has_multi_theme_requirement else []

                # THEME GROUP EXPANSIONS: Expand theme groups using TMDB KEYWORD VOCABULARY
                # KEY FIX: Use search_keywords=True to search 22k+ TMDB keywords, NOT 322 zero-shot tags
                # This gives proper expansion like:
                # - "trauma" -> ["trauma", "traumatic experience", "post-traumatic stress disorder", ...]
                # - "humor" -> ["humor", "comedy", "dark humor", "black comedy", "satire", ...]
                multi_theme_expanded = [[], []]
                if has_multi_theme_requirement and self.semantic_expander:
                    for i, theme_group in enumerate(multi_theme_groups[:2]):
                        expanded_terms = set(t.lower() for t in theme_group)
                        # Expand each term in the group using TMDB keywords
                        for term in theme_group:
                            # MULTI-THEME FIX: Use 'embedding' method ONLY (not 'hybrid')
                            # 'hybrid' does substring matching first which blocks semantic search
                            # We need semantic search to find: trauma->depression, humor->comedian
                            keyword_expansions = self.semantic_expander.expand_term(
                                term,
                                method='embedding',  # EMBEDDING ONLY - no substring matching
                                search_keywords=True,  # KEY: Search TMDB keywords, not zero-shot tags
                                top_k=20,  # Get enough semantic matches
                                min_similarity=0.30  # Lower threshold to catch depression, comedian, etc.
                            )
                            expanded_terms.update(keyword_expansions)
                        multi_theme_expanded[i] = list(expanded_terms)
                        logger.info(f"   [MULTI-THEME EXPANSION] Group {i+1} '{theme_group}' -> {len(multi_theme_expanded[i])} TMDB keywords")
                elif has_multi_theme_requirement:
                    multi_theme_expanded = [[t.lower() for t in g] for g in multi_theme_groups[:2]]

                logger.info(f"   [CANDIDATE FILTER] Query genres: {query_genres}")
                logger.info(f"   [CANDIDATE FILTER] Query entities: {query_entities}")
                logger.info(f"   [CANDIDATE FILTER] Entity expansions: {list(ENTITY_EXPANSIONS.keys())}")
                if has_genre_requirement:
                    logger.info(f"   [CANDIDATE FILTER] REQUIRING genre match via genre_id")
                if has_entity_requirement:
                    logger.info(f"   [CANDIDATE FILTER] REQUIRING entity match (with expansions)")
                if has_multi_theme_requirement:
                    logger.info(f"   [CANDIDATE FILTER] MULTI-THEME MODE: requiring match from BOTH expanded theme groups")
                    logger.info(f"   [CANDIDATE FILTER] Theme Group 1: {multi_theme_groups[0]} -> {len(multi_theme_expanded[0])} expanded terms")
                    logger.info(f"   [CANDIDATE FILTER] Theme Group 2: {multi_theme_groups[1]} -> {len(multi_theme_expanded[1])} expanded terms")
                if not has_genre_requirement and not has_entity_requirement and not has_multi_theme_requirement:
                    logger.info(f"   [CANDIDATE FILTER] No genre/entity/multi-theme requirements - using 2+ match fallback")
                logger.info(f"   [CANDIDATE FILTER] Original query words: {original_query_words}")

                # Genre ID to concept mapping for candidate filtering
                CONCEPT_TO_GENRE_IDS = {
                    'thriller': [53], 'thrillers': [53],
                    'horror': [27], 'comedy': [35], 'comedies': [35],
                    'drama': [18], 'dramas': [18], 'action': [28],
                    'romance': [10749], 'romantic': [10749],
                    'sci-fi': [878], 'science fiction': [878],
                    'fantasy': [14], 'mystery': [9648], 'crime': [80],
                    'war': [10752], 'western': [37], 'animation': [16],
                    'animated': [16], 'adventure': [12], 'documentary': [99],
                    'family': [10751], 'history': [36], 'historical': [36],
                }

                def check_candidate_passes(row):
                    """
                    Check if movie passes the candidate filter.

                    NEW STRICT LOGIC:
                    1. If query has genre -> REQUIRE genre match via genre_id
                    2. If query has entity -> REQUIRE entity match (with expansions)
                    3. If neither -> fall back to 2+ matches, 1+ exact

                    Returns: (passes: bool, query_words_matched: int, exact_matches: int)
                    """
                    movie_keywords = row.get('keywords', None)
                    movie_genre_ids = row.get('genre_ids', None)

                    # Parse movie keywords
                    movie_kw_lower = []
                    if movie_keywords is not None:
                        if isinstance(movie_keywords, float) and np.isnan(movie_keywords):
                            pass
                        elif isinstance(movie_keywords, (list, np.ndarray)) and len(movie_keywords) > 0:
                            movie_kw_lower = [str(k).strip().lower() for k in movie_keywords]
                        elif isinstance(movie_keywords, str):
                            movie_kw_lower = [movie_keywords.strip().lower()]

                    # Parse movie genre_ids
                    genre_ids_list = []
                    if movie_genre_ids is not None:
                        if isinstance(movie_genre_ids, (list, np.ndarray)) and len(movie_genre_ids) > 0:
                            genre_ids_list = [int(gid) for gid in movie_genre_ids]

                    # =============================================================
                    # STRICT REQUIREMENT 1: GENRE MATCH (if query has genre)
                    # =============================================================
                    genre_matched = False
                    if has_genre_requirement:
                        for qg in query_genres:
                            # Check via genre_id
                            if qg in CONCEPT_TO_GENRE_IDS and genre_ids_list:
                                expected_gids = CONCEPT_TO_GENRE_IDS[qg]
                                if any(gid in genre_ids_list for gid in expected_gids):
                                    genre_matched = True
                                    break
                            # Also check via keywords (e.g., "thriller" in keywords)
                            for movie_kw in movie_kw_lower:
                                if qg in movie_kw or movie_kw in qg:
                                    genre_matched = True
                                    break
                            if genre_matched:
                                break

                        # If genre required but not matched, FAIL immediately
                        if not genre_matched:
                            return False, 0, 0

                    # =============================================================
                    # STRICT REQUIREMENT 2: ENTITY MATCH (if query has entity)
                    # =============================================================
                    entity_matched = False
                    if has_entity_requirement:
                        for entity in query_entities:
                            entity_lower = entity.lower()
                            expansions = ENTITY_EXPANSIONS.get(entity_lower, [entity_lower])

                            # Check if any expansion matches movie keywords
                            for exp in expansions:
                                for movie_kw in movie_kw_lower:
                                    if exp in movie_kw or movie_kw in exp:
                                        entity_matched = True
                                        break
                                if entity_matched:
                                    break
                            if entity_matched:
                                break

                        # If entity required but not matched, FAIL immediately
                        if not entity_matched:
                            return False, 0, 0

                    # =============================================================
                    # STRICT REQUIREMENT 3: MULTI-THEME MATCH (if multi-theme mode)
                    # Use EXPANDED theme groups (semantic expansion like entity queries)
                    # Require at least ONE theme group to match
                    # =============================================================
                    theme_groups_matched = 0
                    if has_multi_theme_requirement:
                        # Get movie overview for additional matching
                        movie_overview = row.get('overview', '')
                        overview_lower = str(movie_overview).lower() if movie_overview else ''

                        # Check Theme Group 1 using EXPANDED terms
                        group1_matched = False
                        expanded_group1 = multi_theme_expanded[0] if multi_theme_expanded else [t.lower() for t in multi_theme_groups[0]]
                        for exp_term in expanded_group1:
                            # Check in movie keywords
                            for movie_kw in movie_kw_lower:
                                if exp_term in movie_kw or movie_kw in exp_term:
                                    group1_matched = True
                                    break
                            # Also check overview text
                            if not group1_matched and exp_term in overview_lower:
                                group1_matched = True
                            if group1_matched:
                                break

                        # Check Theme Group 2 using EXPANDED terms
                        group2_matched = False
                        expanded_group2 = multi_theme_expanded[1] if len(multi_theme_expanded) > 1 else [t.lower() for t in multi_theme_groups[1]]
                        for exp_term in expanded_group2:
                            # Check in movie keywords
                            for movie_kw in movie_kw_lower:
                                if exp_term in movie_kw or movie_kw in exp_term:
                                    group2_matched = True
                                    break
                            # Also check overview text
                            if not group2_matched and exp_term in overview_lower:
                                group2_matched = True
                            if group2_matched:
                                break

                        # Count matched groups
                        if group1_matched:
                            theme_groups_matched += 1
                        if group2_matched:
                            theme_groups_matched += 1

                        # Require BOTH theme groups to match
                        # This ensures movies have signal for both themes (e.g., trauma AND humor)
                        if theme_groups_matched < 2:
                            return False, theme_groups_matched, theme_groups_matched

                    # =============================================================
                    # FALLBACK: If no genre/entity/multi-theme requirements, use 2+ match logic
                    # =============================================================
                    if not has_genre_requirement and not has_entity_requirement and not has_multi_theme_requirement:
                        # Original 2+ matches, 1+ exact logic
                        matched_query_words = set()
                        exact_match_words = set()

                        for query_word in original_query_words:
                            query_word_lower = query_word.lower()
                            expansions = concept_to_expansions.get(query_word_lower, [query_word_lower])

                            # Check genre_id first
                            if query_word_lower in CONCEPT_TO_GENRE_IDS and genre_ids_list:
                                expected_gids = CONCEPT_TO_GENRE_IDS[query_word_lower]
                                if any(gid in genre_ids_list for gid in expected_gids):
                                    matched_query_words.add(query_word_lower)
                                    exact_match_words.add(query_word_lower)
                                    continue

                            # Check expansions against genre_ids
                            for expansion in expansions:
                                exp_lower = expansion.lower()
                                if exp_lower in CONCEPT_TO_GENRE_IDS and genre_ids_list:
                                    expected_gids = CONCEPT_TO_GENRE_IDS[exp_lower]
                                    if any(gid in genre_ids_list for gid in expected_gids):
                                        matched_query_words.add(query_word_lower)
                                        break

                            if query_word_lower in matched_query_words:
                                continue

                            # Check keywords
                            for expansion in expansions:
                                exp_lower = expansion.lower()
                                for movie_kw in movie_kw_lower:
                                    if exp_lower in movie_kw or movie_kw in exp_lower:
                                        matched_query_words.add(query_word_lower)
                                        if query_word_lower in movie_kw or movie_kw in query_word_lower:
                                            exact_match_words.add(query_word_lower)
                                        break
                                if query_word_lower in matched_query_words:
                                    break

                        num_matched = len(matched_query_words)
                        num_exact = len(exact_match_words)
                        passes = (num_matched >= 2) and (num_exact >= 1)
                        return passes, num_matched, num_exact

                    # If we reach here, all requirements (genre/entity/multi-theme) were satisfied
                    # Count how many concepts matched for reporting
                    matched_count = 0
                    if genre_matched:
                        matched_count += 1
                    if entity_matched:
                        matched_count += 1
                    if has_multi_theme_requirement:
                        matched_count += theme_groups_matched  # Add both matched theme groups

                    return True, matched_count, matched_count  # All matches are "exact" for strict mode

                # Apply filter to all candidates - now uses row-wise apply to access both keywords AND genre_ids
                if 'keywords' in candidates.columns:
                    filter_results = candidates.apply(check_candidate_passes, axis=1)
                    candidates['_passes'] = filter_results.apply(lambda x: x[0])
                    candidates['_num_matched'] = filter_results.apply(lambda x: x[1])
                    candidates['_num_exact'] = filter_results.apply(lambda x: x[2])

                    # Get candidates that pass the filter
                    keyword_filtered = candidates[candidates['_passes'] == True].copy()

                    # Log stats
                    total_checked = len(candidates)
                    num_passed = len(keyword_filtered)
                    logger.info(f"   [CANDIDATE FILTER] {num_passed}/{total_checked} candidates passed (2+ matches, 1+ exact)")

                    # Clean up temp columns
                    for col in ['_passes', '_num_matched', '_num_exact']:
                        if col in keyword_filtered.columns:
                            keyword_filtered = keyword_filtered.drop(columns=[col])
                        if col in candidates.columns:
                            candidates = candidates.drop(columns=[col])

                    if len(keyword_filtered) > 0:
                        discriminating_candidates = keyword_filtered
                        keyword_search_used = True
                    else:
                        logger.info(f"   [CANDIDATE FILTER] No candidates passed, using fallback (any match)")

                # Merge AUTO-PASS title matches with keyword matches
                if len(title_auto_pass) > 0:
                    if discriminating_candidates is not None and len(discriminating_candidates) > 0:
                        # Combine and deduplicate
                        discriminating_candidates = pd.concat([discriminating_candidates, title_auto_pass]).drop_duplicates(subset=['title'])
                        logger.info(f"   [AUTO-PASS] Merged title matches → {len(discriminating_candidates)} total candidates")
                    else:
                        # Only title matches (no keyword matches)
                        discriminating_candidates = title_auto_pass
                        keyword_search_used = True
                        logger.info(f"   [AUTO-PASS] Using {len(title_auto_pass)} title-matched candidates (no keyword matches)")

        # STEP 2: BROAD TMDB-BASED FILTERING
        # CRITICAL FIX: When keyword search found discriminating terms with entity+genre requirements,
        # SKIP all TMDB filtering on base candidates - we'll use discriminating_candidates directly
        # This implements the correct accuracy mode hierarchy:
        # 1. Entity+Genre filter FIRST (check_candidate_passes ensures BOTH match)
        # 2. Decade filter on those filtered candidates (not all 43858!)
        # 3. Use that as the candidate pool
        if keyword_search_used and discriminating_candidates is not None and len(discriminating_candidates) > 0:
            logger.info(f"   [ACCURACY MODE] Entity+Genre filter found {len(discriminating_candidates)} candidates")
            logger.info(f"   SKIPPING redundant TMDB filtering on base candidates")
            # Apply decade/year filter directly to discriminating_candidates
            if parsed_query.decades and 'year' in discriminating_candidates.columns:
                decade_ranges = [(d, d+9) for d in parsed_query.decades]
                mask = False
                for start, end in decade_ranges:
                    mask = mask | ((discriminating_candidates['year'] >= start) & (discriminating_candidates['year'] <= end))
                discriminating_candidates = discriminating_candidates[mask]
                logger.info(f"   [ENTITY+GENRE POOL] Decade filter: {len(discriminating_candidates)} movies from {[f'{d}s' for d in parsed_query.decades]}")
            elif parsed_query.year_range and 'year' in discriminating_candidates.columns:
                start, end = parsed_query.year_range
                discriminating_candidates = discriminating_candidates[
                    (discriminating_candidates['year'] >= start) & (discriminating_candidates['year'] <= end)
                ]
                logger.info(f"   [ENTITY+GENRE POOL] Year filter: {len(discriminating_candidates)} movies from {start}-{end}")
            elif parsed_query.years and 'year' in discriminating_candidates.columns:
                discriminating_candidates = discriminating_candidates[discriminating_candidates['year'].isin(parsed_query.years)]
                logger.info(f"   [ENTITY+GENRE POOL] Year filter: {len(discriminating_candidates)} movies from {parsed_query.years}")

            # Use discriminating_candidates as the ONLY candidate pool (no merge with base)
            candidates = discriminating_candidates
            # Clear discriminating_candidates to skip redundant merge logic later
            discriminating_candidates = None
            logger.info(f"   [ENTITY+GENRE MODE] Using {len(candidates)} entity+genre filtered candidates as final pool")
            # Skip to end of TMDB filtering section
        elif keyword_search_used:
            logger.info(f"   [ACCURACY MODE] Keyword search found discriminating terms - SKIPPING genre filter")
            logger.info(f"   Using pure TMDB keyword matching for candidate selection")
            # Skip genre filtering entirely - we already have keyword-matched candidates
            # Genre will be part of the keyword scoring, not a separate filter
        elif parsed_query.genres and 'genre_ids' in candidates.columns:
            genre_name_to_id = {
                'action': 28, 'adventure': 12, 'animation': 16, 'comedy': 35,
                'comedies': 35, 'crime': 80, 'drama': 18, 'dramas': 18,
                'fantasy': 14, 'horror': 27, 'horrors': 27,
                'romance': 10749, 'romances': 10749, 'romantic': 10749,
                'sci-fi': 878, 'science fiction': 878,
                'thriller': 53, 'thrillers': 53, 'western': 37, 'westerns': 37,
                'war': 10752, 'mystery': 9648, 'mysteries': 9648
            }

            query_genre_ids = []
            for genre in parsed_query.genres:
                genre_lower = genre.lower()
                if genre_lower in genre_name_to_id:
                    query_genre_ids.append(genre_name_to_id[genre_lower])

            if query_genre_ids:
                def has_genre(genre_array):
                    if genre_array is None:
                        return False
                    if isinstance(genre_array, (list, np.ndarray)):
                        return any(gid in genre_array for gid in query_genre_ids)
                    return False

                genre_mask = candidates['genre_ids'].apply(has_genre)
                candidates = candidates[genre_mask]
                logger.info(f"   Genre filter: {len(candidates)} movies matching {parsed_query.genres}")

                # STRICTER FILTER: For comedy queries, exclude dark/crime/animated comedies
                if 'comedy' in [g.lower() for g in parsed_query.genres]:
                    def is_live_action_comedy(genre_array):
                        if genre_array is None or not isinstance(genre_array, (list, np.ndarray)):
                            return True
                        excluded_genres = {80, 53, 27, 16}  # crime, thriller, horror, animation
                        has_excluded = any(gid in excluded_genres for gid in genre_array)
                        return not has_excluded

                    live_action_comedy_mask = candidates['genre_ids'].apply(is_live_action_comedy)
                    candidates = candidates[live_action_comedy_mask]
                    logger.info(f"   Comedy filter: {len(candidates)} live-action comedies")

        # Filter by year/decade
        if parsed_query.year_range:
            start, end = parsed_query.year_range
            if 'year' in candidates.columns:
                candidates = candidates[
                    (candidates['year'] >= start) & (candidates['year'] <= end)
                ]
                logger.info(f"   Year range filter: {len(candidates)} movies from {start}-{end}")
        elif parsed_query.years:
            if 'year' in candidates.columns:
                candidates = candidates[candidates['year'].isin(parsed_query.years)]
                logger.info(f"   Year filter: {len(candidates)} movies from {parsed_query.years}")
        elif parsed_query.decades:
            if 'year' in candidates.columns:
                decade_ranges = [(d, d+9) for d in parsed_query.decades]
                mask = False
                for start, end in decade_ranges:
                    mask = mask | ((candidates['year'] >= start) & (candidates['year'] <= end))
                candidates = candidates[mask]
                logger.info(f"   Decade filter: {len(candidates)} movies from {[f'{d}s' for d in parsed_query.decades]}")

        # ADULT CONTEXT FILTER
        if parsed_query.adult_context and 'genre_ids' in candidates.columns:
            def is_adult_appropriate(genre_array):
                if genre_array is None or not isinstance(genre_array, (list, np.ndarray)):
                    return True
                excluded_genres = {16, 10751}  # Animation, Family
                has_excluded = any(gid in excluded_genres for gid in genre_array)
                return not has_excluded

            adult_mask = candidates['genre_ids'].apply(is_adult_appropriate)
            candidates = candidates[adult_mask]
            logger.info(f"   Adult context filter: {len(candidates)} movies")

        # PHASE 2: ENTITY-BASED FILTERING (using NER-detected actors, directors, studios)
        # ACTOR/CAST FILTERING: Use NER-detected actors from Phase 1
        detected_actor_name = None  # Track the detected actor name
        if parsed_query.actors and 'cast' in candidates.columns:
            # Use actors detected by EntityExtractor in Phase 1
            for actor_name in parsed_query.actors:
                def has_actor(cast):
                    # Handle various cast formats
                    if cast is None:
                        return False
                    if isinstance(cast, float) and np.isnan(cast):
                        return False
                    if isinstance(cast, str) and cast == '':
                        return False

                    # Convert to string for searching
                    cast_str = str(cast).lower()
                    return actor_name.lower() in cast_str

                actor_mask = candidates['cast'].apply(has_actor)
                if actor_mask.any():
                    candidates = candidates[actor_mask]
                    actor_filter_applied = True
                    detected_actor_name = actor_name  # Store the actor name
                    logger.info(f"   [PHASE 2] Filtered to {len(candidates)} movies with actor: {actor_name}")
                    break  # Use first detected actor name

        # DIRECTOR FILTERING: Use NER-detected directors from Phase 1
        # IMPORTANT: Only run if actor filtering didn't already succeed (avoid double-filtering)
        if not actor_filter_applied and parsed_query.directors and 'director' in candidates.columns:
            # Use directors detected by EntityExtractor in Phase 1
            for director_name in parsed_query.directors:
                def has_director(director):
                    if director is None:
                        return False
                    if isinstance(director, float) and np.isnan(director):
                        return False
                    if isinstance(director, str) and director == '':
                        return False

                    # Convert to string for searching
                    director_str = str(director).lower()
                    return director_name.lower() in director_str

                director_mask = candidates['director'].apply(has_director)
                if director_mask.any():
                    candidates = candidates[director_mask]
                    actor_filter_applied = True  # Reuse flag to skip semantic expansion
                    logger.info(f"   [PHASE 2] Filtered to {len(candidates)} movies with director: {director_name}")
                    break  # Use first detected director name

        # STUDIO/PRODUCTION COMPANY FILTERING: Use NER-detected studios from Phase 1
        # IMPORTANT: Only run if actor/director filtering didn't already succeed
        if not actor_filter_applied and parsed_query.studios and 'production_companies' in candidates.columns:
            # Use studios detected by EntityExtractor in Phase 1
            for studio_name in parsed_query.studios:
                def has_studio(prod_companies):
                    if prod_companies is None:
                        return False
                    if isinstance(prod_companies, float):
                        return False
                    if isinstance(prod_companies, (list, np.ndarray)):
                        # Check if studio name appears in production companies
                        companies_str = ' '.join(str(c) for c in prod_companies).lower()
                        return studio_name.lower() in companies_str
                    return False

                studio_mask = candidates['production_companies'].apply(has_studio)
                if studio_mask.any():
                    candidates = candidates[studio_mask]
                    actor_filter_applied = True  # Reuse flag to skip semantic expansion
                    logger.info(f"   [PHASE 2] Filtered to {len(candidates)} movies from studio: {studio_name}")
                    break  # Use first detected studio

        # CRITICAL FIX: Filter by female lead if query mentions it
        # SKIP this if entity+genre filter was already applied (keyword_search_used handles "female lead" via TMDB keywords)
        # The entity+genre filter uses TMDB keywords like "femme fatale", "female protagonist" which is more comprehensive
        # than this hardcoded actress name list
        if not keyword_search_used and (parsed_query.themes or parsed_query.keywords):
            all_terms = (parsed_query.themes + parsed_query.keywords)
            query_text = " ".join(all_terms).lower()

            # Detect female lead queries
            female_indicators = ['female lead', 'strong female', 'woman', 'women', 'actress', 'girl']
            needs_female_lead = any(indicator in query_text for indicator in female_indicators)

            if needs_female_lead and 'cast' in candidates.columns:
                # Comprehensive female actress list for filtering
                female_names = [
                    # A-list actresses
                    'Sandra Bullock', 'Reese Witherspoon', 'Julia Roberts', 'Jennifer Aniston',
                    'Meryl Streep', 'Charlize Theron', 'Angelina Jolie', 'Cameron Diaz',
                    'Kate Hudson', 'Anne Hathaway', 'Emma Stone', 'Kristen Wiig',
                    'Melissa McCarthy', 'Amy Schumer', 'Jennifer Lawrence', 'Scarlett Johansson',
                    'Natalie Portman', 'Nicole Kidman', 'Cate Blanchett', 'Kate Winslet',
                    # Comedy actresses
                    'Lindsay Lohan', 'Anna Kendrick', 'Jennifer Garner', 'Alicia Silverstone',
                    'Tina Fey', 'Amy Poehler', 'Rebel Wilson', 'Drew Barrymore',
                    'Mila Kunis', 'Zooey Deschanel', 'Katherine Heigl', 'Kate Beckinsale',
                    'Rachel McAdams', 'Emma Watson', 'Amy Adams', 'Jessica Alba',
                    # Additional leading ladies
                    'Keira Knightley', 'Halle Berry', 'Penelope Cruz', 'Naomi Watts',
                    'Uma Thurman', 'Winona Ryder', 'Michelle Pfeiffer', 'Meg Ryan',
                    'Diane Keaton', 'Goldie Hawn', 'Sally Field', 'Sigourney Weaver',
                    'Geena Davis', 'Susan Sarandon', 'Helen Mirren', 'Judi Dench',
                    # CRITICAL: Thriller/Drama leads (for psychological thriller queries)
                    'Jodie Foster', 'Kathy Bates', 'Rosamund Pike', 'Glenn Close',
                    'Jessica Chastain', 'Rooney Mara', 'Mia Wasikowska', 'Tilda Swinton',
                    'Frances McDormand', 'Julianne Moore', 'Viola Davis', 'Octavia Spencer'
                ]

                def has_female_lead(cast):
                    if cast is None or len(cast) == 0:
                        return False
                    # SOFTENED: Check first 3 cast members (not just first)
                    # This catches movies like "The Silence of the Lambs" where Jodie Foster is 2nd billing
                    if isinstance(cast, (list, np.ndarray)):
                        # Check up to first 3 cast members
                        top_cast = ' '.join(str(actor) for actor in cast[:3])
                        return any(name in top_cast for name in female_names)
                    else:
                        # Single cast entry
                        return any(name in str(cast) for name in female_names)

                female_mask = candidates['cast'].apply(has_female_lead)
                if female_mask.any():
                    candidates = candidates[female_mask]
                    logger.info(f"   Filtered to {len(candidates)} movies with female leads (top 3 cast)")

        # STEP 3: Return candidates sorted by popularity
        # NEW STRATEGY: Don't filter by zero-shot tags!
        # Return ALL movies matching TMDB criteria (genre, year, actor, etc.)
        # Zero-shot tags will be used as BOOST signal during scoring, not as filter

        # MERGE DISCRIMINATING CANDIDATES (from keyword search)
        # ACCURACY MODE FIX: When keyword search is used, prioritize keyword-matched candidates
        if discriminating_candidates is not None and len(discriminating_candidates) > 0:
            if keyword_search_used:
                # CRITICAL FIX: Preserve actor/director/studio filters from Phase 2!
                # When NER detected an entity (actor/director/studio), we must keep that filter
                if actor_filter_applied:
                    # INTERSECT keyword candidates with entity-filtered candidates
                    # This ensures we only return movies matching BOTH the entity AND the keywords
                    before_intersect = len(discriminating_candidates)
                    entity_titles = set(candidates['title'])
                    discriminating_candidates = discriminating_candidates[discriminating_candidates['title'].isin(entity_titles)]
                    logger.info(f"   [PHASE 2 PRESERVED] Intersected {before_intersect} keyword candidates with entity filter -> {len(discriminating_candidates)} movies")
                    candidates = discriminating_candidates
                else:
                    # CRITICAL FIX: Filter by decade/year BEFORE merging!
                    if parsed_query.decades or parsed_query.year_range:
                        logger.info(f"   [ACCURACY MODE FILTER] Filtering {len(discriminating_candidates)} keyword matches by decade/year...")

                        if parsed_query.decades:
                            decade_ranges = [(d, d+9) for d in parsed_query.decades]
                            mask = False
                            for start, end in decade_ranges:
                                mask = mask | ((discriminating_candidates['year'] >= start) & (discriminating_candidates['year'] <= end))
                            discriminating_candidates = discriminating_candidates[mask]
                        elif parsed_query.year_range:
                            start_year, end_year = parsed_query.year_range
                            discriminating_candidates = discriminating_candidates[
                                (discriminating_candidates['year'] >= start_year) &
                                (discriminating_candidates['year'] <= end_year)
                            ]

                        logger.info(f"   [ACCURACY MODE FILTER] Kept {len(discriminating_candidates)} keyword matches within date range")

                    # FIX: MERGE keyword candidates instead of replacing all candidates
                    # Keyword matches get boosted during scoring, but don't exclude other good movies
                    before_merge = len(candidates)
                    candidates = pd.concat([candidates, discriminating_candidates]).drop_duplicates(subset=['title'])
                    added = len(candidates) - before_merge
                    logger.info(f"   [ACCURACY MODE] Merged {added} keyword-matched movies with base candidates (total: {len(candidates)})")
            else:
                # Normal mode: Merge keyword candidates with filtered candidates
                # CRITICAL FIX: Filter discriminating_candidates by decade/year BEFORE merging!
                if parsed_query.decades or parsed_query.year_range:
                    logger.info(f"   [KEYWORD FILTER] Filtering {len(discriminating_candidates)} keyword matches by decade/year...")

                    # Apply same decade/year filtering as main candidates
                    if parsed_query.decades:
                        decade_ranges = [(d, d+9) for d in parsed_query.decades]
                        mask = False
                        for start, end in decade_ranges:
                            mask = mask | ((discriminating_candidates['year'] >= start) & (discriminating_candidates['year'] <= end))
                        discriminating_candidates = discriminating_candidates[mask]
                    elif parsed_query.year_range:
                        start_year, end_year = parsed_query.year_range
                        discriminating_candidates = discriminating_candidates[
                            (discriminating_candidates['year'] >= start_year) &
                            (discriminating_candidates['year'] <= end_year)
                        ]

                    logger.info(f"   [KEYWORD FILTER] Kept {len(discriminating_candidates)} keyword matches within date range")

                before_merge = len(candidates)
                candidates = pd.concat([candidates, discriminating_candidates]).drop_duplicates(subset=['title'])
                added = len(candidates) - before_merge
                logger.info(f"   [KEYWORD MERGE] Added {added} keyword-matched movies to candidate pool (total: {len(candidates)})")

        if len(candidates) > max_candidates:
            # Prioritize keyword-matched movies in sampling
            if discriminating_candidates is not None and len(discriminating_candidates) > 0:
                # Ensure ALL keyword-matched movies are included (they have perfect matches!)
                keyword_titles = set(discriminating_candidates['title'])
                keyword_mask = candidates['title'].isin(keyword_titles)
                keyword_movies = candidates[keyword_mask]
                other_movies = candidates[~keyword_mask]

                # Take all keyword-matched movies + fill remaining slots
                remaining_slots = max_candidates - len(keyword_movies)
                if remaining_slots > 0 and len(other_movies) > 0:
                    # Sort others by popularity
                    if 'vote_count' in other_movies.columns:
                        other_movies = other_movies.nlargest(remaining_slots, 'vote_count')
                    else:
                        other_movies = other_movies.sample(min(remaining_slots, len(other_movies)))

                    candidates = pd.concat([keyword_movies, other_movies])
                    logger.info(f"   [SAMPLING] Kept ALL {len(keyword_movies)} keyword-matched + {len(other_movies)} popular movies")
                else:
                    candidates = keyword_movies
                    logger.info(f"   [SAMPLING] Kept ALL {len(keyword_movies)} keyword-matched movies")
            else:
                # No keyword matches - use standard popularity-based sampling
                if 'vote_count' in candidates.columns:
                    candidates = candidates.nlargest(max_candidates, 'vote_count')
                else:
                    candidates = candidates.sample(min(max_candidates, len(candidates)))

        # Return titles
        if 'title' in candidates.columns:
            final_candidates = candidates['title'].tolist()
            logger.info(f"   INVERTED FUNNEL: Returning {len(final_candidates)} candidates (TMDB-filtered, not tag-filtered)")
            logger.info(f"   Zero-shot tags will be used as BOOST signal during scoring")
            if keyword_search_used:
                logger.info(f"   KEYWORD SEARCH was used - semantic expansion will be skipped")
            # Return expanded keywords for scoring (or None if not expanded)
            if expanded_keywords_for_scoring:
                logger.info(f"   Returning {len(expanded_keywords_for_scoring)} expanded keywords for scoring")
            # Return: (candidates, actor_filter_applied, curated_list_used=False, expanded_keywords, concept_groups, keyword_search_used)
            return (final_candidates, actor_filter_applied, False, expanded_keywords_for_scoring, concept_groups, keyword_search_used)
        else:
            logger.warning("No 'title' column in movies dataframe")
            return ([], False, False, None, None, False)

    def _expand_candidates_with_similarity(self,
                                           initial_candidates: List[str],
                                           parsed_query: ParsedQuery,
                                           target_count: int = 200) -> List[str]:
        """
        Expand candidates using semantic similarity when initial results are too few.

        This is the SEMANTIC SIMILARITY FALLBACK for low-result queries like:
        - "mafia movies" (only finds Godfather due to missing tags)
        - "gladiator movies" (only finds Gladiator)
        - "heist movies" (limited tag coverage)

        Strategy:
        1. Preserve ALL movies with theme tags (e.g., all movies with "mafia" tag)
        2. Take only theme-tagged movies as reference (not popular irrelevant movies)
        3. Find similar movies using content embeddings
        4. Filter by genre/decade if specified in query
        5. Return expanded candidate list

        Args:
            initial_candidates: Initial candidate list (may be small)
            parsed_query: Parsed query for filtering
            target_count: Target number of candidates to return

        Returns:
            Expanded list of candidate movie titles
        """
        if len(initial_candidates) >= target_count:
            return initial_candidates  # Already have enough

        logger.info(f"\n[SEMANTIC FALLBACK] Expanding {len(initial_candidates)} candidates to ~{target_count}...")

        # CRITICAL FIX: Find movies that actually have the theme tags
        query_tags = parsed_query.themes + parsed_query.keywords
        theme_tagged_movies = []

        if query_tags:
            unified_tags = self.zero_shot_integrator.merge_tags()
            if unified_tags is not None:
                # Find all movies with the theme tag
                for tag in query_tags:
                    if tag in unified_tags.columns:
                        tagged_titles = unified_tags[unified_tags[tag] > 0]['title_norm'].tolist()
                        # Convert to actual titles (case-insensitive match!)
                        for norm_title in tagged_titles:
                            match = self.movies[self.movies['title'].str.lower().str.strip() == norm_title]
                            if len(match) > 0:
                                actual_title = match.iloc[0]['title']
                                # Case-insensitive check if in initial candidates
                                for candidate in initial_candidates:
                                    if candidate.lower().strip() == actual_title.lower().strip():
                                        theme_tagged_movies.append(candidate)  # Use candidate's spelling
                                        break

                theme_tagged_movies = list(dict.fromkeys(theme_tagged_movies))  # Remove dups
                logger.info(f"   Found {len(theme_tagged_movies)} theme-tagged movies in candidates")

        # Use ONLY theme-tagged movies as references (not Inception/Interstellar!)
        if theme_tagged_movies:
            reference_movies = theme_tagged_movies[:min(5, len(theme_tagged_movies))]
        else:
            # Fallback: use top 3 candidates
            reference_movies = initial_candidates[:min(3, len(initial_candidates))]

        logger.info(f"   Using reference movies: {reference_movies}")

        # Get embeddings for reference movies
        if not self.signal_fusion.content_available:
            logger.warning("   Content embeddings not available, skipping expansion")
            return initial_candidates

        # Find similar movies for each reference
        # CRITICAL: Start with theme-tagged movies to ensure they're included!
        similar_movies = set(theme_tagged_movies) if theme_tagged_movies else set()

        for ref_movie in reference_movies:
            # Get embedding for reference movie
            if ref_movie not in self.signal_fusion.title_to_embedding_idx:
                logger.warning(f"   {ref_movie} not in embedding index")
                continue

            ref_idx = self.signal_fusion.title_to_embedding_idx[ref_movie]
            ref_emb = self.signal_fusion.movie_embeddings[ref_idx].reshape(1, -1)

            # Compute similarity to ALL movies
            all_embs = self.signal_fusion.movie_embeddings
            similarities = cosine_similarity(ref_emb, all_embs)[0]

            # Get top N most similar movies
            # INCREASED from 100 to 300 to capture GoodFellas (#163) and other mafia films
            top_indices = np.argsort(similarities)[::-1][:300]  # Top 300 similar

            # Convert indices to titles
            idx_to_title = {idx: title for title, idx in self.signal_fusion.title_to_embedding_idx.items()}

            for idx in top_indices:
                if idx in idx_to_title:
                    title = idx_to_title[idx]
                    similar_movies.add(title)

                    if len(similar_movies) >= target_count:
                        break

            if len(similar_movies) >= target_count:
                break

        # Convert to list
        expanded = list(similar_movies)

        # Filter by genre/decade if specified in query (to stay relevant)
        if parsed_query.genres or parsed_query.decades or parsed_query.year_range:
            filtered = []
            for title in expanded:
                movie_row = self.movies[self.movies['title'] == title]
                if len(movie_row) == 0:
                    continue

                movie_row = movie_row.iloc[0]

                # Check genre
                if parsed_query.genres:
                    genre_ids = movie_row.get('genre_ids', [])
                    genre_name_to_id = {
                        'action': 28, 'adventure': 12, 'animation': 16, 'comedy': 35,
                        'comedies': 35, 'crime': 80, 'drama': 18, 'dramas': 18,
                        'fantasy': 14, 'horror': 27, 'horrors': 27,
                        'romance': 10749, 'romances': 10749, 'romantic': 10749,
                        'sci-fi': 878, 'science fiction': 878,
                        'thriller': 53, 'thrillers': 53, 'western': 37, 'westerns': 37,
                        'war': 10752, 'mystery': 9648, 'mysteries': 9648
                    }
                    query_genre_ids = [genre_name_to_id.get(g.lower()) for g in parsed_query.genres]
                    query_genre_ids = [g for g in query_genre_ids if g is not None]

                    if not any(gid in genre_ids for gid in query_genre_ids):
                        continue  # Skip if doesn't match genre

                # Check year/decade
                year = movie_row.get('year')

                # CRITICAL FIX: Validate year is a valid number
                if year is None or pd.isna(year):
                    continue  # Skip movies with missing years when decade/year filter is active

                try:
                    year = int(year)
                except (ValueError, TypeError):
                    continue  # Skip movies with invalid years

                if parsed_query.year_range:
                    start, end = parsed_query.year_range
                    if year < start or year > end:
                        continue
                elif parsed_query.decades:
                    decade_match = False
                    for decade in parsed_query.decades:
                        if decade <= year < decade + 10:
                            decade_match = True
                            break
                    if not decade_match:
                        continue

                filtered.append(title)

            expanded = filtered

        logger.info(f"   Expanded to {len(expanded)} candidates via semantic similarity")
        return expanded[:target_count]  # Return up to target count

    def _compute_semantic_similarity(self, query_text: str, movie_tags: List[str]) -> float:
        """
        Compute semantic similarity between query and movie tags using BERT embeddings.

        Args:
            query_text: The user query (e.g., "ancient history revenge")
            movie_tags: List of tags for the movie (e.g., ["epic", "gladiator", "trojan war"])

        Returns:
            Semantic similarity score (0-1), higher = more similar
        """
        if not self.use_semantic_expansion or not movie_tags:
            return 0.0

        try:
            import torch

            # Encode query
            query_inputs = self.bert_tokenizer(
                query_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                query_outputs = self.bert_model(**query_inputs)
                query_embedding = query_outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token

            # Encode movie tags (combine into single string)
            tags_text = " ".join(movie_tags)
            tag_inputs = self.bert_tokenizer(
                tags_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                tag_outputs = self.bert_model(**tag_inputs)
                tag_embedding = tag_outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token

            # Compute cosine similarity
            similarity = cosine_similarity(query_embedding, tag_embedding)[0][0]
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            return max(0.0, (similarity + 1) / 2)

        except Exception as e:
            logger.warning(f"Error computing semantic similarity: {e}")
            return 0.0

    def recommend(self,
                  query: str,
                  user_id: Optional[int] = None,
                  reference_movie: Optional[str] = None,
                  top_n: int = 10,
                  max_candidates: int = 500,
                  preference_mode: str = "balanced") -> RecommendationResult:
        """
        Generate recommendations for a natural language query with user preference.

        Args:
            query: Natural language query (e.g., "kung fu movies from the 80s")
            user_id: Optional user ID for personalization
            reference_movie: Optional reference movie for similarity
            top_n: Number of recommendations to return
            max_candidates: Maximum candidates to evaluate
            preference_mode: User preference - "accuracy" (prioritize query match),
                           "ratings" (prioritize ratings/popularity), or
                           "balanced" (equal weights, default)

        Returns:
            RecommendationResult object
        """
        # Validate preference_mode
        valid_modes = ["accuracy", "ratings", "balanced"]
        if preference_mode not in valid_modes:
            logger.warning(f"Invalid preference_mode '{preference_mode}', using 'balanced'")
            preference_mode = "balanced"
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING QUERY: '{query}'")
        logger.info(f"PREFERENCE MODE: {preference_mode.upper()}")
        logger.info(f"{'='*60}")

        # SITUATION + OUTCOME DETECTION (before any other processing)
        sit_out_result = detect_situation_outcome_query(query)
        sit_out_expansions = None  # Will hold expansion data if situation+outcome detected

        if sit_out_result['is_situation_outcome']:
            logger.info(f"\n   *** SITUATION + OUTCOME QUERY DETECTED ***")
            logger.info(f"   Situations: {sit_out_result['situations']}")
            logger.info(f"   Outcomes: {sit_out_result['outcomes']}")

            # Expand situation and outcome terms
            sit_out_expansions = expand_situation_outcome_terms(
                sit_out_result['situations'],
                sit_out_result['outcomes'],
                self.semantic_expander
            )

            # Log the structured extraction
            logger.info(f"\n   SITUATION extraction:")
            logger.info(f"      Primary (phrasal verbs): {sit_out_expansions['situation_primary']}")
            logger.info(f"      Secondary (context nouns): {sit_out_expansions['situation_secondary']}")
            logger.info(f"      Adjectives: {sit_out_expansions['situation_adjectives']}")

            logger.info(f"\n   OUTCOME extraction:")
            logger.info(f"      Primary (phrasal verbs): {sit_out_expansions['outcome_primary']}")
            logger.info(f"      Secondary (context nouns): {sit_out_expansions['outcome_secondary']}")
            logger.info(f"      Adjectives: {sit_out_expansions['outcome_adjectives']}")

            # Log expansions
            if sit_out_expansions['situation_expansions']:
                logger.info(f"\n   Situation expansions:")
                for term, expanded in sit_out_expansions['situation_expansions'].items():
                    logger.info(f"      '{term}' -> {len(expanded)} keywords")
            if sit_out_expansions['outcome_expansions']:
                logger.info(f"\n   Outcome expansions:")
                for term, expanded in sit_out_expansions['outcome_expansions'].items():
                    logger.info(f"      '{term}' -> {len(expanded)} keywords")
        else:
            logger.info(f"\n   Situation+Outcome: Not detected (clauses: {len(sit_out_result['clause_details'])})")

        # Step 1: Parse query
        logger.info("\n[1/4] Parsing query...")
        parsed_query = self.query_parser.parse(query)
        logger.info(f"   Extracted: {len(parsed_query.genres)} genres, "
                   f"{len(parsed_query.decades)} decades, "
                   f"{len(parsed_query.themes)} themes, "
                   f"{len(parsed_query.keywords)} keywords")

        # Step 2: Get candidate movies
        logger.info("\n[2/4] Getting candidate movies...")

        # SITUATION+OUTCOME: Use specialized candidate filtering
        # This bypasses the normal filtering to use outcome-based selection
        situation_outcome_mode = False
        if sit_out_result['is_situation_outcome'] and sit_out_expansions:
            logger.info("   [SITUATION+OUTCOME MODE] Using outcome-based candidate filtering")
            situation_outcome_mode = True

            # Get the actual clauses for dynamic filtering (use first of each)
            situation_clause = sit_out_result['situations'][0] if sit_out_result['situations'] else ""
            outcome_clause = sit_out_result['outcomes'][0] if sit_out_result['outcomes'] else ""

            # Get candidates - must match BOTH situation AND outcome
            # SITUATION: matches via TMDB keywords (content)
            # OUTCOME: matches via genre + zero-shot tags (experience)
            zero_shot_df = self.signal_fusion.zero_shot_tags if hasattr(self.signal_fusion, 'zero_shot_tags') else None
            candidates = get_situation_outcome_candidates(
                self.movies,
                sit_out_expansions,
                situation_clause,
                outcome_clause,
                self.semantic_expander,
                zero_shot_df,
                max_candidates
            )

            # Set flags for downstream processing
            actor_filter_applied = False
            curated_list_used = False
            expanded_keywords = None
            concept_groups = None
            keyword_search_used = False

            logger.info(f"   [SITUATION+OUTCOME MODE] Found {len(candidates)} candidates matching BOTH situation AND outcome")

        else:
            # Normal candidate selection (existing logic)
            candidates, actor_filter_applied, curated_list_used, expanded_keywords, concept_groups, keyword_search_used = self._get_candidate_movies(parsed_query, max_candidates)
        logger.info(f"   Found {len(candidates)} candidates")

        # Initialize concept_to_expansions (will be populated by _get_candidate_movies if semantic expansion used)
        concept_to_expansions = getattr(self, '_concept_to_expansions', {})
        if curated_list_used:
            logger.info(f"   CURATED LIST USED - will prioritize CF scoring")
        if concept_groups:
            logger.info(f"   Using CONCEPT-BASED SCORING with {len(concept_groups)} concepts (semantic expansion with AND logic)")
        elif expanded_keywords:
            logger.info(f"   Using {len(expanded_keywords)} EXPANDED KEYWORDS for scoring (semantic expansion applied)")

        # SEMANTIC FALLBACK: For rare theme queries (mafia, gladiator, heist)
        # Check if query has themes/keywords but very few movies have those tags
        # CRITICAL: Skip semantic expansion for actor queries (actor filter was applied)
        # CRITICAL: Skip semantic expansion for curated list queries (already high-quality)
        semantic_expansion_used = False
        should_expand = False
        tag_match_rate = 1.0  # Default: assume tags match (will be overridden if checking)

        if actor_filter_applied:
            logger.info("   Skipping semantic expansion (actor filter applied)")
            should_expand = False
        elif curated_list_used:
            logger.info("   Skipping semantic expansion (curated list provides high-quality candidates)")
            should_expand = False
        elif keyword_search_used:
            logger.info("   Skipping semantic expansion (keyword search found discriminating terms in TMDB metadata)")
            should_expand = False
        else:
            # Detect rare theme queries
            query_tags = parsed_query.themes + parsed_query.keywords
            if query_tags and len(candidates) > 0:
                # Load unified tags once
                unified_tags = self.zero_shot_integrator.merge_tags()

                # Check how many candidates actually have the queried tags
                tag_match_count = 0
                if unified_tags is not None and not unified_tags.empty:
                    for candidate in candidates[:100]:  # Sample first 100
                        # Check if candidate has any of the query tags
                        candidate_lower = candidate.lower()
                        tag_row = unified_tags[
                            unified_tags['title_norm'] == candidate_lower
                        ]
                        if len(tag_row) > 0:
                            for tag in query_tags:
                                if tag in tag_row.columns and tag_row[tag].values[0] > 0:
                                    tag_match_count += 1
                                    break

                # If 1-10% of candidates have the tag, it's a rare theme query - expand!
                # But if 0% match, don't expand (prevents "Inception Loop")
                tag_match_rate = tag_match_count / min(100, len(candidates))
                logger.info(f"   Tag match rate: {tag_match_count}/{min(100, len(candidates))} = {tag_match_rate:.1%}")

                if 0.01 <= tag_match_rate < 0.10:  # Between 1-10% have the tag
                    should_expand = True
                    logger.info(f"   Rare theme detected! Triggering semantic expansion...")
                elif tag_match_rate == 0.0:
                    logger.info(f"   No tag matches found - relying on TMDB metadata scoring (no reference movie)")
                    # CRITICAL: Don't set a reference movie when there are no tag matches!
                    # This allows the strengthened TMDB fallback scoring to work properly
                    # (query_score, content_score, theme_score check TMDB keywords/overview)

        # Set reference movie for similarity-based signals ONLY if we have tag matches
        # Otherwise, let the TMDB metadata-based scoring (fallbacks) handle it
        if reference_movie is None and len(candidates) > 0 and tag_match_rate > 0.0:
            reference_movie = candidates[0]
            logger.info(f"   Using '{reference_movie}' as reference for similarity-based signals")

        if should_expand:
            # Semantic expansion for rare themes
            candidates = self._expand_candidates_with_similarity(
                initial_candidates=candidates[:20],  # Use top 20 as seeds
                parsed_query=parsed_query,
                target_count=min(200, max_candidates)
            )
            semantic_expansion_used = True

        if len(candidates) == 0:
            logger.warning("   No candidates found! Returning empty results.")
            return RecommendationResult(
                query=query,
                parsed_query=parsed_query,
                recommendations=[],
                num_candidates=0
            )
        
        # Step 3: Compute signals
        logger.info("\n[3/4] Computing signals for all candidates...")

        # DEBUG: Log first few candidates
        logger.info(f"   DEBUG: First 3 candidates: {candidates[:3]}")
        logger.info(f"   DEBUG: Reference movie: {reference_movie}")
        logger.info(f"   DEBUG: User ID: {user_id}")

        # Combine themes and keywords for query context
        # SEMANTIC EXPANSION FIX: Use expanded_keywords if available (from semantic expansion)
        # Otherwise fall back to original parsed_query themes + keywords
        query_tags = parsed_query.themes + parsed_query.keywords

        # For QUERY SCORING (TMDB keyword matching), use expanded keywords if available
        if expanded_keywords:
            all_query_terms = expanded_keywords
            logger.info(f"   Using {len(expanded_keywords)} EXPANDED KEYWORDS for query scoring")
            logger.info(f"   [DEBUG] Expanded keywords: {expanded_keywords[:20] if len(expanded_keywords) > 20 else expanded_keywords}")
        else:
            all_query_terms = query_tags if query_tags else parsed_query.keywords
            logger.info(f"   Using original query terms for scoring: {all_query_terms}")

        # CRITICAL FIX: Pass TMDB metadata for fallback signals
        candidate_metadata = {
            title: self.title_to_metadata.get(title, {})
            for title in candidates
        }

        # NOUN ENTITY BOOST: Extract ONLY key entity nouns from query
        # These are specific concrete nouns (female, dog, lawyer, etc.) NOT genre/mood nouns
        # e.g., "female" in "strong female lead", "dog" in "my dog died"
        # KEY ENTITY NOUNS: Primary nouns that represent concrete subjects
        # These are specific concrete nouns (female, dog, lawyer, etc.) NOT genre/mood nouns
        # Ordered by priority - first match wins for compound phrases
        KEY_ENTITY_NOUNS_FOR_BOOST = {
            'female', 'male', 'woman', 'man', 'girl', 'boy',
            'dog', 'cat', 'animal', 'pet',
            'lawyer', 'doctor', 'cop', 'detective', 'soldier', 'teacher',
            'mother', 'father', 'daughter', 'son', 'wife', 'husband',
            'girlfriend', 'boyfriend', 'friend', 'family',
            'hero', 'heroine'
            # REMOVED: 'lead', 'protagonist' - these are modifiers, not primary entities
            # "female lead" should extract "female" as the entity, not both "female" AND "lead"
        }
        entity_nouns = []
        all_keywords = parsed_query.keywords + parsed_query.themes
        if all_keywords:
            # Extract only KEY entity nouns, not genre/mood nouns like 'thriller', 'noir'
            # CRITICAL: For compound phrases like "female lead", extract ONE entity noun, not multiple
            # The phrase represents ONE concept, not multiple separate entities
            for term in all_keywords:
                term_lower = term.lower()
                if ' ' in term_lower:
                    # Multi-word phrase: find the PRIMARY entity noun (first match wins)
                    # e.g., "female lead" -> extract "female" only (not "lead")
                    # e.g., "dog movie" -> extract "dog" only
                    words = term_lower.split()
                    for word in words:
                        if word in KEY_ENTITY_NOUNS_FOR_BOOST and word not in entity_nouns:
                            entity_nouns.append(word)
                            break  # ONE entity noun per compound phrase
                elif term_lower in KEY_ENTITY_NOUNS_FOR_BOOST and term_lower not in entity_nouns:
                    entity_nouns.append(term_lower)
            logger.info(f"   [NOUN ENTITY BOOST] Entity nouns for boosting: {entity_nouns}")

        # Build situation_outcome_data if in situation+outcome mode
        situation_outcome_data = None
        if situation_outcome_mode and sit_out_expansions:
            # Extract entity nouns from situation secondary terms (e.g., "girlfriend")
            sit_entity_nouns = list(sit_out_expansions.get('situation_secondary', []))
            # Situation phrases are the primary situation terms (e.g., "broke up")
            situation_phrases = list(sit_out_expansions.get('situation_primary', []))
            # Outcome phrases are the primary outcome terms (e.g., "cheer up")
            outcome_phrases = list(sit_out_expansions.get('outcome_primary', []))
            # Get outcome genres from the candidate filtering (already computed)
            # We need to get these from get_situation_outcome_candidates or recompute
            OUTCOME_TO_GENRE = {
                'cheer': ['comedy'], 'cheer up': ['comedy'], 'cheerful': ['comedy'],
                'feel good': ['comedy', 'family'], 'feel-good': ['comedy', 'family'],
                'uplifting': ['comedy', 'drama'], 'uplift': ['comedy', 'drama'],
                'happy': ['comedy', 'family'], 'funny': ['comedy'],
                'laugh': ['comedy'], 'lighthearted': ['comedy', 'family'],
                'scary': ['horror'], 'terrifying': ['horror'], 'frightening': ['horror'],
                'thrilling': ['thriller'], 'suspenseful': ['thriller'],
                'romantic': ['romance'], 'love': ['romance', 'drama'],
                'sad': ['drama'], 'cry': ['drama'], 'emotional': ['drama'],
                'inspiring': ['drama'], 'motivating': ['drama'],
                'relaxing': ['comedy', 'family'], 'comforting': ['comedy', 'drama', 'family'],
            }
            outcome_genres = set()
            for phrase in outcome_phrases:
                phrase_lower = phrase.lower()
                if phrase_lower in OUTCOME_TO_GENRE:
                    outcome_genres.update(OUTCOME_TO_GENRE[phrase_lower])

            situation_outcome_data = {
                'entity_nouns': sit_entity_nouns,
                'situation_phrases': situation_phrases,
                'outcome_phrases': outcome_phrases,
                'outcome_genres': outcome_genres,
                'situation_keywords': list(sit_out_expansions.get('situation_keywords', [])),
                'outcome_tags': list(sit_out_expansions.get('outcome_tags', []))
            }
            logger.info(f"   [SIT+OUT SCORING] Prepared situation_outcome_data:")
            logger.info(f"      entity_nouns: {sit_entity_nouns}")
            logger.info(f"      situation_phrases: {situation_phrases}")
            logger.info(f"      outcome_phrases: {outcome_phrases}")
            logger.info(f"      outcome_genres: {outcome_genres}")

        signal_scores = self.signal_fusion.compute_all_signals(
            candidate_movies=candidates,
            query_keywords=all_query_terms,  # Use combined tags for query matching
            query_tags=query_tags,
            reference_movie=reference_movie,
            user_id=user_id,
            rare_tags=self.rare_tags,
            candidate_metadata=candidate_metadata,  # NEW: TMDB metadata
            parsed_query=parsed_query,  # NEW: Full query context
            concept_groups=concept_groups,  # NEW: Concept groups for AND-logic scoring
            entity_nouns=entity_nouns if not situation_outcome_mode else None,  # Skip for sit+out (handled differently)
            concept_to_expansions=concept_to_expansions,  # NEW: Concept-based scoring (1 match per concept)
            situation_outcome_data=situation_outcome_data  # NEW: Situation+outcome scoring
        )

        logger.info(f"   Computed signals for {len(signal_scores)} movies")

        # DEBUG: Log first result
        if signal_scores:
            first = signal_scores[0]
            logger.info(f"   DEBUG: First movie signals: {first.movie_title}")
            logger.info(f"          CF={first.cf_score:.3f}, Content={first.content_score:.3f}, "
                       f"Theme={first.theme_score:.3f}, Tag={first.tag_score:.3f}")

        # V2 ENHANCEMENT: Add semantic similarity bonus to query_score
        if self.use_semantic_expansion:
            logger.info("\n[V2] Adding semantic similarity bonus to query scores...")
            for scored_movie in signal_scores:
                # Get movie tags from TMDB metadata (keywords + genres)
                metadata = candidate_metadata.get(scored_movie.movie_title, {})
                movie_tags = []

                # Add TMDB keywords
                if 'keywords' in metadata:
                    keywords = metadata['keywords']
                    if isinstance(keywords, (list, np.ndarray)) and len(keywords) > 0:
                        movie_tags.extend([str(kw).lower() for kw in keywords])

                # Add genres
                if 'genres' in metadata:
                    genres = metadata['genres']
                    if isinstance(genres, (list, np.ndarray)) and len(genres) > 0:
                        movie_tags.extend([str(g).lower() for g in genres])

                # Compute semantic similarity
                if movie_tags:
                    semantic_sim = self._compute_semantic_similarity(query, movie_tags)
                    # Add as bonus (scaled to max 0.3 to not overwhelm substring matches)
                    semantic_bonus = semantic_sim * 0.3
                    scored_movie.query_score += semantic_bonus

                    if semantic_bonus > 0.1:  # Log significant boosts
                        logger.debug(f"   {scored_movie.movie_title}: semantic_bonus={semantic_bonus:.3f}")

            logger.info(f"   Applied semantic similarity bonuses to {len(signal_scores)} movies")

        # Step 4: Score and rank
        logger.info("\n[4/4] Scoring and ranking...")

        # CRITICAL: Apply dynamic weighting based on query type
        self.scorer.adjust_weights_for_query(parsed_query)

        # INTERACTIVE PREFERENCE MODE: Apply user preference weights
        # This happens AFTER dynamic weighting, so user preference overrides defaults
        # Special logic (curated lists, two-tier) may further override these
        if preference_mode == "accuracy":
            logger.info(f"   Applying ACCURACY MODE weights (prioritize query match)")
            self.scorer.weights.query_weight = 0.60  # Prioritize TMDB keyword matching
            self.scorer.weights.tag_weight = 0.15    # Support with zero-shot tags
            self.scorer.weights.theme_weight = 0.10  # LDA themes
            self.scorer.weights.content_weight = 0.08  # Embeddings
            self.scorer.weights.cf_weight = 0.05     # Minimal ratings influence
            self.scorer.weights.sentiment_weight = 0.02
            self.scorer.weights.normalize()
        elif preference_mode == "ratings":
            logger.info(f"   Applying RATINGS MODE weights (prioritize ratings/popularity)")
            self.scorer.weights.cf_weight = 0.60     # Prioritize ratings/popularity
            self.scorer.weights.query_weight = 0.15  # Some query relevance
            self.scorer.weights.tag_weight = 0.10
            self.scorer.weights.theme_weight = 0.08
            self.scorer.weights.content_weight = 0.05
            self.scorer.weights.sentiment_weight = 0.02
            self.scorer.weights.normalize()
        else:  # balanced
            logger.info(f"   Applying BALANCED MODE weights (equal signal consideration)")
            # Use current weights from adjust_weights_for_query (no override)
            pass

        signal_dicts = [s.to_dict() for s in signal_scores]

        # Define variables for scoring logic
        num_themes = len(parsed_query.themes)
        TAG_THRESHOLD = 0.5  # Increased from 0.3 to filter out noisy tag matches

        # CURATED LIST SCORING: When curated lists are used,
        # use curated list ORDER + vote_count for ranking
        if curated_list_used:
            logger.info(f"   Using CURATED LIST ORDER + VOTE_COUNT for ranking")

            # Get the curated list for this query
            from src.curated_movies import get_curated_movies
            curated_list = get_curated_movies(parsed_query.raw_query.lower())

            # Rank by curated list position (priority) + vote_count
            def get_ranking_score(sig):
                title = sig['movie_title']
                metadata = candidate_metadata.get(title, {})
                vote_count = metadata.get('vote_count', 0)

                # Check if movie is in curated list and get its position
                # Movies earlier in the list get a massive boost
                curated_position_boost = 0
                for idx, curated_movie in enumerate(curated_list):
                    if curated_movie.lower() in title.lower():
                        # First movie gets +50000, second +49000, etc.
                        curated_position_boost = (len(curated_list) - idx) * 5000
                        break

                return vote_count + curated_position_boost

            # Sort by ranking score (descending)
            scored_movies = sorted(signal_dicts, key=get_ranking_score, reverse=True)

            # Convert to ScoredMovie format with ranking score
            from src.scoring import ScoredMovie
            scored_movies = [
                ScoredMovie(
                    movie_title=sig['movie_title'],
                    final_score=get_ranking_score(sig),  # Use ranking score with classic boost
                    cf_score=sig.get('cf_score', 0.0),
                    content_score=sig.get('content_score', 0.0),
                    theme_score=sig.get('theme_score', 0.0),
                    sentiment_score=sig.get('sentiment_score', 0.0),
                    tag_score=sig.get('tag_score', 0.0),
                    query_score=sig.get('query_score', 0.0)
                )
                for sig in scored_movies
            ]

        # TWO-TIER RANKING: For high-intensity theme queries
        # NEW: Content-first approach - Tier based on TMDB keyword matches (query_score)
        # Tier 1: Movies with strong content match (query_score >= 0.4 OR combined >= 0.5)
        # Tier 2: Weak matches (fallback to ratings)
        elif num_themes >= 2 and not actor_filter_applied:
            logger.info(f"   Applying TWO-TIER RANKING (high-intensity theme query: {num_themes} themes)")

            # NEW: Separate into tiers based on content matching (query + tag)
            # query_score >= 0.4 means 2+ TMDB keyword matches (discriminating!)
            # Combined >= 0.5 catches cases where both signals contribute
            def is_strong_match(sig):
                query_score = sig.get('query_score', 0.0)
                tag_score = sig.get('tag_score', 0.0)
                return query_score >= 0.4 or (query_score + tag_score) >= 0.5

            tier1_signals = [s for s in signal_dicts if is_strong_match(s)]
            tier2_signals = [s for s in signal_dicts if not is_strong_match(s)]

            logger.info(f"   Tier 1 (query_score >= 0.4 OR combined >= 0.5): {len(tier1_signals)} movies")
            logger.info(f"   Tier 2 (weak content match): {len(tier2_signals)} movies")

            # Score both tiers using the PREFERENCE_MODE weights (don't override!)
            # This ensures accuracy/ratings/balanced modes produce different results
            if tier1_signals:
                logger.info(f"   Scoring Tier 1 using preference_mode='{preference_mode}' weights")
                tier1_scored = self.scorer.score_movies(tier1_signals)

                # CRITICAL: Boost Tier 1 scores to ensure they ALWAYS rank above Tier 2
                # Add 10.0 to all Tier 1 final scores (since scores are 0-1, this ensures separation)
                for movie in tier1_scored:
                    movie.final_score += 10.0

                logger.info(f"   Tier 1 scored with preference_mode weights (+10.0 tier boost)")

                # Tier 2: Use same preference_mode weights (don't override!)
                # ALWAYS include Tier 2 to let all signals contribute (scientific approach)
                if tier2_signals:
                    logger.info(f"   Scoring Tier 2 using preference_mode='{preference_mode}' weights")
                    tier2_scored = self.scorer.score_movies(tier2_signals)
                    logger.info(f"   Tier 2 scored with preference_mode weights (no tier boost)")

                    # Combine tiers (Tier 1 will rank above Tier 2 due to +10 boost)
                    scored_movies = tier1_scored + tier2_scored
                    logger.info(f"   Combined: {len(tier1_scored)} tier1 + {len(tier2_scored)} tier2")
                else:
                    scored_movies = tier1_scored
                    logger.info(f"   Using Tier 1 only ({len(tier1_scored)} movies)")
            else:
                # No tier1 movies, fall back to normal scoring
                logger.info(f"   No Tier 1 movies found, using standard scoring")
                scored_movies = self.scorer.score_movies(signal_dicts)
        else:
            # Normal scoring for non-theme or low-intensity queries
            scored_movies = self.scorer.score_movies(signal_dicts)

        # ACCURACY MODE POST-PROCESSING: Hierarchical weighting with conditional zero-shot
        # NEW SCHEME: TMDB keywords 75%, LDA 10%, BERT 15%, zero-shot conditional
        if preference_mode == "accuracy":
            logger.info(f"   ACCURACY MODE: Applying hierarchical weighting (TMDB 75%, LDA 10%, BERT 15%, zero-shot conditional)")
            for movie in scored_movies:
                # Count TMDB keyword matches to determine zero-shot weight
                num_keyword_matches = movie.tmdb_keyword_matches

                # Conditional zero-shot inclusion based on TMDB keyword match count
                if num_keyword_matches >= 3:
                    # High confidence from TMDB keywords - skip zero-shot entirely
                    tag_weight = 0.0
                    tag_component = 0.0
                    logger.debug(f"   {movie.movie_title}: {num_keyword_matches} TMDB matches → skipping zero-shot")
                elif num_keyword_matches == 2:
                    # Medium confidence - use zero-shot as supplement (low weight)
                    tag_weight = 0.05  # 5% weight
                    tag_component = movie.tag_score * tag_weight
                    logger.debug(f"   {movie.movie_title}: {num_keyword_matches} TMDB matches → minimal zero-shot (5%)")
                else:
                    # Low confidence - need zero-shot to fill gaps (medium weight)
                    tag_weight = 0.15  # 15% weight (replaces some TMDB weight)
                    tag_component = movie.tag_score * tag_weight
                    logger.debug(f"   {movie.movie_title}: {num_keyword_matches} TMDB matches → using zero-shot (15%)")

                # Adjust TMDB weight based on whether zero-shot is used
                # If zero-shot is used heavily (15%), reduce TMDB from 75% to 60%
                if tag_weight >= 0.15:
                    tmdb_weight = 0.60  # Reduce to make room for zero-shot
                else:
                    tmdb_weight = 0.75  # Full weight when zero-shot is minimal/skipped

                # New weighted formula: hierarchical with conditional zero-shot
                movie.final_score = (
                    movie.query_score * tmdb_weight +      # TMDB keywords: 75% (or 60% if zero-shot heavy)
                    movie.theme_score * 0.10 +             # LDA themes: 10%
                    movie.content_score * 0.15 +           # BERT embeddings: 15%
                    tag_component                          # Zero-shot: 0-15% (conditional)
                )

            # Re-sort by new final_score
            scored_movies.sort(key=lambda x: x.final_score, reverse=True)
            logger.info(f"   Re-ranked by hierarchical weighting (TMDB-first, conditional zero-shot)")

        # DUAL-TRACK: Check if query is complex and should use dual-track (BEFORE concept penalty)
        # Returns None, "entity_mood", or "multi_theme"
        # SITUATION+OUTCOME OVERRIDE: When situation+outcome is detected, skip dual_track entirely
        # Context nouns like "girlfriend" should not trigger entity_mood - situation+outcome handles its own scoring
        if situation_outcome_mode:
            dual_track_mode = None
            logger.info(f"   [SITUATION+OUTCOME] Skipping dual_track detection (single combined list)")
        else:
            dual_track_mode = self._should_use_dual_track(parsed_query)

        # V4: Apply concept coverage penalty for multi-concept queries
        # SKIP for dual-track queries - dual-track separation handles entity vs mood distinction
        if concept_groups and len(concept_groups) > 1 and not dual_track_mode:
            scored_movies = self._apply_concept_coverage_penalty(
                scored_movies, concept_groups, self.movies
            )
            logger.info(f"   Applied concept coverage penalty for {len(concept_groups)}-concept query")
        elif dual_track_mode and concept_groups and len(concept_groups) > 1:
            logger.info(f"   SKIPPING concept penalty for dual-track query (tracks handle entity vs mood)")

        top_movies = scored_movies[:top_n]
        logger.info(f"   Selected top {len(top_movies)} recommendations")

        entity_track = None
        mood_track = None
        # For multi-theme mode, we use theme_track_1 and theme_track_2 but map to entity_track/mood_track for output
        theme_track_1 = None
        theme_track_2 = None

        if dual_track_mode == "multi_theme":
            # MULTI-THEME MODE: No entity, two theme groups
            # Track 1: Theme Group 1 (e.g., trauma/recovery)
            # Track 2: Theme Group 2 (e.g., humor)
            logger.info(f"\n   MULTI-THEME MODE ACTIVATED:")
            theme_groups = parsed_query.theme_groups
            logger.info(f"   Theme Group 1: {theme_groups[0]}")
            logger.info(f"   Theme Group 2: {theme_groups[1]}")
            logger.info(f"   BOOSTED zero-shot tag weight: 0.30 (vs 0.10-0.15 for entity queries)")

            # Compute theme-specific scores for each movie
            for movie in scored_movies:
                # Score for Theme Group 1
                theme1_score = self._compute_theme_group_score(movie.movie_title, theme_groups[0])
                movie.theme1_score = theme1_score

                # Score for Theme Group 2
                theme2_score = self._compute_theme_group_score(movie.movie_title, theme_groups[1])
                movie.theme2_score = theme2_score

            # THEME TRACK 1: Sort by theme group 1 relevance
            # BOOSTED zero-shot weight (0.30) for thematic queries
            def theme1_score_func(m):
                base = getattr(m, 'theme1_score', 0.0)
                tag_boost = m.tag_score * 0.30  # Boosted zero-shot weight for multi-theme
                if preference_mode == "accuracy":
                    return base + tag_boost + m.query_score * 0.25 + m.cf_score * 0.05
                elif preference_mode == "ratings":
                    return base + tag_boost + m.query_score * 0.15 + m.cf_score * 0.35
                else:  # balanced
                    return base + tag_boost + m.query_score * 0.20 + m.cf_score * 0.20

            theme_track_1 = sorted(scored_movies, key=theme1_score_func, reverse=True)[:top_n]
            logger.info(f"   Theme Track 1 ({theme_groups[0]}): {len(theme_track_1)} results")

            # THEME TRACK 2: Sort by theme group 2 relevance
            # BOOSTED zero-shot weight (0.30) for thematic queries
            def theme2_score_func(m):
                base = getattr(m, 'theme2_score', 0.0)
                tag_boost = m.tag_score * 0.30  # Boosted zero-shot weight for multi-theme
                if preference_mode == "accuracy":
                    return base + tag_boost + m.query_score * 0.25 + m.cf_score * 0.05
                elif preference_mode == "ratings":
                    return base + tag_boost + m.query_score * 0.15 + m.cf_score * 0.35
                else:  # balanced
                    return base + tag_boost + m.query_score * 0.20 + m.cf_score * 0.20

            theme_track_2 = sorted(scored_movies, key=theme2_score_func, reverse=True)[:top_n]
            logger.info(f"   Theme Track 2 ({theme_groups[1]}): {len(theme_track_2)} results")

            # Map to entity_track/mood_track for output compatibility
            entity_track = theme_track_1
            mood_track = theme_track_2

        elif dual_track_mode == "entity_mood":
            # ENTITY_MOOD MODE: Original dual-track behavior
            logger.info(f"\n   ENTITY_MOOD DUAL-TRACK MODE ACTIVATED:")
            logger.info(f"   Generating separate entity and mood tracks using preference_mode='{preference_mode}'...")

            # ENTITY TRACK: Score each movie by entity-relevance
            # Adjust formula based on preference_mode
            def entity_score(m):
                if preference_mode == "accuracy":
                    # Prioritize query matching (TMDB keywords + zero-shot tags + embeddings)
                    return m.query_score * 2.0 + m.tag_score + m.content_score
                elif preference_mode == "ratings":
                    # Include CF score for popularity/quality
                    return m.query_score + m.tag_score + m.content_score + m.cf_score * 0.5
                else:  # balanced
                    # Balance content matching with ratings
                    return m.query_score + m.tag_score + m.content_score + m.cf_score * 0.3

            entity_scored = sorted(scored_movies, key=entity_score, reverse=True)
            logger.info(f"   Entity track: Sorted using {preference_mode} mode weighting")

            # MOOD TRACK: Score each movie by mood-relevance
            # If themes exist: compute mood theme matching score from zero-shot tags
            # If NO themes: fall back to CF (popularity/ratings) + content
            has_themes = len(parsed_query.themes) > 0 or len(parsed_query.moods) > 0

            if has_themes:
                # Compute mood theme score for each movie based on zero-shot tag matching
                # This specifically checks if movie has the inferred mood themes (uplifting, comforting, etc.)
                logger.info(f"   Mood track: Computing mood theme scores for {len(parsed_query.themes)} themes: {parsed_query.themes[:5]}...")

                for movie in scored_movies:
                    # Compute how well this movie matches the mood themes
                    mood_score = self._compute_mood_theme_score(movie.movie_title, parsed_query.themes)
                    # Store in a custom attribute
                    movie.mood_theme_score = mood_score

                # Sort by mood theme score with CF weight adjusted by preference_mode
                def mood_score_func(m):
                    mood_base = getattr(m, 'mood_theme_score', 0.0)
                    if preference_mode == "accuracy":
                        # Prioritize mood theme matching
                        return mood_base + m.cf_score * 0.2
                    elif preference_mode == "ratings":
                        # Weight ratings more heavily
                        return mood_base + m.cf_score * 0.6
                    else:  # balanced
                        # Balance mood themes with ratings
                        return mood_base + m.cf_score * 0.4

                mood_scored = sorted(scored_movies, key=mood_score_func, reverse=True)
                logger.info(f"   Mood track: Sorted by mood_theme_score with {preference_mode} mode CF weighting")
            else:
                # Fall back to CF + content (no themes extracted)
                # This gives popular/well-rated movies that match the content
                mood_scored = sorted(scored_movies, key=lambda m: (m.cf_score + m.content_score), reverse=True)
                logger.info(f"   Mood track: No themes detected, falling back to CF+content scoring (popularity)")

            # Split results: Even split preferred, but flexible if limited matches
            # If entity track has <5 strong matches, give it fewer spots
            entity_limit = min(top_n, len([m for m in entity_scored if m.query_score > 0.2]))
            mood_limit = top_n

            entity_track = entity_scored[:entity_limit]
            mood_track = mood_scored[:mood_limit]

            logger.info(f"   Entity track: {len(entity_track)} results (query_score prioritized)")
            logger.info(f"   Mood track: {len(mood_track)} results (theme+sentiment prioritized)")

        # SITUATION+OUTCOME SCORING: Special handling for "my girlfriend broke up with me - cheer me up" queries
        # Mirrors entity_mood logic exactly
        if situation_outcome_mode and sit_out_expansions:
            logger.info(f"\n   SITUATION+OUTCOME SCORING MODE ACTIVATED:")
            logger.info(f"   Generating separate situation and outcome tracks using preference_mode='{preference_mode}'...")

            # SITUATION TRACK: Score each movie by situation-relevance
            # Adjust formula based on preference_mode
            def situation_score(m):
                if preference_mode == "accuracy":
                    # Prioritize query matching (TMDB keywords + zero-shot tags + embeddings)
                    return m.query_score * 2.0 + m.tag_score + m.content_score
                elif preference_mode == "ratings":
                    # Include CF score for popularity/quality
                    return m.query_score + m.tag_score + m.content_score + m.cf_score * 0.5
                else:  # balanced
                    # Balance content matching with ratings
                    return m.query_score + m.tag_score + m.content_score + m.cf_score * 0.3

            situation_scored = sorted(scored_movies, key=situation_score, reverse=True)
            logger.info(f"   Situation track: Sorted using {preference_mode} mode weighting")

            # OUTCOME TRACK: Score each movie by outcome-relevance
            # If themes exist: compute outcome theme matching score from zero-shot tags
            # If NO themes: fall back to CF (popularity/ratings) + content
            has_themes = len(parsed_query.themes) > 0 or len(parsed_query.moods) > 0

            if has_themes:
                # Compute outcome theme score for each movie based on zero-shot tag matching
                # This specifically checks if movie has the inferred outcome themes (uplifting, comforting, etc.)
                logger.info(f"   Outcome track: Computing outcome theme scores for {len(parsed_query.themes)} themes: {parsed_query.themes[:5]}...")

                for movie in scored_movies:
                    # Compute how well this movie matches the outcome themes
                    outcome_score = self._compute_mood_theme_score(movie.movie_title, parsed_query.themes)
                    # Store in a custom attribute
                    movie.outcome_theme_score = outcome_score

                # Sort by outcome theme score with CF weight adjusted by preference_mode
                def outcome_score_func(m):
                    outcome_base = getattr(m, 'outcome_theme_score', 0.0)
                    if preference_mode == "accuracy":
                        # Prioritize outcome theme matching
                        return outcome_base + m.cf_score * 0.2
                    elif preference_mode == "ratings":
                        # Weight ratings more heavily
                        return outcome_base + m.cf_score * 0.6
                    else:  # balanced
                        # Balance outcome themes with ratings
                        return outcome_base + m.cf_score * 0.4

                outcome_scored = sorted(scored_movies, key=outcome_score_func, reverse=True)
                logger.info(f"   Outcome track: Sorted by outcome_theme_score with {preference_mode} mode CF weighting")
            else:
                # Fall back to CF + content (no themes extracted)
                # This gives popular/well-rated movies that match the content
                outcome_scored = sorted(scored_movies, key=lambda m: (m.cf_score + m.content_score), reverse=True)
                logger.info(f"   Outcome track: No themes detected, falling back to CF+content scoring (popularity)")

            # Split results: Even split preferred, but flexible if limited matches
            # If situation track has <5 strong matches, give it fewer spots
            situation_limit = min(top_n, len([m for m in situation_scored if m.query_score > 0.2]))
            outcome_limit = top_n

            entity_track = situation_scored[:situation_limit]
            mood_track = outcome_scored[:outcome_limit]

            logger.info(f"   Situation track: {len(entity_track)} results (query_score prioritized)")
            logger.info(f"   Outcome track: {len(mood_track)} results (theme+sentiment prioritized)")

        # Create result
        result = RecommendationResult(
            query=query,
            parsed_query=parsed_query,
            recommendations=top_movies,
            num_candidates=len(candidates),
            entity_track=entity_track,
            mood_track=mood_track,
            dual_track_mode=bool(dual_track_mode)  # Convert string to bool for output
        )

        logger.info(f"\n{'='*60}")
        logger.info("RECOMMENDATION COMPLETE!")
        logger.info(f"{'='*60}\n")

        return result
    
    def recommend_similar(self,
                         movie_title: str,
                         top_n: int = 10,
                         max_candidates: int = 500) -> RecommendationResult:
        """
        Find movies similar to a given movie.
        
        Args:
            movie_title: Reference movie title
            top_n: Number of recommendations
            max_candidates: Maximum candidates to evaluate
            
        Returns:
            RecommendationResult object
        """
        # Get movie metadata
        movie = self.data_loader.get_movie_by_title(movie_title)
        
        if movie is None:
            logger.warning(f"Movie '{movie_title}' not found")
            # Return empty result
            empty_query = ParsedQuery(raw_query=f"similar to {movie_title}")
            return RecommendationResult(
                query=f"similar to {movie_title}",
                parsed_query=empty_query,
                recommendations=[],
                num_candidates=0
            )
        
        # Build query from movie metadata
        query_parts = []
        
        if 'genres' in movie and pd.notna(movie['genres']):
            query_parts.append(movie['genres'])
        
        query = f"movies like {movie_title}"
        if query_parts:
            query += " " + " ".join(query_parts)
        
        # Use regular recommend with reference movie
        return self.recommend(
            query=query,
            reference_movie=movie_title,
            top_n=top_n,
            max_candidates=max_candidates
        )


# Testing code
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING MOVIE RECOMMENDER")
    print("="*60 + "\n")
    
    # Initialize
    print("Initializing recommender...")
    recommender = MovieRecommender()
    print("✅ Recommender initialized\n")
    
    # Test 1: Simple query
    print("Test 1: Simple genre query")
    print("-" * 60)
    result = recommender.recommend("action movies", top_n=5)
    result.print_summary(top_n=5)
    
    # Test 2: Complex query with decade
    print("\n\nTest 2: Complex query with filters")
    print("-" * 60)
    result = recommender.recommend("kung fu movies from the 80s", top_n=5)
    result.print_summary(top_n=5)
    
    # Test 3: Query with themes
    print("\n\nTest 3: Theme-based query")
    print("-" * 60)
    result = recommender.recommend("dark psychological thriller", top_n=5)
    result.print_summary(top_n=5)
    
    # Test 4: Similar movies
    print("\n\nTest 4: Similar movie recommendations")
    print("-" * 60)
    result = recommender.recommend_similar("Toy Story", top_n=5)
    result.print_summary(top_n=5)
    
    print("\n\n" + "="*60)
    print("ALL TESTS COMPLETE! ✅")
    print("="*60)
    print("\n💡 Usage:")
    print("  from src.recommender import MovieRecommender")
    print("  recommender = MovieRecommender()")
    print('  results = recommender.recommend("kung fu movies from the 80s")')
    print("  results.print_summary()")