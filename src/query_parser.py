"""
Query Parser Module
Parses natural language movie queries into structured filters and keywords.

Fixes the "80s" decade extraction bug from CapstoneMasterV4.
Extracts: genres, decades, years, moods, themes, and search keywords.

Examples:
    "kung fu movies from the 80s" -> {decade: 1980, genres: [], keywords: ["kung fu"]}
    "dark thriller from 2020" -> {year: 2020, moods: ["dark"], genres: ["thriller"]}
    "funny heist movie" -> {moods: ["funny"], keywords: ["heist"]}
"""

import re
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
import logging
import spacy

# PHASE 1: Import EntityExtractor for NER (detect-only, no behavioral changes)
from entity_extractor import EntityExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured representation of a parsed query."""

    # Original query
    raw_query: str

    # Extracted filters
    genres: List[str] = field(default_factory=list)
    decades: List[int] = field(default_factory=list)  # [1980, 1990]
    years: List[int] = field(default_factory=list)    # [2020, 2021]
    year_range: Optional[tuple] = None                 # (1980, 1989) or None
    moods: List[str] = field(default_factory=list)    # ["dark", "funny"]
    themes: List[str] = field(default_factory=list)   # ["heist", "war"]

    # Search keywords (for semantic search)
    keywords: List[str] = field(default_factory=list)

    # Query without filter words (cleaned)
    cleaned_query: str = ""

    # Adult context flag (exclude family-friendly content)
    adult_context: bool = False

    # Track which QUERY_EXPANSIONS were triggered (for concept grouping)
    # Example: ['historical', 'revenge'] instead of all 12+ expanded tags
    expanded_categories: List[str] = field(default_factory=list)

    # Store original themes/keywords BEFORE query parser expansion
    # Used for concept-based scoring to avoid treating expansion results as separate concepts
    original_themes: List[str] = field(default_factory=list)
    original_keywords: List[str] = field(default_factory=list)

    # Named entities (actors, directors, studios) - detected via NER
    # PHASE 1: Detect-only, no behavioral changes
    actors: List[str] = field(default_factory=list)      # ["Bill Murray", "Tom Hanks"]
    directors: List[str] = field(default_factory=list)   # ["Quentin Tarantino"]
    studios: List[str] = field(default_factory=list)     # ["Pixar", "Marvel"]

    # Multi-theme support: Content nouns extracted via spaCy POS tagging
    # Used when no entities detected to identify thematic concepts
    content_nouns: List[str] = field(default_factory=list)  # ["trauma", "humor"]
    # Theme groups: clusters of related nouns based on dependency parsing
    # E.g., [["trauma", "recovery"], ["humor"]] for "trauma and recovery through humor"
    theme_groups: List[List[str]] = field(default_factory=list)

    def __str__(self):
        parts = [f"Query: '{self.raw_query}'"]
        if self.adult_context:
            parts.append(f"  Adult Context: True (excluding family-friendly)")
        # PHASE 1: Display detected entities
        if self.actors:
            parts.append(f"  Actors: {self.actors}")
        if self.directors:
            parts.append(f"  Directors: {self.directors}")
        if self.studios:
            parts.append(f"  Studios: {self.studios}")
        if self.genres:
            parts.append(f"  Genres: {self.genres}")
        if self.decades:
            parts.append(f"  Decades: {[f'{d}s' for d in self.decades]}")
        if self.years:
            parts.append(f"  Years: {self.years}")
        if self.year_range:
            parts.append(f"  Year Range: {self.year_range[0]}-{self.year_range[1]}")
        if self.moods:
            parts.append(f"  Moods: {self.moods}")
        if self.themes:
            parts.append(f"  Themes: {self.themes}")
        if self.keywords:
            parts.append(f"  Keywords: {self.keywords}")
        # Multi-theme mode fields
        if self.content_nouns:
            parts.append(f"  Content Nouns: {self.content_nouns}")
        if self.theme_groups:
            parts.append(f"  Theme Groups: {self.theme_groups}")
        return "\n".join(parts)


class QueryParser:
    """Parse natural language movie queries into structured filters."""
    
    # Genre keywords (including plural forms for natural language queries)
    GENRES = {
        'action', 'comedy', 'comedies', 'drama', 'dramas', 'thriller', 'thrillers',
        'horror', 'romance', 'romances', 'romantic', 'sci-fi', 'science fiction', 'fantasy',
        'mystery', 'mysteries', 'crime', 'adventure', 'adventures', 'animation',
        'documentary', 'documentaries', 'musical', 'musicals', 'western', 'westerns',
        'war', 'biography', 'biographies', 'historical', 'superhero', 'superheroes',
        # Compound genres
        'romantic comedy', 'romantic comedies', 'action comedy', 'action thriller',
        'sci-fi thriller', 'sci-fi horror', 'horror comedy', 'romantic drama'
    }
    
    # Mood/tone keywords
    MOODS = {
        'dark', 'funny', 'lighthearted', 'serious', 'intense', 'suspenseful',
        'emotional', 'uplifting', 'depressing', 'scary', 'thrilling',
        'romantic', 'heartwarming', 'gritty', 'violent', 'psychological'
    }
    
    # Common theme keywords
    THEMES = {
        'heist', 'kung fu', 'martial arts', 'sports', 'gangster', 'mafia',
        'detective', 'spy', 'zombie', 'vampire', 'alien', 'robot',
        'time travel', 'space', 'superhero', 'coming of age', 'coming-of-age', 'revenge',
        'survival', 'prison', 'family', 'friendship', 'betrayal',
        'supernatural', 'paranormal', 'ghost', 'horror', 'witchcraft',
        # LGBTQ+ themes (Level 4 attribute support)
        'lgbtq', 'lgbt', 'lgbtq+', 'queer', 'gay', 'lesbian'
    }

    # Query expansions - maps category-level queries to component tags
    QUERY_EXPANSIONS = {
        # Theme-based
        'supernatural': ['ghost', 'haunted house', 'possession', 'witchcraft', 'paranormal',
                         'zombie', 'vampire', 'werewolf', 'poltergeist', 'dimension', 'spiritual'],
        'sports': ['sports', 'football', 'basketball', 'baseball', 'boxing', 'soccer',
                   'hockey', 'olympics', 'skating', 'running'],
        'crime': ['mafia', 'organized crime', 'gangster', 'mob', 'crime', 'noir', 'heist',
                  'detective', 'serial killer', 'murder mystery', 'drug cartel'],
        'political': ['political thriller', 'politics', 'government', 'conspiracy', 'espionage',
                      'intelligence agency', 'washington', 'corruption', 'election'],
        'war': ['war film', 'military', 'combat', 'world war ii', 'world war i', 'civil war',
                'vietnam war', 'iraq', 'afghanistan', 'fighting', 'invasion'],
        'military': ['war film', 'military', 'combat', 'world war ii', 'world war i', 'civil war',
                     'vietnam war', 'iraq', 'afghanistan', 'soldier', 'army', 'navy', 'marines',
                     'troops', 'battle', 'warfare', 'veteran'],
        'horror': ['vampire', 'zombie', 'psychological horror', 'ghost', 'haunted house',
                   'possession', 'witchcraft', 'paranormal'],
        'music': ['musical', 'music film', 'concert film', 'dance', 'ballet', 'band',
                  'artist', 'broadway', 'show'],
        'historical': ['ancient rome', 'ancient greece', 'medieval', 'victorian era',
                       'ancient china', 'samurai', 'wild west', 'elizabethan', 'renaissance',
                       'based on true story', 'historical', 'biography'],
        'nature': ['animals', 'pets', 'dinosaurs', 'nature', 'natural disaster',
                   'jungle', 'forest', 'desert', 'mountain', 'ocean', 'wildlife'],
        'sci-fi': ['sci-fi', 'science fiction', 'cyberpunk', 'dystopian', 'post-apocalyptic',
                   'time travel', 'space opera', 'alien', 'robot ai'],

        # Mood-based
        'feel-good': ['inspiring', 'uplifting', 'comforting', 'feel-good', 'family friendly',
                      'heartwarming', 'hope', 'redemption', 'friendship'],
        'dark': ['dark', 'bleak', 'psychological horror', 'noir', 'gritty', 'tragic',
                 'bittersweet', 'depression', 'terminal illness'],
        'intense': ['intense', 'fast-paced', 'thriller', 'suspenseful', 'gritty',
                    'action', 'fighting'],

        # Character-based
        'strong female lead': ['strong female lead', 'female protagonist'],
        'underdog': ['underdog story', 'overcoming adversity', 'survival', 'redemption'],

        # LGBTQ+ themes (Level 4 attribute support)
        'lgbtq': ['lgbtq', 'lgbt', 'queer', 'gay', 'lesbian', 'lgbtq+'],
        'lgbt': ['lgbtq', 'lgbt', 'queer', 'gay', 'lesbian'],
        'lgbtq+': ['lgbtq', 'lgbt', 'queer', 'gay', 'lesbian', 'lgbtq+'],
        'queer': ['lgbtq', 'lgbt', 'queer'],

        # Coming-of-age (support both hyphenated and non-hyphenated)
        'coming-of-age': ['coming of age', 'coming-of-age', 'teenage', 'teen', 'adolescence'],
        'coming of age': ['coming of age', 'coming-of-age', 'teenage', 'teen', 'adolescence'],

        # Relationship-based
        'romance': ['romantic', 'love triangle', 'forbidden love', 'slow-burn',
                    'unrequited love', 'affair', 'mistress'],
        'family': ['family friendly', 'family drama', 'father son', 'mother son',
                   'siblings', 'found family', 'reunion', 'parenthood'],

        # Emotional/Situational (adult context)
        'breakup': ['romantic', 'bittersweet', 'heartbreak', 'relationship drama',
                    'moving on', 'love', 'romance'],
        'broke up': ['romantic', 'bittersweet', 'heartbreak', 'relationship drama',
                     'moving on', 'love', 'romance'],
        'break up': ['romantic', 'bittersweet', 'heartbreak', 'relationship drama',
                     'moving on', 'love', 'romance'],
        'dumped': ['romantic', 'bittersweet', 'heartbreak', 'relationship drama',
                   'moving on', 'love', 'romance'],
        'girlfriend': ['romantic', 'relationship drama', 'love', 'romance', 'dating'],
        'boyfriend': ['romantic', 'relationship drama', 'love', 'romance', 'dating'],
        'heartbreak': ['romantic', 'bittersweet', 'tragic', 'love', 'romance',
                       'relationship drama', 'heartbreak'],
        'divorce': ['marriage', 'relationship drama', 'family drama', 'romantic',
                    'separation', 'moving on'],
        'cheer me up': ['uplifting', 'feel-good', 'comforting', 'heartwarming',
                        'inspiring', 'hope', 'friendship', 'funny'],
        'feel better': ['uplifting', 'feel-good', 'comforting', 'heartwarming',
                        'inspiring', 'hope', 'friendship', 'funny'],

        # Holidays
        'halloween': ['halloween', 'horror', 'supernatural', 'ghost', 'haunted house',
                      'witchcraft', 'scary', 'spooky', 'monster'],
        'christmas': ['christmas', 'holiday', 'santa', 'festive', 'family friendly',
                      'heartwarming', 'winter', 'snow'],

        # Life stages / Settings
        'college': ['college', 'university', 'student', 'campus', 'coming of age',
                    'young adult', 'fraternity', 'sorority'],
        'high school': ['high school', 'teen', 'teenager', 'coming of age', 'school',
                        'adolescence', 'prom', 'teenage'],

        # Professions
        'lawyer': ['lawyer', 'courtroom', 'legal drama', 'attorney', 'trial', 'justice'],
        'lawyers': ['lawyer', 'courtroom', 'legal drama', 'attorney', 'trial', 'justice'],
        'doctor': ['doctor', 'medical drama', 'hospital', 'surgeon', 'healthcare'],

        # Vacation/Travel
        'vacation': ['travel', 'beach', 'tropical', 'adventure', 'journey', 'exotic',
                     'tourism', 'road trip'],

        # Specific genre combinations
        'political thriller': ['political thriller', 'politics', 'government', 'conspiracy',
                               'espionage', 'intelligence agency', 'corruption'],

        # Real events
        'based on real events': ['based on true story', 'true events', 'real life',
                                  'historical', 'biography', 'true story'],
        'based on true events': ['based on true story', 'true events', 'real life',
                                  'historical', 'biography', 'true story'],
        'based on a true story': ['based on true story', 'true events', 'real life', 'biography'],
        'true story': ['based on true story', 'true events', 'real life', 'biography'],
        'true events': ['based on true story', 'true events', 'real life', 'biography'],
        'real events': ['based on true story', 'true events', 'real life', 'historical'],
    }

    # Keywords/themes that should also imply a genre (for filtering)
    # This allows "military themes" to also include War genre movies
    KEYWORD_TO_GENRE = {
        'military': 'war',
        'soldier': 'war',
        'soldiers': 'war',
        'army': 'war',
        'navy': 'war',
        'marines': 'war',
        'combat': 'war',
        'battlefield': 'war',
        'troops': 'war',
        'veteran': 'war',
        'warfare': 'war',
        'world war': 'war',
        'vietnam': 'war',
        'wwii': 'war',
        'ww2': 'war',
    }

    # Adult context keywords (triggers exclusion of family-friendly content)
    ADULT_CONTEXT_KEYWORDS = {
        # Direct breakup terms
        'breakup', 'break up', 'broke up', 'broke up with', 'dumped', 'split up',
        'relationship ended', 'separated', 'broke my heart',

        # Relationship terms
        'girlfriend', 'boyfriend', 'ex-girlfriend', 'ex-boyfriend', 'ex',
        'my girl', 'my guy', 'my woman', 'my man',

        # Relationship issues
        'divorce', 'affair', 'cheating', 'heartbreak', 'relationship drama',
        'dating', 'one night stand', 'hookup', 'relationship problems',
        'relationship trouble', 'love life'
    }

    # Emotion words for context-aware mood inference
    # Negative emotions (trigger uplifting/comforting themes)
    NEGATIVE_EMOTIONS = {
        # Loss/death
        'died', 'dead', 'dying', 'death', 'lost', 'loss', 'funeral', 'grief', 'mourning',
        'passed away', 'passed', 'terminal', 'illness', 'sick', 'disease',

        # Breakup/relationship
        'broke up', 'breakup', 'break up', 'dumped', 'split up', 'divorced', 'divorce',
        'separated', 'separation', 'heartbroken', 'heartbreak',

        # Relationship distance/estrangement (added for subtle queries)
        "haven't talked", "haven't spoken", "not talking", "not speaking",
        "haven't seen", "not seen", "haven't met", "not met",
        'miss', 'missing', 'estranged', 'distant', 'arguing', 'fighting', 'fight',
        'not close', 'grown apart', 'drifted apart', 'cut off', 'no contact',

        # Emotional states
        'depressed', 'depression', 'sad', 'sadness', 'lonely', 'loneliness', 'alone',
        'miserable', 'unhappy', 'upset', 'crying', 'devastated', 'hopeless', 'helpless',
        'anxious', 'anxiety', 'stressed', 'stress', 'worried', 'fear', 'afraid', 'scared',
        'angry', 'rage', 'furious', 'frustrated', 'irritated', 'annoyed',
        'hurt', 'pain', 'painful', 'suffering', 'anguish', 'agony', 'torment',
        'guilty', 'shame', 'regret', 'remorse', 'disappointed', 'disappointment',
        'bitter', 'resentment', 'jealous', 'envy',

        # Life events
        'fired', 'laid off', 'unemployed', 'jobless', 'bankrupt', 'bankruptcy',
        'failed', 'failure', 'rejected', 'rejection', 'abandoned', 'betrayed',
        'bullied', 'bullying', 'abused', 'abuse', 'traumatized', 'trauma',

        # Mental states
        'overwhelmed', 'exhausted', 'tired', 'drained', 'burned out', 'burnout',
        'confused', 'lost', 'stuck', 'trapped', 'numb', 'empty', 'void',

        # Physical
        'injured', 'injury', 'accident', 'hospitalized', 'surgery', 'disabled'
    }

    # Positive emotions (trigger exciting/fun themes)
    POSITIVE_EMOTIONS = {
        # Happy states
        'happy', 'excited', 'excited', 'thrilled', 'joy', 'joyful', 'cheerful',
        'delighted', 'ecstatic', 'elated', 'euphoric', 'blissful',

        # Celebration
        'celebrating', 'celebration', 'party', 'birthday', 'anniversary',
        'graduation', 'wedding', 'promoted', 'promotion', 'victory', 'won',

        # Energy/motivation
        'pumped', 'pump up', 'pump him up', 'pump her up', 'pump them up', 'pump me up',
        'motivated', 'inspired', 'energized', 'enthusiastic', 'hyped', 'hype up',
        'psyched', 'psyched up', 'fired up', 'amped', 'amped up',
        'confident', 'optimistic', 'hopeful', 'positive',

        # Life events
        'pregnant', 'expecting', 'new baby', 'newborn', 'engaged',
        'vacation', 'holiday', 'travel', 'adventure',

        # Social
        'date', 'dating', 'first date', 'romantic', 'love', 'loving',
        'friends', 'reunion', 'gathering', 'hangout'
    }

    # Request words (indicate user wants recommendation)
    REQUEST_WORDS = {
        'recommend', 'recommendation', 'need', 'want', 'looking', 'looking for',
        'suggest', 'suggestion', 'help', 'show', 'find', 'give', 'tell',
        'can you', 'could you', 'would you', 'please'
    }
    
    # Decade patterns
    DECADE_PATTERNS = [
        (r'\b(19|20)(\d0)s\b', 'numeric_s'),           # 1980s, 2000s
        (r'\b(\d0)s\b', 'short_s'),                     # 80s, 90s
        (r'\b(nineteen|twenty)[-\s]?(\w+)\b', 'word'), # nineteen eighties
    ]
    
    # Decade word mappings
    DECADE_WORDS = {
        'twenties': 20, 'thirties': 30, 'forties': 40, 'fifties': 50,
        'sixties': 60, 'seventies': 70, 'eighties': 80, 'nineties': 90,
        'tens': 10, 'zeros': 0, 'aughts': 0, 'oughts': 0
    }
    
    def __init__(self):
        """Initialize parser with compiled regex patterns."""
        self.decade_regex = [
            (re.compile(pattern, re.IGNORECASE), ptype)
            for pattern, ptype in self.DECADE_PATTERNS
        ]

        # Year patterns
        self.year_pattern = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')
        self.year_range_pattern = re.compile(r'\b(19\d{2}|20[0-2]\d)\s*-\s*(19\d{2}|20[0-2]\d)\b')

        # PHASE 1: Initialize EntityExtractor for NER (detect-only, no behavioral changes)
        self.entity_extractor = EntityExtractor()

        # Load spaCy model for content noun extraction (multi-theme support)
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("✅ Loaded spaCy model for content noun extraction")
        except OSError:
            logger.warning("⚠️ spaCy model 'en_core_web_sm' not found, content noun extraction disabled")
            self.nlp = None
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured components.
        
        Args:
            query: Natural language query string
            
        Returns:
            ParsedQuery object with extracted filters and keywords
        """
        query_lower = query.lower().strip()
        parsed = ParsedQuery(raw_query=query)

        # PHASE 1: Extract named entities (actors, directors, studios) - detect-only, no behavioral changes
        entities = self.entity_extractor.extract(query)
        parsed.actors = entities.actors
        parsed.directors = entities.directors
        parsed.studios = entities.studios

        # Log detected entities
        if entities.actors or entities.directors or entities.studios:
            logger.info(f"🎭 Detected entities:")
            if entities.actors:
                logger.info(f"  Actors: {entities.actors}")
            if entities.directors:
                logger.info(f"  Directors: {entities.directors}")
            if entities.studios:
                logger.info(f"  Studios: {entities.studios}")

        # Track words that have been classified (to avoid duplicate keywords)
        classified_words = set()
        
        # Extract decades (FIXES THE 80s BUG!)
        parsed.decades = self._extract_decades(query_lower, classified_words)
        
        # Extract years and year ranges
        parsed.year_range = self._extract_year_range(query_lower, classified_words)
        parsed.years = self._extract_years(query_lower, classified_words)
        
        # Extract genres
        parsed.genres = self._extract_from_set(query_lower, self.GENRES, classified_words)
        
        # Extract moods
        parsed.moods = self._extract_from_set(query_lower, self.MOODS, classified_words)
        
        # Extract themes
        parsed.themes = self._extract_from_set(query_lower, self.THEMES, classified_words)

        # Add genres to themes so they can be used for BOTH filtering AND scoring
        # This fixes "2000s romantic comedies" which would otherwise have no scoring terms
        if parsed.genres:
            parsed.themes.extend(parsed.genres)

        # Save original themes BEFORE expansion (for concept-based scoring)
        parsed.original_themes = list(parsed.themes)

        # Expand category-level queries (e.g., "supernatural" -> all supernatural tags)
        # Also populate expanded_categories to track which concepts were expanded
        parsed.themes = self._expand_query_categories(query_lower, parsed.themes, classified_words, parsed)

        # Context-aware mood inference (detect emotional context + request → infer mood themes)
        # E.g., "died" + "recommend" → add uplifting/comforting themes
        parsed.themes = self._infer_mood_from_context(query_lower, parsed.themes)

        # Detect adult context (for filtering family-friendly content)
        parsed.adult_context = self._detect_adult_context(query_lower)

        # Extract remaining keywords (words not classified as filters)
        parsed.keywords = self._extract_keywords(query_lower, classified_words)

        # Save original keywords (for concept-based scoring)
        parsed.original_keywords = list(parsed.keywords)

        # Multi-theme support: Extract content nouns when NO entities detected
        # This enables multi-theme mode for queries like "trauma and recovery through humor"
        has_entities = bool(parsed.actors or parsed.directors or parsed.studios)
        if not has_entities:
            parsed.content_nouns, parsed.theme_groups = self._extract_content_nouns_and_groups(
                query, original_themes=parsed.original_themes
            )

        # Infer additional genres from keywords/themes (e.g., "military" -> also include "war" genre)
        # This allows "90s action films with military themes" to also match War genre movies
        # NOTE: Must be after content_nouns extraction so we can check those too
        inferred_genres = set()
        all_terms = list(parsed.keywords) + list(parsed.themes)
        if hasattr(parsed, 'content_nouns') and parsed.content_nouns:
            all_terms.extend(parsed.content_nouns)
        for term in all_terms:
            term_lower = term.lower()
            if term_lower in self.KEYWORD_TO_GENRE:
                inferred_genre = self.KEYWORD_TO_GENRE[term_lower]
                if inferred_genre not in parsed.genres:
                    inferred_genres.add(inferred_genre)
                    logger.info(f"💡 Inferred genre '{inferred_genre}' from keyword '{term}'")

        # Add inferred genres
        if inferred_genres:
            parsed.genres.extend(list(inferred_genres))
            # Also add to themes for scoring
            parsed.themes.extend(list(inferred_genres))

        if parsed.theme_groups:
            logger.info(f"🎭 Multi-theme mode available: {len(parsed.theme_groups)} theme groups detected")

        # Create cleaned query (original with filter words removed)
        parsed.cleaned_query = self._clean_query(query_lower, classified_words)

        logger.debug(f"Parsed: {parsed}")
        return parsed
    
    def _extract_decades(self, query: str, classified: Set[str]) -> List[int]:
        """
        Extract decade references from query.
        
        Handles: 1980s, 80s, eighties, nineteen eighties, etc.
        FIXES THE BUG: "80s" now correctly maps to 1980
        
        Args:
            query: Lowercase query string
            classified: Set to track classified words
            
        Returns:
            List of decades as base years (e.g., [1980, 1990])
        """
        decades = []
        
        for pattern, ptype in self.decade_regex:
            for match in pattern.finditer(query):
                classified.add(match.group(0))
                
                if ptype == 'numeric_s':
                    # "1980s" or "2000s" -> extract directly
                    century = match.group(1)  # "19" or "20"
                    decade = match.group(2)   # "80", "90", etc.
                    decade_year = int(century + decade)
                    decades.append(decade_year)
                    logger.debug(f"Extracted decade: {decade_year} from '{match.group(0)}'")
                
                elif ptype == 'short_s':
                    # "80s" or "90s" -> assume 1900s if < 50, else 2000s
                    decade_num = int(match.group(1))  # 80, 90, etc.
                    
                    # FIX: Properly handle short decade references
                    if decade_num >= 50:
                        decade_year = 1900 + decade_num  # 80s -> 1980
                    else:
                        decade_year = 2000 + decade_num  # 20s -> 2020
                    
                    decades.append(decade_year)
                    logger.debug(f"Extracted decade: {decade_year} from '{match.group(0)}'")
                
                elif ptype == 'word':
                    # "nineteen eighties" or "twenty tens"
                    century_word = match.group(1)  # "nineteen" or "twenty"
                    decade_word = match.group(2)   # "eighties", "tens"
                    
                    if decade_word in self.DECADE_WORDS:
                        decade_num = self.DECADE_WORDS[decade_word]
                        
                        if century_word == 'nineteen':
                            decade_year = 1900 + decade_num
                        elif century_word == 'twenty':
                            decade_year = 2000 + decade_num
                        else:
                            continue
                        
                        decades.append(decade_year)
                        logger.debug(f"Extracted decade: {decade_year} from '{match.group(0)}'")
        
        return sorted(list(set(decades)))  # Remove duplicates, sort
    
    def _extract_year_range(self, query: str, classified: Set[str]) -> Optional[tuple]:
        """
        Extract year range from query.
        
        Examples: "1980-1989", "2000 - 2010"
        
        Returns:
            Tuple of (start_year, end_year) or None
        """
        match = self.year_range_pattern.search(query)
        if match:
            classified.add(match.group(0))
            start = int(match.group(1))
            end = int(match.group(2))
            logger.debug(f"Extracted year range: {start}-{end}")
            return (start, end)
        return None
    
    def _extract_years(self, query: str, classified: Set[str]) -> List[int]:
        """
        Extract individual years from query.
        
        Examples: "from 2020", "in 1999"
        """
        years = []
        for match in self.year_pattern.finditer(query):
            year_str = match.group(0)
            # Skip if already part of a year range
            if year_str not in classified:
                classified.add(year_str)
                years.append(int(year_str))
                logger.debug(f"Extracted year: {year_str}")
        
        return sorted(list(set(years)))
    
    def _extract_from_set(self, query: str, keyword_set: Set[str], 
                          classified: Set[str]) -> List[str]:
        """
        Extract keywords from a predefined set (genres, moods, themes).
        
        Args:
            query: Lowercase query string
            keyword_set: Set of keywords to search for
            classified: Set to track classified words
            
        Returns:
            List of found keywords
        """
        found = []
        
        for keyword in keyword_set:
            # Use word boundaries for single words, phrase matching for multi-word
            if ' ' in keyword or '-' in keyword:
                # Multi-word phrase (e.g., "kung fu", "time travel", "coming-of-age")
                # Normalize hyphens to spaces for matching
                normalized_keyword = keyword.replace('-', ' ')
                normalized_query = query.replace('-', ' ')
                if normalized_keyword in normalized_query:
                    found.append(keyword)
                    classified.add(keyword)
                    # Also classify individual words (split on both spaces and hyphens)
                    for word in re.split(r'[\s\-]+', keyword):
                        if word:  # Skip empty strings
                            classified.add(word)
            else:
                # Single word with boundaries
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, query):
                    found.append(keyword)
                    classified.add(keyword)
        
        return sorted(found)

    def _expand_query_categories(self, query: str, themes: List[str],
                                   classified: Set[str], parsed: ParsedQuery = None) -> List[str]:
        """
        Expand category-level queries to their component tags.

        For example, if "supernatural" is in the query, expand it to:
        ['ghost', 'vampire', 'zombie', 'paranormal', 'werewolf', etc.]

        Also tracks which categories were expanded in parsed.expanded_categories
        for concept-based scoring.

        IMPORTANT: For entity nouns (girlfriend, boyfriend, etc.), preserve them as keywords
        by NOT adding to classified set (so dual-track can use them for entity track).

        Args:
            query: Lowercase query string
            themes: Already extracted themes list
            classified: Set to track classified words
            parsed: ParsedQuery object to populate expanded_categories

        Returns:
            Expanded list of themes including all expansion tags
        """
        expanded_themes = list(themes)  # Start with existing themes
        expanded_categories = []  # Track which categories were expanded

        # Entity keywords that should be preserved even when expanded
        # These are concrete nouns that can match to specific movies
        ENTITY_NOUNS = {
            'girlfriend', 'boyfriend', 'wife', 'husband', 'mother', 'father',
            'brother', 'sister', 'son', 'daughter', 'friend', 'family'
        }

        # Check each expansion category
        for category, expansion_tags in self.QUERY_EXPANSIONS.items():
            # Check if category keyword appears in the query
            # Use word boundaries for single words, phrase matching for multi-word
            if ' ' in category:
                # Multi-word phrase (e.g., "feel-good", "strong female lead")
                if category in query:
                    logger.info(f"Query expansion: '{category}' -> {len(expansion_tags)} tags")
                    expanded_themes.extend(expansion_tags)
                    expanded_categories.append(category)  # Track this category

                    # Don't classify entity nouns (preserve for keywords)
                    if category.lower() not in ENTITY_NOUNS:
                        classified.add(category)
            else:
                # Single word with boundaries
                pattern = r'\b' + re.escape(category) + r'\b'
                if re.search(pattern, query):
                    logger.info(f"Query expansion: '{category}' -> {len(expansion_tags)} tags")
                    expanded_themes.extend(expansion_tags)
                    expanded_categories.append(category)  # Track this category

                    # Don't classify entity nouns (preserve for keywords)
                    if category.lower() not in ENTITY_NOUNS:
                        classified.add(category)

        # Populate expanded_categories in ParsedQuery for concept-based scoring
        if parsed:
            parsed.expanded_categories = expanded_categories

        # Remove duplicates while preserving order
        seen = set()
        unique_themes = []
        for theme in expanded_themes:
            if theme not in seen:
                seen.add(theme)
                unique_themes.append(theme)

        return unique_themes

    def _detect_adult_context(self, query: str) -> bool:
        """
        Detect if query has adult relationship context (breakup, divorce, etc.)
        This triggers exclusion of family-friendly/children's content.

        Args:
            query: Lowercase query string

        Returns:
            True if adult context detected, False otherwise
        """
        for keyword in self.ADULT_CONTEXT_KEYWORDS:
            # Check for multi-word phrases
            if ' ' in keyword:
                if keyword in query:
                    logger.info(f"Adult context detected: '{keyword}'")
                    return True
            else:
                # Single word with boundaries
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, query):
                    logger.info(f"Adult context detected: '{keyword}'")
                    return True

        return False

    def _infer_mood_from_context(self, query: str, themes: List[str]) -> List[str]:
        """
        Infer mood themes from emotional context in the query.

        Uses sentiment polarity approach:
        - Negative emotion + request → add uplifting/comforting themes
        - Positive emotion + request → add exciting/fun themes

        This allows queries like "my dog died. can you recommend a movie for me."
        to automatically infer uplifting themes without explicitly mentioning them.

        Args:
            query: Lowercase query string
            themes: Current list of themes

        Returns:
            Updated list of themes with inferred mood themes added
        """
        # Check for multi-word emotion/request phrases first
        negative_detected = False
        positive_detected = False
        request_detected = False

        detected_emotions = []
        detected_requests = []

        # Check negative emotions (including multi-word phrases)
        for emotion in self.NEGATIVE_EMOTIONS:
            if ' ' in emotion:
                # Multi-word phrase
                if emotion in query:
                    negative_detected = True
                    detected_emotions.append(emotion)
            else:
                # Single word with boundaries
                pattern = r'\b' + re.escape(emotion) + r'\b'
                if re.search(pattern, query):
                    negative_detected = True
                    detected_emotions.append(emotion)

        # Check positive emotions (including multi-word phrases)
        for emotion in self.POSITIVE_EMOTIONS:
            if ' ' in emotion:
                # Multi-word phrase
                if emotion in query:
                    positive_detected = True
                    detected_emotions.append(emotion)
            else:
                # Single word with boundaries
                pattern = r'\b' + re.escape(emotion) + r'\b'
                if re.search(pattern, query):
                    positive_detected = True
                    detected_emotions.append(emotion)

        # Check request words (including multi-word phrases)
        for request in self.REQUEST_WORDS:
            if ' ' in request:
                # Multi-word phrase
                if request in query:
                    request_detected = True
                    detected_requests.append(request)
            else:
                # Single word with boundaries
                pattern = r'\b' + re.escape(request) + r'\b'
                if re.search(pattern, query):
                    request_detected = True
                    detected_requests.append(request)

        # Infer mood themes based on emotion + request combination
        inferred_themes = list(themes)  # Start with existing themes

        if request_detected:
            if negative_detected:
                # Negative emotion + request → uplifting/comforting themes
                uplifting_themes = ['uplifting', 'comforting', 'feel-good', 'hopeful', 'heartwarming']

                # Only add themes that aren't already present
                for theme in uplifting_themes:
                    if theme not in inferred_themes:
                        inferred_themes.append(theme)

                logger.info(f"💡 Context-aware inference: Detected negative emotion ({', '.join(detected_emotions[:2])}) + request → adding uplifting themes")

            elif positive_detected:
                # Positive emotion + request → exciting/fun themes
                exciting_themes = ['fun', 'exciting', 'entertaining', 'inspiring']

                # Only add themes that aren't already present
                for theme in exciting_themes:
                    if theme not in inferred_themes:
                        inferred_themes.append(theme)

                logger.info(f"💡 Context-aware inference: Detected positive emotion ({', '.join(detected_emotions[:2])}) + request → adding exciting themes")

        return inferred_themes

    def _extract_keywords(self, query: str, classified: Set[str]) -> List[str]:
        """
        Extract remaining meaningful keywords not classified as filters.
        
        Args:
            query: Lowercase query string
            classified: Set of already classified words/phrases
            
        Returns:
            List of keyword strings
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', query)
        
        # Stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'is', 'was', 'are', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
            'movie', 'movies', 'film', 'films', 'show', 'shows', 'like', 'similar',
            # Meta-words that should not be keywords
            'themes', 'theme', 'genre', 'genres', 'type', 'types', 'style', 'styles'
        }
        
        keywords = []
        for word in words:
            # Skip if: too short, stop word, or already classified
            if len(word) < 3 or word in stop_words or word in classified:
                continue
            
            # Check if part of classified phrase
            is_part_of_phrase = any(word in phrase for phrase in classified if ' ' in phrase)
            if is_part_of_phrase:
                continue
            
            keywords.append(word)

        return keywords

    def _extract_content_nouns_and_groups(self, query: str, original_themes: List[str] = None) -> tuple:
        """
        Extract content nouns from query using spaCy POS tagging and group them
        based on dependency parsing.

        This is used for MULTI-THEME MODE when no entities (actors/directors/studios)
        are detected. It identifies the main thematic concepts in the query.

        Args:
            query: Original query string
            original_themes: Already-extracted themes from THEMES set (for compound terms)

        Returns:
            Tuple of (content_nouns: List[str], theme_groups: List[List[str]])

        Example:
            "Movies about trauma and recovery through humor"
            -> content_nouns: ['trauma', 'recovery', 'humor']
            -> theme_groups: [['trauma', 'recovery'], ['humor']]
        """
        if self.nlp is None:
            return [], []

        doc = self.nlp(query)

        # Start with original_themes (captures compound terms like 'coming-of-age', 'lgbtq')
        # These are already correctly extracted by _extract_from_set()
        # Deduplicate equivalent themes (e.g., 'coming of age' and 'coming-of-age' are the same)
        content_nouns = []
        seen_normalized = set()  # Track normalized forms to avoid duplicates
        if original_themes:
            for theme in original_themes:
                theme_lower = theme.lower()
                # Normalize: replace hyphens with spaces for comparison
                normalized = theme_lower.replace('-', ' ')
                if normalized not in seen_normalized:
                    seen_normalized.add(normalized)
                    content_nouns.append(theme_lower)

        # Words to exclude (generic movie request words, not thematic)
        EXCLUDE_WORDS = {
            'movie', 'movies', 'film', 'films', 'show', 'shows',
            'something', 'thing', 'things', 'one', 'ones',
            'recommendation', 'recommendations', 'suggestion', 'suggestions',
            'theme', 'themes', 'vibe', 'vibes', 'element', 'elements',
            'story', 'stories', 'style', 'styles', 'tone', 'tones'
        }

        # Track which words are already covered by original_themes (including parts of compounds)
        covered_words = set()
        for theme in content_nouns:  # content_nouns already has original_themes
            # Add the theme itself and its parts (for compounds like 'coming-of-age')
            covered_words.add(theme)
            for part in theme.replace('-', ' ').split():
                covered_words.add(part.lower())

        # Extract content nouns (non-stop NOUNS, excluding generic words)
        # Also extract thematic ADJECTIVES that modify generic nouns like "themes", "movies", "films"
        noun_tokens = []  # Keep track of tokens for dependency analysis

        # First pass: collect nouns (skip those already covered by original_themes)
        for token in doc:
            token_lower = token.text.lower()
            # Skip if already covered, is a decade (ends in 0s), or is too short
            is_decade = token_lower.endswith('0s') and len(token_lower) <= 5
            if (token.pos_ == 'NOUN' and
                not token.is_stop and
                token_lower not in EXCLUDE_WORDS and
                token_lower not in covered_words and
                not is_decade and
                len(token.text) > 2):
                content_nouns.append(token_lower)
                noun_tokens.append(token)

        # Second pass: collect thematic adjectives (amod) that modify generic nouns
        # e.g., "military themes" -> "military" is the thematic content, not "themes"
        GENERIC_NOUNS = {'themes', 'theme', 'movies', 'movie', 'films', 'film', 'stories', 'story', 'vibes', 'vibe', 'elements', 'element'}
        for token in doc:
            if (token.pos_ == 'ADJ' and
                token.dep_ == 'amod' and  # Adjectival modifier
                token.head.text.lower() in GENERIC_NOUNS and
                not token.is_stop and
                len(token.text) > 2):
                adj_text = token.text.lower()
                if adj_text not in content_nouns and adj_text not in covered_words:  # Avoid duplicates
                    content_nouns.append(adj_text)
                    noun_tokens.append(token)
                    logger.debug(f"🎯 Extracted thematic adjective: '{adj_text}' (modifies '{token.head.text}')")

        if len(content_nouns) < 2:
            # Not enough nouns for multi-theme mode
            return content_nouns, []

        # Group nouns based on dependency structure
        # Nouns linked by conjunction (cc/conj) go together
        # Nouns in separate prepositional phrases are separate groups
        theme_groups = []
        used_nouns = set()

        for token in noun_tokens:
            if token.text.lower() in used_nouns:
                continue

            # Start a new group with this noun
            group = [token.text.lower()]
            used_nouns.add(token.text.lower())

            # Find conjuncts (nouns linked by "and", "or", etc.)
            # Check children for conjuncts
            for child in token.children:
                if child.dep_ == 'conj' and child.pos_ == 'NOUN':
                    if child.text.lower() not in used_nouns:
                        group.append(child.text.lower())
                        used_nouns.add(child.text.lower())

            # Check if this token is a conjunct of another noun
            if token.dep_ == 'conj' and token.head.pos_ == 'NOUN':
                head_noun = token.head.text.lower()
                # Find the group containing the head noun and add to it
                found_group = False
                for existing_group in theme_groups:
                    if head_noun in existing_group:
                        if token.text.lower() not in existing_group:
                            existing_group.append(token.text.lower())
                        found_group = True
                        break
                if found_group:
                    continue  # Don't create new group, already added to existing

            if group:
                theme_groups.append(group)

        # If we have ungrouped nouns, add them as separate single-noun groups
        for noun in content_nouns:
            if noun not in used_nouns:
                theme_groups.append([noun])
                used_nouns.add(noun)

        # Merge any single-noun groups if they should be together
        # (e.g., if we missed a conjunction relationship)
        # For now, keep groups as detected

        # Sort groups by position in original query (first mentioned = first group)
        def group_position(group):
            for i, noun in enumerate(content_nouns):
                if noun in group:
                    return i
            return 999

        theme_groups.sort(key=group_position)

        logger.info(f"🎯 Multi-theme extraction: nouns={content_nouns}, groups={theme_groups}")

        return content_nouns, theme_groups

    def _clean_query(self, query: str, classified: Set[str]) -> str:
        """
        Create cleaned query with filter words removed.
        
        Useful for semantic search on remaining content.
        """
        # Remove classified words/phrases
        cleaned = query
        
        # Remove phrases first (longest to shortest to avoid partial matches)
        phrases = sorted([p for p in classified if ' ' in p], 
                        key=len, reverse=True)
        for phrase in phrases:
            cleaned = cleaned.replace(phrase, ' ')
        
        # Remove individual words
        words = [w for w in classified if ' ' not in w]
        for word in words:
            pattern = r'\b' + re.escape(word) + r'\b'
            cleaned = re.sub(pattern, ' ', cleaned)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned


# Convenience function
def parse_query(query: str) -> ParsedQuery:
    """
    Convenience function to parse a query.
    
    Args:
        query: Natural language query string
        
    Returns:
        ParsedQuery object
        
    Example:
        >>> parsed = parse_query("kung fu movies from the 80s")
        >>> print(parsed.decades)  # [1980]
        >>> print(parsed.keywords)  # ["kung", "fu"]
    """
    parser = QueryParser()
    return parser.parse(query)


# Testing code
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING QUERY PARSER")
    print("="*60 + "\n")
    
    parser = QueryParser()
    
    # Test queries covering various patterns
    test_queries = [
        # Decade tests (including the 80s bug fix!)
        "kung fu movies from the 80s",
        "action movies from the 1980s",
        "comedy from the nineties",
        "films from the nineteen eighties",
        
        # Year tests
        "thriller from 2020",
        "movies from 1999-2005",
        "film from 2010",
        
        # Genre + mood tests
        "dark psychological thriller",
        "funny romantic comedy",
        "intense action movie",
        
        # Theme tests
        "heist movie",
        "gangster films",
        "martial arts action",
        "zombie horror",
        
        # Complex queries
        "dark gangster movie from the 90s",
        "funny heist film from 2001",
        "intense martial arts thriller from the 1980s",
        
        # Edge cases
        "sports drama",
        "science fiction",
        "coming of age story",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: '{query}'")
        print("-" * 60)
        parsed = parser.parse(query)
        print(parsed)
        print()
    
    print("="*60)
    print("ALL TESTS COMPLETE! ✅")
    print("="*60)
    
    # Specific 80s bug test
    print("\n🔬 SPECIFIC TEST: 80s DECADE BUG FIX")
    print("-" * 60)
    
    test_cases = [
        ("80s", 1980),
        ("90s", 1990),
        ("1980s", 1980),
        ("2000s", 2000),
        ("20s", 2020),
        ("30s", 2030),
    ]
    
    print("Testing decade extraction:")
    for query, expected in test_cases:
        parsed = parser.parse(query)
        actual = parsed.decades[0] if parsed.decades else None
        status = "✅" if actual == expected else "❌"
        print(f"  {status} '{query}' -> {actual} (expected {expected})")
    
    print("\n💡 Usage in other modules:")
    print("  from src.query_parser import parse_query")
    print("  parsed = parse_query('kung fu movies from the 80s')")
    print("  print(parsed.decades)  # [1980]")
    print("  print(parsed.keywords)  # ['kung', 'fu']")