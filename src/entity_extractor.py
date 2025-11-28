"""
Entity Extractor Module
Extracts named entities (actors, directors, studios) from movie queries using spaCy NER.
"""

import re
import spacy
import pandas as pd
import numpy as np
from typing import List, Set, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Named entities extracted from a query."""
    actors: List[str]  # Person names (likely actors)
    directors: List[str]  # Person names with "directed by" context
    studios: List[str]  # Organization names (likely studios)
    all_person_names: List[str]  # All person names detected
    all_org_names: List[str]  # All organization names detected


class EntityExtractor:
    """Extract named entities from movie queries using spaCy NER."""

    # Director indicators (multi-word phrases that indicate directors)
    DIRECTOR_PHRASES = {
        'directed by', 'made by', 'from director', 'by director'
    }

    # Single-word director indicators (only when adjacent to name)
    DIRECTOR_WORDS = {
        'director'
    }

    # Known studios (for disambiguation)
    KNOWN_STUDIOS = {
        'warner bros', 'paramount', 'universal', 'disney', 'pixar', 'dreamworks',
        'columbia', 'fox', '20th century', 'mgm', 'sony', 'netflix', 'hbo', 'a24',
        'miramax', 'lionsgate', 'marvel', 'dc', 'lucasfilm', 'new line', 'wb'
    }

    # Known directors (for better classification)
    KNOWN_DIRECTORS = {
        'quentin tarantino', 'christopher nolan', 'martin scorsese', 'steven spielberg',
        'ridley scott', 'james cameron', 'peter jackson', 'david fincher', 'denis villeneuve',
        'wes anderson', 'paul thomas anderson', 'coen brothers', 'guillermo del toro'
    }

    def __init__(self, tmdb_data_path: Optional[str] = "data/raw/tmdb_fully_enriched.parquet"):
        """
        Initialize the entity extractor with spaCy model.

        Args:
            tmdb_data_path: Path to TMDB data for actor fallback (optional)
        """
        try:
            # Load small English model (fast, good enough for person/org detection)
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
            logger.info("✅ Loaded spaCy NER model (en_core_web_sm)")
        except OSError:
            logger.error("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # TMDB fallback data (lazy-loaded on first use)
        self.tmdb_data_path = tmdb_data_path
        self._tmdb_actors = None  # Cache of all actor names from TMDB

    def extract(self, query: str) -> ExtractedEntities:
        """
        Extract named entities from a movie query.

        Args:
            query: Natural language movie query

        Returns:
            ExtractedEntities object with detected actors, directors, studios

        Example:
            >>> extractor = EntityExtractor()
            >>> entities = extractor.extract("Bill Murray movies from the 90s")
            >>> print(entities.actors)  # ['Bill Murray']
        """
        if not self.nlp:
            # Fallback if spaCy not available
            return ExtractedEntities(
                actors=[], directors=[], studios=[],
                all_person_names=[], all_org_names=[]
            )

        # Store query for studio detection
        self._query_text = query

        # Run NER
        doc = self.nlp(query)

        # Words to exclude from person names (common movie-related words and genres)
        EXCLUDED_WORDS = {
            'Movies', 'Films', 'Pictures', 'Productions', 'Studios', 'Entertainment',
            # Common genres that shouldn't be part of actor names
            'Drama', 'Dramas', 'Comedy', 'Comedies', 'Thriller', 'Thrillers',
            'Action', 'Horror', 'Romance', 'Romances', 'Romantic', 'SciFi', 'Western', 'Westerns',
            'Documentary', 'Documentaries', 'Animation', 'Animated', 'Fantasy',
            'Adventure', 'Adventures', 'Musical', 'Musicals', 'Mystery', 'Mysteries',
            'Crime', 'War', 'Historical', 'Biography', 'Biographies'
        }

        # Extract all person and org entities
        person_names = []
        org_names = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Clean up the name
                name = ent.text.strip()

                # Remove excluded words from the end of the name (case-insensitive)
                # E.g., "Bill Murray Movies" -> "Bill Murray"
                words = name.split()
                excluded_words_lower = {w.lower() for w in EXCLUDED_WORDS}
                while words and words[-1].lower() in excluded_words_lower:
                    words.pop()

                name = ' '.join(words)

                # Skip single-word names that might be generic (e.g., "Bill", "Murray")
                # unless they have clear context
                if name and len(name.split()) >= 2:  # Full names only
                    person_names.append(name)
                elif name and len(name.split()) == 1:
                    # Single names like "Bill" - check if followed by capitalized word
                    # This handles cases like "Bill Murray" where spaCy splits them
                    pass  # Skip for now, handle in cleanup

            elif ent.label_ == "ORG":
                name = ent.text.strip()
                if name:
                    org_names.append(name)

        # Additional cleanup: Merge adjacent single-word person names
        person_names = self._merge_adjacent_names(query, person_names)

        # Classify person names as actors or directors based on context
        directors = self._extract_directors(query, person_names)
        actors = [name for name in person_names if name not in directors]

        # Filter org names for likely studios
        studios = self._filter_studios(org_names)

        # TMDB FALLBACK: If NER didn't find any actors, try TMDB actor matching
        if not actors:
            logger.info(f"NER found no actors, trying TMDB fallback for query: '{query}'")
            tmdb_actors = self._tmdb_actor_fallback(query)
            if tmdb_actors:
                actors = tmdb_actors
                person_names.extend(tmdb_actors)  # Add to all_person_names too
                logger.info(f"✅ TMDB fallback found actors: {actors}")
            else:
                logger.info(f"⚠️ TMDB fallback found no actors either")

        logger.debug(f"NER extracted: {len(actors)} actors, {len(directors)} directors, {len(studios)} studios")
        logger.debug(f"  Actors: {actors}")
        logger.debug(f"  Directors: {directors}")
        logger.debug(f"  Studios: {studios}")

        return ExtractedEntities(
            actors=actors,
            directors=directors,
            studios=studios,
            all_person_names=person_names,
            all_org_names=org_names
        )

    def _merge_adjacent_names(self, query: str, person_names: List[str]) -> List[str]:
        """
        Merge adjacent capitalized words that might be full names.

        E.g., "Bill" + "Murray" -> "Bill Murray"
        """
        # Words to exclude from person names (common movie-related words and genres)
        EXCLUDED_WORDS = {
            'Movies', 'Films', 'Pictures', 'Productions', 'Studios', 'Entertainment',
            # Common genres that shouldn't be part of actor names
            'Drama', 'Dramas', 'Comedy', 'Comedies', 'Thriller', 'Thrillers',
            'Action', 'Horror', 'Romance', 'Romances', 'Romantic', 'SciFi', 'Western', 'Westerns',
            'Documentary', 'Documentaries', 'Animation', 'Animated', 'Fantasy',
            'Adventure', 'Adventures', 'Musical', 'Musicals', 'Mystery', 'Mysteries',
            'Crime', 'War', 'Historical', 'Biography', 'Biographies'
        }

        # Find all sequences of 2-3 capitalized words
        # Pattern: one or more uppercase letter followed by lowercase letters
        pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b'
        matches = re.findall(pattern, query)

        # Merge with existing person_names
        merged = list(person_names)
        for match in matches:
            if match not in merged:
                # Exclude matches that contain movie-related words (case-insensitive check)
                words = match.split()
                words_lower = [w.lower() for w in words]
                excluded_words_lower = {w.lower() for w in EXCLUDED_WORDS}
                if any(word in excluded_words_lower for word in words_lower):
                    continue  # Skip this match

                # Check if it's a known entity type (not a place, etc.)
                # For simplicity, add if it looks like a person name (2-3 words)
                word_count = len(words)
                if 2 <= word_count <= 3:
                    merged.append(match)

        return sorted(set(merged))  # Remove duplicates

    def _extract_directors(self, query: str, person_names: List[str]) -> List[str]:
        """
        Extract directors based on context indicators.

        E.g., "directed by Quentin Tarantino", "Tarantino movies"
        """
        query_lower = query.lower()
        directors = []

        for name in person_names:
            name_lower = name.lower()

            # Check if it's a known director
            is_known_director = name_lower in self.KNOWN_DIRECTORS

            # Check if name appears near director indicators
            has_director_context = False

            # First check multi-word director phrases (can appear anywhere near the name)
            for phrase in self.DIRECTOR_PHRASES:
                # Pattern: "directed by NAME" or "NAME directed by"
                if re.search(rf'\b{re.escape(phrase)}\b.*\b{re.escape(name_lower)}\b', query_lower):
                    has_director_context = True
                    break
                elif re.search(rf'\b{re.escape(name_lower)}\b.*\b{re.escape(phrase)}\b', query_lower):
                    has_director_context = True
                    break

            # If no phrase match, check single-word indicators (must be adjacent)
            if not has_director_context:
                for word in self.DIRECTOR_WORDS:
                    # Pattern: "NAME director" or "director NAME" (adjacent words only)
                    # Use \s+ to ensure adjacency
                    if re.search(rf'\b{re.escape(name_lower)}\s+{re.escape(word)}\b', query_lower):
                        has_director_context = True
                        break
                    elif re.search(rf'\b{re.escape(word)}\s+{re.escape(name_lower)}\b', query_lower):
                        has_director_context = True
                        break

            # Only mark as director if there's explicit context OR they're a known director
            if has_director_context or is_known_director:
                directors.append(name)

        return directors

    def _filter_studios(self, org_names: List[str]) -> List[str]:
        """
        Filter organization names for likely movie studios.

        Also add standalone known studio names from the query even if NER missed them.
        """
        studios = []

        # Check NER-detected org names
        for org in org_names:
            org_lower = org.lower()

            # Check if it matches known studios
            is_studio = any(studio in org_lower for studio in self.KNOWN_STUDIOS)

            # Or if it contains studio-like keywords
            studio_keywords = ['studios', 'pictures', 'films', 'entertainment', 'productions']
            is_studio = is_studio or any(kw in org_lower for kw in studio_keywords)

            if is_studio:
                studios.append(org)

        # Also check for known studios that NER might have missed (Pixar, Marvel, etc.)
        if hasattr(self, '_query_text'):
            query_lower = self._query_text.lower()
            for studio in self.KNOWN_STUDIOS:
                if studio in query_lower and studio.capitalize() not in studios:
                    studios.append(studio.title())  # Add with title case

        return list(set(studios))  # Remove duplicates

    def _load_tmdb_actors(self) -> Set[str]:
        """
        Load all actor names from TMDB data (lazy-loaded and cached).

        Returns:
            Set of all actor names from TMDB cast data
        """
        if self._tmdb_actors is not None:
            return self._tmdb_actors

        try:
            if not self.tmdb_data_path or not Path(self.tmdb_data_path).exists():
                logger.warning(f"TMDB data not found at {self.tmdb_data_path}, actor fallback disabled")
                self._tmdb_actors = set()
                return self._tmdb_actors

            logger.info(f"Loading TMDB actor data from {self.tmdb_data_path}...")
            df = pd.read_parquet(self.tmdb_data_path, columns=['cast'], engine='pyarrow')

            # Extract all unique actor names from cast lists
            all_actors = set()
            for cast_list in df['cast']:
                # Handle both lists and numpy arrays
                if isinstance(cast_list, (list, np.ndarray)):
                    for actor in cast_list:
                        if isinstance(actor, str) and actor.strip():
                            all_actors.add(actor.strip())

            self._tmdb_actors = all_actors
            logger.info(f"✅ Loaded {len(all_actors)} unique actors from TMDB data")
            return self._tmdb_actors

        except Exception as e:
            logger.warning(f"Could not load TMDB actors: {e}, actor fallback disabled")
            self._tmdb_actors = set()
            return self._tmdb_actors

    def _tmdb_actor_fallback(self, query: str) -> List[str]:
        """
        Fallback method to detect actors when spaCy NER fails.
        Uses pattern matching against known TMDB actor names.

        Args:
            query: The query string

        Returns:
            List of detected actor names from TMDB data
        """
        # Load TMDB actors (cached after first call)
        tmdb_actors = self._load_tmdb_actors()
        if not tmdb_actors:
            return []

        # Words to exclude from person names (same as in _merge_adjacent_names)
        EXCLUDED_WORDS = {
            'Movies', 'Films', 'Pictures', 'Productions', 'Studios', 'Entertainment',
            'Drama', 'Dramas', 'Comedy', 'Comedies', 'Thriller', 'Thrillers',
            'Action', 'Horror', 'Romance', 'Romances', 'Romantic', 'SciFi', 'Western', 'Westerns',
            'Documentary', 'Documentaries', 'Animation', 'Animated', 'Fantasy',
            'Adventure', 'Adventures', 'Musical', 'Musicals', 'Mystery', 'Mysteries',
            'Crime', 'War', 'Historical', 'Biography', 'Biographies'
        }

        # Find sequences of 2-3 capitalized words that could be names
        # Pattern: "Firstname Lastname" or "Firstname Middle Lastname"
        pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b'
        candidate_names = re.findall(pattern, query)
        logger.info(f"  TMDB fallback found {len(candidate_names)} candidate names: {candidate_names}")

        detected_actors = []
        for name in candidate_names:
            # Remove excluded words from the END of the name
            # E.g., "Julia Roberts Romances" -> "Julia Roberts"
            words = name.split()
            words_lower = [w.lower() for w in words]
            excluded_words_lower = {w.lower() for w in EXCLUDED_WORDS}

            # Strip excluded words from the end
            while words and words[-1].lower() in excluded_words_lower:
                words.pop()

            # If nothing left after stripping, skip
            if not words:
                continue

            cleaned_name = ' '.join(words)

            # Check if this cleaned name exists in TMDB actor database
            if cleaned_name in tmdb_actors:
                detected_actors.append(cleaned_name)
                logger.info(f"  ✅ TMDB fallback matched: '{cleaned_name}' (from '{name}')")

        return detected_actors


# Convenience function
def extract_entities(query: str) -> ExtractedEntities:
    """
    Convenience function to extract entities from a query.

    Args:
        query: Natural language movie query

    Returns:
        ExtractedEntities object

    Example:
        >>> entities = extract_entities("Bill Murray movies from the 90s")
        >>> print(entities.actors)  # ['Bill Murray']
    """
    extractor = EntityExtractor()
    return extractor.extract(query)


# Testing code
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING ENTITY EXTRACTOR")
    print("="*80 + "\n")

    extractor = EntityExtractor()

    test_queries = [
        "Bill Murray movies from the 90s",
        "movies directed by Quentin Tarantino",
        "Tom Hanks and Meg Ryan romantic comedies",
        "Pixar animated films",
        "Christopher Nolan thrillers",
        "Meryl Streep dramas",
        "Warner Bros action movies",
        "Marvel superhero films",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: '{query}'")
        print("-" * 80)
        entities = extractor.extract(query)
        print(f"  Actors: {entities.actors}")
        print(f"  Directors: {entities.directors}")
        print(f"  Studios: {entities.studios}")
        print()

    print("="*80)
    print("ALL TESTS COMPLETE! ✅")
    print("="*80)
