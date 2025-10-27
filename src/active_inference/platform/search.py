"""
Search Platform Service

Provides intelligent search capabilities for the Active Inference Knowledge Environment.
Includes query processing, indexing, ranking, and result presentation.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class SearchResult:
    """Search result representation"""

    def __init__(self, node_id: str, title: str, content_type: str, score: float,
                 highlights: List[str] = None, metadata: Dict[str, Any] = None):
        self.node_id = node_id
        self.title = title
        self.content_type = content_type
        self.score = score
        self.highlights = highlights or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'node_id': self.node_id,
            'title': self.title,
            'content_type': self.content_type,
            'score': self.score,
            'highlights': self.highlights,
            'metadata': self.metadata
        }


class QueryProcessor:
    """Processes and analyzes search queries"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize query processor"""
        self.config = config
        self.stop_words = config.get('stop_words', {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'})

    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess search query"""
        # Convert to lowercase
        query = query.lower()

        # Remove punctuation and split
        query = re.sub(r'[^\w\s]', ' ', query)
        tokens = query.split()

        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)

        return unique_tokens

    def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract search intent from query"""
        tokens = self.preprocess_query(query)

        intent = {
            'primary_intent': 'general_search',
            'query_terms': tokens,
            'filters': {},
            'sort_by': 'relevance'
        }

        # Check for specific intents
        if any(term in tokens for term in ['tutorial', 'learn', 'guide']):
            intent['primary_intent'] = 'learning'
        elif any(term in tokens for term in ['research', 'paper', 'study']):
            intent['primary_intent'] = 'research'
        elif any(term in tokens for term in ['implement', 'code', 'example']):
            intent['primary_intent'] = 'implementation'
        elif any(term in tokens for term in ['application', 'use', 'case']):
            intent['primary_intent'] = 'application'

        # Extract potential filters
        content_types = ['foundation', 'mathematics', 'implementation', 'application']
        for content_type in content_types:
            if content_type in tokens:
                intent['filters']['content_type'] = content_type
                break

        difficulty_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        for difficulty in difficulty_levels:
            if difficulty in tokens:
                intent['filters']['difficulty'] = difficulty
                break

        return intent

    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expansions = {
            'active inference': ['active inference', 'free energy principle', 'predictive processing'],
            'entropy': ['entropy', 'information', 'uncertainty'],
            'bayesian': ['bayesian', 'probabilistic', 'statistical'],
            'neural': ['neural', 'brain', 'cognitive']
        }

        expanded_terms = []
        query_lower = query.lower()

        for key, synonyms in expansions.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)

        return list(set(expanded_terms)) if expanded_terms else self.preprocess_query(query)


class IndexManager:
    """Manages search index operations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize index manager"""
        self.config = config
        self.index: Dict[str, Dict[str, Any]] = {}
        self.reverse_index: Dict[str, Set[str]] = {}  # term -> set of node_ids

    def add_to_index(self, node_id: str, content: Dict[str, Any]) -> bool:
        """Add content to search index"""
        try:
            # Store full content
            self.index[node_id] = content

            # Update reverse index
            searchable_text = self._extract_searchable_text(content)

            # Tokenize and add to reverse index
            tokens = self._tokenize_text(searchable_text)
            for token in tokens:
                if token not in self.reverse_index:
                    self.reverse_index[token] = set()
                self.reverse_index[token].add(node_id)

            logger.debug(f"Indexed content for node: {node_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing content for {node_id}: {e}")
            return False

    def remove_from_index(self, node_id: str) -> bool:
        """Remove content from search index"""
        try:
            if node_id in self.index:
                # Remove from reverse index
                if node_id in self.index:
                    searchable_text = self._extract_searchable_text(self.index[node_id])
                    tokens = self._tokenize_text(searchable_text)

                    for token in tokens:
                        if token in self.reverse_index:
                            self.reverse_index[token].discard(node_id)
                            if not self.reverse_index[token]:
                                del self.reverse_index[token]

                del self.index[node_id]
                logger.debug(f"Removed content from index: {node_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error removing content from index for {node_id}: {e}")
            return False

    def search_index(self, query_terms: List[str]) -> Set[str]:
        """Search index for matching node IDs"""
        if not query_terms:
            return set()

        # Find intersection of all query terms
        matching_nodes = None

        for term in query_terms:
            term_nodes = self.reverse_index.get(term.lower(), set())

            if matching_nodes is None:
                matching_nodes = term_nodes.copy()
            else:
                matching_nodes &= term_nodes

            # Early exit if no matches
            if not matching_nodes:
                break

        return matching_nodes or set()

    def _extract_searchable_text(self, content: Dict[str, Any]) -> str:
        """Extract searchable text from content"""
        searchable_parts = []

        # Add title and description
        if 'title' in content:
            searchable_parts.append(content['title'])
        if 'description' in content:
            searchable_parts.append(content['description'])

        # Add content sections
        if 'content' in content:
            content_section = content['content']
            if isinstance(content_section, dict):
                for key, value in content_section.items():
                    if isinstance(value, str):
                        searchable_parts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                searchable_parts.append(item)

        # Add tags
        if 'tags' in content:
            tags = content['tags']
            if isinstance(tags, list):
                searchable_parts.extend(tags)

        return ' '.join(searchable_parts)

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for indexing"""
        # Simple tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        # Remove short tokens and duplicates
        tokens = [token for token in tokens if len(token) > 2]
        return list(set(tokens))

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_documents': len(self.index),
            'total_terms': len(self.reverse_index),
            'average_terms_per_doc': len(self.reverse_index) / len(self.index) if self.index else 0
        }


class ResultRanker:
    """Ranks and sorts search results"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize result ranker"""
        self.config = config

    def rank_results(self, results: List[Dict[str, Any]], query_terms: List[str],
                    ranking_method: str = 'relevance') -> List[Dict[str, Any]]:
        """Rank search results"""
        if ranking_method == 'relevance':
            return self._rank_by_relevance(results, query_terms)
        elif ranking_method == 'recency':
            return self._rank_by_recency(results)
        elif ranking_method == 'popularity':
            return self._rank_by_popularity(results)
        else:
            return results

    def _rank_by_relevance(self, results: List[Dict[str, Any]], query_terms: List[str]) -> List[Dict[str, Any]]:
        """Rank by relevance to query"""
        for result in results:
            score = 0.0

            # Title match bonus
            title = result.get('title', '').lower()
            for term in query_terms:
                if term.lower() in title:
                    score += 10.0

            # Content match
            content = result.get('content', '').lower()
            for term in query_terms:
                if term.lower() in content:
                    score += 1.0

            # Tag match bonus
            tags = result.get('tags', [])
            for tag in tags:
                for term in query_terms:
                    if term.lower() in tag.lower():
                        score += 5.0

            result['relevance_score'] = score

        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)

    def _rank_by_recency(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank by recency"""
        # Sort by last_updated if available
        return sorted(results, key=lambda x: x.get('last_updated', ''), reverse=True)

    def _rank_by_popularity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank by popularity metrics"""
        # Sort by some popularity metric (placeholder)
        return sorted(results, key=lambda x: x.get('usage_count', 0), reverse=True)


class SearchEngine:
    """Main search engine coordinating all search components"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize search engine"""
        self.config = config

        # Initialize components
        self.query_processor = QueryProcessor(config.get('query_processor', {}))
        self.index_manager = IndexManager(config.get('index_manager', {}))
        self.result_ranker = ResultRanker(config.get('result_ranker', {}))

        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'average_response_time': 0.0,
            'popular_queries': {}
        }

    def search(self, query: str, filters: Dict[str, Any] = None,
               limit: int = 10, ranking: str = 'relevance') -> List[Dict[str, Any]]:
        """Perform search"""
        start_time = time.time()

        try:
            # Process query
            query_terms = self.query_processor.preprocess_query(query)

            if not query_terms:
                return []

            # Search index
            matching_node_ids = self.index_manager.search_index(query_terms)

            # Convert to results
            results = []
            for node_id in matching_node_ids:
                if node_id in self.index_manager.index:
                    content = self.index_manager.index[node_id]
                    result = {
                        'node_id': node_id,
                        'title': content.get('title', ''),
                        'content_type': content.get('content_type', ''),
                        'difficulty': content.get('difficulty', ''),
                        'description': content.get('description', ''),
                        'tags': content.get('tags', []),
                        'content': content.get('content', {}),
                        'last_updated': content.get('last_updated', '')
                    }
                    results.append(result)

            # Apply filters
            if filters:
                results = self._apply_filters(results, filters)

            # Rank results
            results = self.result_ranker.rank_results(results, query_terms, ranking)

            # Limit results
            results = results[:limit]

            # Update statistics
            self._update_search_stats(query, time.time() - start_time, len(results))

            return results

        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return []

    def index_content(self, content: Dict[str, Any]) -> bool:
        """Index new content"""
        node_id = content.get('id', '')
        if not node_id:
            logger.error("Cannot index content without ID")
            return False

        return self.index_manager.add_to_index(node_id, content)

    def remove_content(self, node_id: str) -> bool:
        """Remove content from index"""
        return self.index_manager.remove_from_index(node_id)

    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions for partial query"""
        if len(partial_query) < 2:
            return []

        suggestions = []
        partial_lower = partial_query.lower()

        # Look for terms in reverse index that start with partial query
        for term in self.index_manager.reverse_index.keys():
            if term.startswith(partial_lower) and term not in suggestions:
                suggestions.append(term)
                if len(suggestions) >= limit:
                    break

        return suggestions

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        stats = self.search_stats.copy()
        stats['index_stats'] = self.index_manager.get_index_statistics()
        return stats

    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to search results"""
        filtered_results = []

        for result in results:
            include = True

            # Content type filter
            if 'content_type' in filters:
                if result.get('content_type') != filters['content_type']:
                    include = False

            # Difficulty filter
            if 'difficulty' in filters:
                if result.get('difficulty') != filters['difficulty']:
                    include = False

            # Tag filter
            if 'tags' in filters:
                result_tags = set(result.get('tags', []))
                filter_tags = set(filters['tags'])
                if not filter_tags.issubset(result_tags):
                    include = False

            if include:
                filtered_results.append(result)

        return filtered_results

    def _update_search_stats(self, query: str, response_time: float, result_count: int):
        """Update search statistics"""
        self.search_stats['total_searches'] += 1

        # Update average response time
        current_avg = self.search_stats['average_response_time']
        total_searches = self.search_stats['total_searches']
        self.search_stats['average_response_time'] = (
            (current_avg * (total_searches - 1)) + response_time
        ) / total_searches

        # Update popular queries
        if query not in self.search_stats['popular_queries']:
            self.search_stats['popular_queries'][query] = 0
        self.search_stats['popular_queries'][query] += 1