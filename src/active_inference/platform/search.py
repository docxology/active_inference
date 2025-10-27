"""
Platform - Search Engine

Intelligent search capabilities for the Active Inference Knowledge Environment.
Provides semantic search, filtering, ranking, and indexing of knowledge content
with natural language query processing and relevance scoring.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with relevance score"""
    node_id: str
    title: str
    content_type: str
    relevance_score: float
    snippet: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QueryProcessor:
    """Processes and analyzes search queries"""

    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        logger.info("QueryProcessor initialized")

    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess search query"""
        # Simple preprocessing: lowercase, split, remove stop words
        terms = query.lower().split()
        terms = [term for term in terms if term not in self.stop_words and len(term) > 1]

        return terms

    def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract search intent from query"""
        # Simple intent extraction based on keywords
        intent = {
            "primary_intent": "search",
            "filters": {},
            "preferences": {}
        }

        query_lower = query.lower()

        # Detect content type preferences
        if "foundation" in query_lower:
            intent["filters"]["content_type"] = "foundation"
        if "mathematics" in query_lower or "math" in query_lower:
            intent["filters"]["content_type"] = "mathematics"
        if "implementation" in query_lower or "code" in query_lower:
            intent["filters"]["content_type"] = "implementation"
        if "application" in query_lower:
            intent["filters"]["content_type"] = "application"

        # Detect difficulty preferences
        if "beginner" in query_lower or "basic" in query_lower:
            intent["filters"]["difficulty"] = "beginner"
        if "advanced" in query_lower or "expert" in query_lower:
            intent["filters"]["difficulty"] = "advanced"

        return intent


class IndexManager:
    """Manages search indices for efficient querying"""

    def __init__(self):
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)  # term -> node_ids
        self.node_index: Dict[str, Dict[str, Any]] = {}  # node_id -> node_data

        logger.info("IndexManager initialized")

    def add_to_index(self, node_id: str, content: Dict[str, Any]) -> None:
        """Add or update node in search index"""
        self.node_index[node_id] = content

        # Extract searchable text
        searchable_text = self._extract_searchable_text(content)

        # Update inverted index
        for term in searchable_text:
            if node_id not in self.inverted_index[term]:
                self.inverted_index[term].append(node_id)

        logger.debug(f"Indexed node {node_id} with {len(searchable_text)} terms")

    def _extract_searchable_text(self, content: Dict[str, Any]) -> List[str]:
        """Extract searchable terms from content"""
        text_parts = []

        if "title" in content:
            text_parts.append(content["title"])
        if "description" in content:
            text_parts.append(content["description"])
        if "tags" in content:
            text_parts.extend(content["tags"])
        if "learning_objectives" in content:
            text_parts.extend(content["learning_objectives"])

        # Combine and preprocess
        combined_text = " ".join(text_parts)
        query_processor = QueryProcessor()
        terms = query_processor.preprocess_query(combined_text)

        return terms

    def remove_from_index(self, node_id: str) -> None:
        """Remove node from search index"""
        if node_id not in self.node_index:
            return

        # Remove from inverted index
        for term, node_ids in self.inverted_index.items():
            if node_id in node_ids:
                node_ids.remove(node_id)
                if not node_ids:
                    del self.inverted_index[term]

        del self.node_index[node_id]
        logger.debug(f"Removed node {node_id} from index")


class SearchEngine:
    """Main search engine with semantic and keyword search"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_processor = QueryProcessor()
        self.index_manager = IndexManager()

        logger.info("SearchEngine initialized")

    def index_knowledge_base(self, knowledge_nodes: Dict[str, Any]) -> int:
        """Index all knowledge nodes for searching"""
        indexed_count = 0

        for node_id, node_data in knowledge_nodes.items():
            self.index_manager.add_to_index(node_id, node_data)
            indexed_count += 1

        logger.info(f"Indexed {indexed_count} knowledge nodes")
        return indexed_count

    def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[SearchResult]:
        """Perform search with the given query"""
        if not query.strip():
            return []

        logger.info(f"Searching for: '{query}'")

        # Process query
        terms = self.query_processor.preprocess_query(query)
        intent = self.query_processor.extract_intent(query)

        # Merge filters
        search_filters = filters or {}
        search_filters.update(intent.get("filters", {}))

        # Perform search
        results = self._keyword_search(terms, search_filters)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply limit
        results = results[:limit]

        logger.info(f"Search completed: {len(results)} results")
        return results

    def _keyword_search(self, terms: List[str], filters: Dict[str, Any]) -> List[SearchResult]:
        """Perform keyword-based search"""
        if not terms:
            return []

        # Find nodes matching all terms (AND operation)
        candidate_nodes = None

        for term in terms:
            if term in self.index_manager.inverted_index:
                term_nodes = set(self.index_manager.inverted_index[term])

                if candidate_nodes is None:
                    candidate_nodes = term_nodes
                else:
                    candidate_nodes = candidate_nodes.intersection(term_nodes)
            else:
                # Term not found, no results
                return []

        if not candidate_nodes:
            return []

        # Score and filter results
        results = []

        for node_id in candidate_nodes:
            node_data = self.index_manager.node_index.get(node_id)
            if not node_data:
                continue

            # Apply filters
            if self._passes_filters(node_data, filters):
                # Compute relevance score
                score = self._compute_relevance_score(node_id, terms, node_data)

                # Generate snippet
                snippet = self._generate_snippet(node_data, terms)

                result = SearchResult(
                    node_id=node_id,
                    title=node_data.get("title", ""),
                    content_type=node_data.get("content_type", ""),
                    relevance_score=score,
                    snippet=snippet,
                    metadata=node_data
                )

                results.append(result)

        return results

    def _passes_filters(self, node_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if node passes all filters"""
        for filter_name, filter_value in filters.items():
            if filter_name == "content_type" and node_data.get("content_type") != filter_value:
                return False
            if filter_name == "difficulty" and node_data.get("difficulty") != filter_value:
                return False
            # Add more filter types as needed

        return True

    def _compute_relevance_score(self, node_id: str, terms: List[str], node_data: Dict[str, Any]) -> float:
        """Compute relevance score for a node"""
        score = 0.0

        # Title matches get higher weight
        title = node_data.get("title", "").lower()
        for term in terms:
            if term in title:
                score += 2.0

        # Description matches
        description = node_data.get("description", "").lower()
        for term in terms:
            if term in description:
                score += 1.0

        # Tag matches get bonus
        tags = [tag.lower() for tag in node_data.get("tags", [])]
        for term in terms:
            if any(term in tag for tag in tags):
                score += 1.5

        return score

    def _generate_snippet(self, node_data: Dict[str, Any], terms: List[str]) -> str:
        """Generate search result snippet"""
        description = node_data.get("description", "")

        if len(description) <= 150:
            return description

        # Truncate to reasonable length
        snippet = description[:150]

        # Try to end at word boundary
        if 150 < len(description):
            last_space = snippet.rfind(" ")
            if last_space > 100:
                snippet = snippet[:last_space]

        return snippet + "..."

    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions for partial query"""
        if len(partial_query) < 2:
            return []

        suggestions = set()
        partial_lower = partial_query.lower()

        # Find terms that start with the partial query
        for term in self.index_manager.inverted_index.keys():
            if term.startswith(partial_lower):
                suggestions.add(term)

        # Also check node titles
        for node_data in self.index_manager.node_index.values():
            title = node_data.get("title", "").lower()
            if partial_lower in title:
                suggestions.add(node_data["title"])

        return sorted(list(suggestions))[:10]

    def get_faceted_search_options(self) -> Dict[str, List[str]]:
        """Get available filter options for faceted search"""
        facets = {
            "content_types": set(),
            "difficulties": set(),
            "tags": set()
        }

        for node_data in self.index_manager.node_index.values():
            facets["content_types"].add(node_data.get("content_type", ""))
            facets["difficulties"].add(node_data.get("difficulty", ""))

            for tag in node_data.get("tags", []):
                facets["tags"].add(tag)

        return {
            "content_types": sorted(list(facets["content_types"])) if facets["content_types"] else [],
            "difficulties": sorted(list(facets["difficulties"])) if facets["difficulties"] else [],
            "tags": sorted(list(facets["tags"]))[:50]  # Limit tags
        }
