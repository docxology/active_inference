"""
Platform - Search Engine

Intelligent search capabilities for the Active Inference Knowledge Environment.
Provides semantic search, filtering, ranking, and indexing of knowledge content
with natural language query processing and relevance scoring.
"""

import logging
import time
import json
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
    metadata: Optional[Dict[str, Any]] = None

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

    def advanced_search(self, query: str, filters: Dict[str, Any] = None, sort_by: str = "relevance", limit: int = 20) -> List[SearchResult]:
        """Perform advanced search with multiple filters and sorting"""
        if filters is None:
            filters = {}

        # Get basic results
        basic_results = self.search(query, limit=100)  # Get more results for filtering

        # Apply filters
        filtered_results = self._apply_filters(basic_results, filters)

        # Sort results
        sorted_results = self._sort_results(filtered_results, sort_by)

        # Limit results
        return sorted_results[:limit]

    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply filters to search results"""
        filtered = results

        # Filter by content type
        if "content_types" in filters:
            allowed_types = set(filters["content_types"])
            filtered = [r for r in filtered if r.content_type in allowed_types]

        # Filter by difficulty
        if "difficulties" in filters:
            allowed_difficulties = set(filters["difficulties"])
            # Extract difficulty from metadata
            filtered = [r for r in filtered if r.metadata.get("difficulty") in allowed_difficulties]

        # Filter by tags
        if "tags" in filters:
            required_tags = set(filters["tags"])
            filtered = [r for r in filtered if required_tags.issubset(set(r.metadata.get("tags", [])))]

        # Filter by date range
        if "date_from" in filters:
            from_date = filters["date_from"]
            filtered = [r for r in filtered if r.metadata.get("last_updated", "") >= from_date]

        if "date_to" in filters:
            to_date = filters["date_to"]
            filtered = [r for r in filtered if r.metadata.get("last_updated", "") <= to_date]

        return filtered

    def _sort_results(self, results: List[SearchResult], sort_by: str) -> List[SearchResult]:
        """Sort search results by specified criteria"""
        if sort_by == "relevance":
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == "date":
            return sorted(results, key=lambda x: x.metadata.get("last_updated", ""), reverse=True)
        elif sort_by == "title":
            return sorted(results, key=lambda x: x.title.lower())
        elif sort_by == "type":
            return sorted(results, key=lambda x: (x.content_type, x.relevance_score), reverse=True)
        else:
            # Default to relevance
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def search_similar_content(self, node_id: str, limit: int = 10) -> List[SearchResult]:
        """Find content similar to a given node"""
        if node_id not in self.index_manager.node_index:
            return []

        # Get the source node
        source_node = self.index_manager.node_index[node_id]

        # Find nodes with similar tags or content
        similar_nodes = []
        source_tags = set(source_node.get("tags", []))

        for other_id, node_data in self.index_manager.node_index.items():
            if other_id == node_id:
                continue

            # Calculate similarity score
            other_tags = set(node_data.get("tags", []))
            tag_overlap = len(source_tags.intersection(other_tags))

            if tag_overlap > 0:
                # Calculate similarity based on tag overlap and content similarity
                similarity = tag_overlap / max(len(source_tags), len(other_tags))

                # Boost score if content types match
                if node_data.get("content_type") == source_node.get("content_type"):
                    similarity *= 1.2

                similar_nodes.append((other_id, similarity))

        # Sort by similarity and create results
        similar_nodes.sort(key=lambda x: x[1], reverse=True)

        results = []
        for other_id, similarity in similar_nodes[:limit]:
            node_data = self.index_manager.node_index[other_id]
            result = SearchResult(
                node_id=other_id,
                title=node_data["title"],
                content_type=node_data.get("content_type", ""),
                relevance_score=similarity,
                snippet=self._generate_snippet(node_data, []),
                metadata=node_data.get("metadata", {})
            )
            results.append(result)

        return results

    def get_popular_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular search queries"""
        # In a real implementation, this would track search queries
        # For now, return common Active Inference topics
        popular_queries = [
            {"query": "active inference", "count": 100},
            {"query": "free energy principle", "count": 85},
            {"query": "bayesian inference", "count": 70},
            {"query": "entropy", "count": 60},
            {"query": "variational inference", "count": 55},
            {"query": "generative models", "count": 50},
            {"query": "information theory", "count": 45},
            {"query": "expected free energy", "count": 40},
            {"query": "perception", "count": 35},
            {"query": "action", "count": 30}
        ]

        return popular_queries[:limit]

    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and performance metrics"""
        return {
            "total_indexed_nodes": len(self.index_manager.node_index),
            "total_search_terms": len(self.index_manager.inverted_index),
            "index_memory_usage": self._estimate_index_memory_usage(),
            "search_performance": {
                "avg_response_time": 0.05,  # seconds (placeholder)
                "max_response_time": 0.2,
                "min_response_time": 0.01
            },
            "popular_queries": self.get_popular_searches(5),
            "search_trends": {
                "top_content_types": self._get_content_type_distribution(),
                "top_difficulties": self._get_difficulty_distribution(),
                "top_tags": self._get_top_tags()
            }
        }

    def _estimate_index_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage of search indices"""
        # Rough estimation
        node_index_size = len(str(self.index_manager.node_index))
        inverted_index_size = len(str(self.index_manager.inverted_index))

        return {
            "node_index_bytes": node_index_size * 2,  # Rough estimate
            "inverted_index_bytes": inverted_index_size * 2,
            "total_bytes": (node_index_size + inverted_index_size) * 2
        }

    def _get_content_type_distribution(self) -> Dict[str, int]:
        """Get distribution of content types in index"""
        distribution = {}
        for node_data in self.index_manager.node_index.values():
            content_type = node_data.get("content_type", "unknown")
            distribution[content_type] = distribution.get(content_type, 0) + 1

        return distribution

    def _get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of difficulty levels in index"""
        distribution = {}
        for node_data in self.index_manager.node_index.values():
            difficulty = node_data.get("difficulty", "unknown")
            distribution[difficulty] = distribution.get(difficulty, 0) + 1

        return distribution

    def _get_top_tags(self, limit: int = 20) -> Dict[str, int]:
        """Get most frequent tags in index"""
        tag_counts = {}
        for node_data in self.index_manager.node_index.values():
            for tag in node_data.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Sort by frequency and return top tags
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_tags[:limit])

    def reindex_all_content(self) -> Dict[str, Any]:
        """Reindex all content (useful after bulk updates)"""
        logger.info("Starting full reindex of all content")

        # Clear existing indices
        self.index_manager.clear_indices()

        # Rebuild indices
        start_time = time.time()
        nodes_indexed = 0

        for node_id, node_data in self.index_manager.node_index.items():
            self.index_manager.add_to_index(node_id, node_data)
            nodes_indexed += 1

        end_time = time.time()
        indexing_time = end_time - start_time

        logger.info(f"Reindexing completed: {nodes_indexed} nodes in {indexing_time:.2f}s")

        return {
            "success": True,
            "nodes_indexed": nodes_indexed,
            "indexing_time_seconds": indexing_time,
            "avg_indexing_rate": nodes_indexed / indexing_time if indexing_time > 0 else 0
        }

    def optimize_search_indices(self) -> Dict[str, Any]:
        """Optimize search indices for better performance"""
        logger.info("Optimizing search indices")

        # Remove unused terms from inverted index
        start_time = time.time()

        # Get all terms that are actually used in nodes
        used_terms = set()
        for node_data in self.index_manager.node_index.values():
            title_terms = set(node_data.get("title", "").lower().split())
            desc_terms = set(node_data.get("description", "").lower().split())
            tag_terms = set(tag.lower() for tag in node_data.get("tags", []))

            used_terms.update(title_terms, desc_terms, tag_terms)

        # Remove unused terms from inverted index
        removed_terms = 0
        for term in list(self.index_manager.inverted_index.keys()):
            if term not in used_terms:
                del self.index_manager.inverted_index[term]
                removed_terms += 1

        end_time = time.time()
        optimization_time = end_time - start_time

        logger.info(f"Index optimization completed: removed {removed_terms} unused terms in {optimization_time:.2f}s")

        return {
            "success": True,
            "unused_terms_removed": removed_terms,
            "optimization_time_seconds": optimization_time,
            "remaining_terms": len(self.index_manager.inverted_index)
        }

    def validate_search_functionality(self) -> Dict[str, Any]:
        """Validate search functionality and report issues"""
        validation = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }

        # Check index consistency
        if len(self.index_manager.node_index) == 0:
            validation["issues"].append("No content indexed for search")
            validation["valid"] = False

        # Check inverted index consistency
        for term, node_ids in self.index_manager.inverted_index.items():
            for node_id in node_ids:
                if node_id not in self.index_manager.node_index:
                    validation["issues"].append(f"Orphaned reference to node {node_id} for term '{term}'")
                    validation["valid"] = False

        # Check for missing metadata
        missing_metadata = 0
        for node_id, node_data in self.index_manager.node_index.items():
            if not node_data.get("title"):
                missing_metadata += 1
            if not node_data.get("description"):
                missing_metadata += 1

        if missing_metadata > 0:
            validation["warnings"].append(f"{missing_metadata} nodes missing title or description")

        # Performance recommendations
        if len(self.index_manager.inverted_index) > 1000:
            validation["recommendations"].append("Consider index optimization - large number of terms")

        return validation

    def export_search_data(self, format: str = "json") -> str:
        """Export search index data for backup or analysis"""
        export_data = {
            "node_index": dict(self.index_manager.node_index),
            "inverted_index": dict(self.index_manager.inverted_index),
            "metadata": {
                "total_nodes": len(self.index_manager.node_index),
                "total_terms": len(self.index_manager.inverted_index),
                "export_timestamp": time.time(),
                "export_format": format
            }
        }

        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            return str(export_data)

    def import_search_data(self, data: Dict[str, Any]) -> bool:
        """Import search index data from backup"""
        try:
            if "node_index" in data:
                self.index_manager.node_index = data["node_index"]
            if "inverted_index" in data:
                self.index_manager.inverted_index = data["inverted_index"]

            logger.info(f"Search data imported successfully: {len(self.index_manager.node_index)} nodes")
            return True

        except Exception as e:
            logger.error(f"Failed to import search data: {e}")
            return False
