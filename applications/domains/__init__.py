"""
Domain Applications Module

This module provides domain-specific entry points, interfaces, and bundled implementations
for applying Active Inference across different fields of study and application.

Each domain provides curated interfaces to core Active Inference functionality,
optimized for specific use cases and research areas.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Domain registry for tracking available domains
DOMAIN_REGISTRY = {
    'neuroscience': 'Biological and neural systems applications',
    'robotics': 'Autonomous systems and control applications',
    'psychology': 'Cognitive and behavioral models',
    'artificial_intelligence': 'AI and machine learning applications',
    'engineering': 'Control systems and optimization',
    'education': 'Learning and teaching tools'
}

class DomainManager:
    """Manages domain-specific Active Inference applications and interfaces"""

    def __init__(self):
        """Initialize the domain manager"""
        self.domains = {}
        self.active_domain = None
        self._load_domains()

    def _load_domains(self) -> None:
        """Load all available domain implementations"""
        try:
            # Import domain modules dynamically
            for domain_name in DOMAIN_REGISTRY.keys():
                try:
                    module = __import__(f'active_inference.applications.domains.{domain_name}',
                                      fromlist=[''])
                    if hasattr(module, 'DomainInterface'):
                        self.domains[domain_name] = module.DomainInterface
                        logger.info("Loaded domain: %s", domain_name)
                except ImportError as e:
                    logger.warning("Could not load domain %s: %s", domain_name, str(e))
        except Exception as e:
            logger.error("Error loading domains: %s", str(e))

    def get_domain(self, domain_name: str) -> Optional[Any]:
        """
        Get a domain interface by name

        Args:
            domain_name: Name of the domain to retrieve

        Returns:
            Domain interface class or None if not found
        """
        if domain_name not in self.domains:
            logger.warning("Domain %s not found", domain_name)
            return None
        return self.domains[domain_name]

    def list_domains(self) -> Dict[str, str]:
        """
        List all available domains with descriptions

        Returns:
            Dictionary mapping domain names to descriptions
        """
        return DOMAIN_REGISTRY.copy()

    def activate_domain(self, domain_name: str) -> bool:
        """
        Activate a domain for current session

        Args:
            domain_name: Name of domain to activate

        Returns:
            True if activation successful, False otherwise
        """
        if domain_name not in self.domains:
            logger.error("Cannot activate unknown domain: %s", domain_name)
            return False

        self.active_domain = domain_name
        logger.info("Activated domain: %s", domain_name)
        return True

    def get_active_domain(self) -> Optional[str]:
        """
        Get the currently active domain

        Returns:
            Name of active domain or None
        """
        return self.active_domain

    def create_domain_interface(self, domain_name: str, config: Dict[str, Any]) -> Optional[Any]:
        """
        Create a domain interface instance

        Args:
            domain_name: Name of domain
            config: Configuration for domain interface

        Returns:
            Domain interface instance or None if creation failed
        """
        domain_class = self.get_domain(domain_name)
        if domain_class is None:
            return None

        try:
            instance = domain_class(config)
            logger.info("Created domain interface: %s", domain_name)
            return instance
        except Exception as e:
            logger.error("Failed to create domain interface %s: %s", domain_name, str(e))
            return None

# Global domain manager instance
domain_manager = DomainManager()

def get_domain_manager() -> DomainManager:
    """
    Get the global domain manager instance

    Returns:
        Domain manager instance
    """
    return domain_manager

def list_available_domains() -> Dict[str, str]:
    """
    List all available domain applications

    Returns:
        Dictionary mapping domain names to descriptions
    """
    return domain_manager.list_domains()

def create_domain_interface(domain_name: str, config: Dict[str, Any]) -> Optional[Any]:
    """
    Create a domain-specific interface

    Args:
        domain_name: Name of the domain
        config: Configuration for the domain interface

    Returns:
        Domain interface instance or None if creation failed
    """
    return domain_manager.create_domain_interface(domain_name, config)

def activate_domain(domain_name: str) -> bool:
    """
    Activate a domain for the current session

    Args:
        domain_name: Name of domain to activate

    Returns:
        True if activation successful, False otherwise
    """
    return domain_manager.activate_domain(domain_name)

def get_active_domain() -> Optional[str]:
    """
    Get the currently active domain

    Returns:
        Name of active domain or None
    """
    return domain_manager.get_active_domain()

# Convenience functions for quick access to common domains
def neuroscience(config: Dict[str, Any]) -> Optional[Any]:
    """Create neuroscience domain interface"""
    return create_domain_interface('neuroscience', config)

def robotics(config: Dict[str, Any]) -> Optional[Any]:
    """Create robotics domain interface"""
    return create_domain_interface('robotics', config)

def psychology(config: Dict[str, Any]) -> Optional[Any]:
    """Create psychology domain interface"""
    return create_domain_interface('psychology', config)

def artificial_intelligence(config: Dict[str, Any]) -> Optional[Any]:
    """Create artificial intelligence domain interface"""
    return create_domain_interface('artificial_intelligence', config)

def engineering(config: Dict[str, Any]) -> Optional[Any]:
    """Create engineering domain interface"""
    return create_domain_interface('engineering', config)

def education(config: Dict[str, Any]) -> Optional[Any]:
    """Create education domain interface"""
    return create_domain_interface('education', config)
