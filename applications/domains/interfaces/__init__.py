"""
Domain Interfaces Module

This module provides interfaces for domain-specific Active Inference implementations.
Each domain interface provides specialized tools and methods for applying Active Inference
in particular research areas and applications.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Import domain interfaces
try:
    from .neuroscience import NeuroscienceInterface
    from .robotics import RoboticsInterface
    from .psychology import PsychologyInterface
    from .artificial_intelligence import ArtificialIntelligenceInterface
    from .engineering import EngineeringInterface
    from .education import EducationInterface

    # Make interfaces available at module level
    __all__ = [
        'NeuroscienceInterface',
        'RoboticsInterface',
        'PsychologyInterface',
        'ArtificialIntelligenceInterface',
        'EngineeringInterface',
        'EducationInterface'
    ]

except ImportError as e:
    logger.warning("Some domain interfaces could not be imported: %s", str(e))

    # Create placeholder classes for missing interfaces
    class PlaceholderInterface:
        """Placeholder interface for missing domain implementations"""

        def __init__(self, config: Dict[str, Any]):
            self.config = config
            logger.warning("Using placeholder interface - domain not fully implemented")

        def process(self, data: Any) -> Dict[str, Any]:
            """Placeholder processing method"""
            return {'status': 'placeholder', 'message': 'Domain interface not implemented'}

    # Define placeholder interfaces
    NeuroscienceInterface = PlaceholderInterface
    RoboticsInterface = PlaceholderInterface
    PsychologyInterface = PlaceholderInterface
    ArtificialIntelligenceInterface = PlaceholderInterface
    EngineeringInterface = PlaceholderInterface
    EducationInterface = PlaceholderInterface

    __all__ = [
        'NeuroscienceInterface',
        'RoboticsInterface',
        'PsychologyInterface',
        'ArtificialIntelligenceInterface',
        'EngineeringInterface',
        'EducationInterface'
    ]

# Domain interface registry
DOMAIN_INTERFACES = {
    'neuroscience': 'Neural and brain modeling interfaces',
    'robotics': 'Robot control and autonomous systems interfaces',
    'psychology': 'Cognitive and behavioral modeling interfaces',
    'artificial_intelligence': 'AI and machine learning interfaces',
    'engineering': 'Control systems and optimization interfaces',
    'education': 'Learning systems and educational interfaces'
}

class DomainInterfaceRegistry:
    """Registry for domain-specific interfaces"""

    def __init__(self):
        self.interfaces = {}
        self._load_interfaces()

    def _load_interfaces(self) -> None:
        """Load all domain interfaces"""
        try:
            domain_classes = {
                'neuroscience': 'NeuroscienceInterface',
                'robotics': 'RoboticsInterface',
                'psychology': 'PsychologyInterface',
                'artificial_intelligence': 'ArtificialIntelligenceInterface',
                'engineering': 'EngineeringInterface',
                'education': 'EducationInterface'
            }

            for domain, class_name in domain_classes.items():
                try:
                    module = __import__(f'active_inference.applications.domains.interfaces.{domain}',
                                      fromlist=[class_name])
                    if hasattr(module, class_name):
                        self.interfaces[domain] = getattr(module, class_name)
                        logger.info("Loaded interface: %s", domain)
                except ImportError as e:
                    logger.warning("Could not load interface %s: %s", domain, str(e))
        except Exception as e:
            logger.error("Error loading interfaces: %s", str(e))

    def get_interface(self, domain: str) -> Optional[Any]:
        """Get domain interface by name"""
        return self.interfaces.get(domain)

    def list_interfaces(self) -> Dict[str, str]:
        """List all available domain interfaces"""
        return DOMAIN_INTERFACES.copy()

    def create_interface(self, domain: str, config: Dict[str, Any]) -> Optional[Any]:
        """Create domain interface instance"""
        interface_class = self.get_interface(domain)
        if interface_class is None:
            logger.error("Interface not found: %s", domain)
            return None

        try:
            instance = interface_class(config)
            logger.info("Created interface: %s", domain)
            return instance
        except Exception as e:
            logger.error("Failed to create interface %s: %s", domain, str(e))
            return None

# Global interface registry
interface_registry = DomainInterfaceRegistry()

def get_interface_registry() -> DomainInterfaceRegistry:
    """Get the global interface registry"""
    return interface_registry

def create_domain_interface(domain: str, config: Dict[str, Any]) -> Optional[Any]:
    """Create a domain-specific interface"""
    return interface_registry.create_interface(domain, config)

def list_available_interfaces() -> Dict[str, str]:
    """List all available domain interfaces"""
    return interface_registry.list_interfaces()