# Accessibility and Inclusive Design Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Implement Comprehensive Accessibility and Inclusive Design

You are tasked with developing comprehensive accessibility and inclusive design implementations for the Active Inference Knowledge Environment. This involves creating interfaces and content that are usable by people with diverse abilities, following WCAG guidelines, and ensuring equitable access to complex scientific and technical information.

## 📋 Accessibility Requirements

### Core Accessibility Standards (MANDATORY)
1. **WCAG 2.1 AA Compliance**: Meet all Web Content Accessibility Guidelines
2. **Multiple Modality Support**: Support for visual, auditory, motor, and cognitive accessibility
3. **Progressive Enhancement**: Core functionality accessible without advanced features
4. **Universal Design**: Design for the widest possible range of users
5. **Continuous Testing**: Regular accessibility audits and user testing

### Accessibility Architecture Components
```
accessibility/
├── wcag_compliance.py           # WCAG guideline implementation and validation
├── screen_reader_support.py     # Screen reader optimization and testing
├── keyboard_navigation.py       # Keyboard-only navigation systems
├── color_accessibility.py      # Color contrast and visual accessibility
├── cognitive_support.py         # Cognitive accessibility and plain language
├── motor_accessibility.py       # Motor impairment support and alternatives
├── multimodal_interfaces.py     # Multiple interface modalities
└── accessibility_testing.py     # Automated accessibility testing and validation
```

## 🏗️ WCAG Compliance Framework

### Phase 1: Perceivable Content Implementation

#### 1.1 Alternative Text and Media Alternatives
```python
from typing import Dict, List, Any, Optional, Callable
import re
import logging

class AccessibilityValidator:
    """Comprehensive accessibility validation framework"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize accessibility validator"""
        self.config = config
        self.logger = logging.getLogger('AccessibilityValidator')

        # WCAG compliance levels
        self.compliance_levels = {
            'A': 1.0,    # Basic accessibility
            'AA': 2.0,   # Enhanced accessibility
            'AAA': 3.0   # Highest accessibility standard
        }

        # Validation rules
        self.validation_rules = self.initialize_validation_rules()

    def initialize_validation_rules(self) -> Dict[str, Callable]:
        """Initialize accessibility validation rules"""
        return {
            'image_alt_text': self.check_image_alt_text,
            'color_contrast': self.check_color_contrast,
            'keyboard_navigation': self.check_keyboard_navigation,
            'screen_reader_compatibility': self.check_screen_reader_compatibility,
            'text_alternatives': self.check_text_alternatives,
            'semantic_markup': self.check_semantic_markup,
            'focus_management': self.check_focus_management,
            'error_identification': self.check_error_identification,
            'language_identification': self.check_language_identification,
            'consistent_navigation': self.check_consistent_navigation
        }

class ImageAccessibilityHandler:
    """Handle image accessibility with alternative text and descriptions"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize image accessibility handler"""
        self.config = config
        self.logger = logging.getLogger('ImageAccessibilityHandler')

    def generate_alt_text(self, image_path: str, context: str = "") -> str:
        """Generate appropriate alt text for images"""
        # Use AI/ML to analyze image and generate descriptive alt text
        # For now, provide template-based generation

        image_description = self.analyze_image_content(image_path)

        if context:
            alt_text = f"{context}: {image_description}"
        else:
            alt_text = image_description

        # Ensure alt text is concise but descriptive
        if len(alt_text) > 125:
            alt_text = alt_text[:122] + "..."

        return alt_text

    def analyze_image_content(self, image_path: str) -> str:
        """Analyze image content for alt text generation"""
        # Placeholder for image analysis
        # In practice, would use computer vision APIs

        # Return descriptive text based on filename/context
        if 'graph' in image_path.lower():
            return "Data visualization graph showing relationships between concepts"
        elif 'diagram' in image_path.lower():
            return "Technical diagram illustrating system architecture"
        elif 'chart' in image_path.lower():
            return "Chart displaying quantitative data and trends"
        else:
            return "Visual content supporting the educational material"

    def create_image_descriptions(self, image_path: str) -> Dict[str, str]:
        """Create comprehensive image descriptions for different contexts"""
        return {
            'alt_text': self.generate_alt_text(image_path),
            'long_description': self.generate_long_description(image_path),
            'caption': self.generate_caption(image_path),
            'transcript': self.generate_transcript(image_path) if self.is_complex_image(image_path) else None
        }

    def generate_long_description(self, image_path: str) -> str:
        """Generate detailed description for complex images"""
        # For complex images like graphs, provide detailed textual description
        base_description = self.analyze_image_content(image_path)

        detailed_description = f"""
        Detailed Description: {base_description}

        This visualization presents complex relationships and data patterns.
        Key elements include: [detailed breakdown would be generated here]

        For a complete understanding, please refer to the surrounding text
        and data tables that accompany this visual representation.
        """

        return detailed_description.strip()

    def generate_caption(self, image_path: str) -> str:
        """Generate caption for images"""
        description = self.analyze_image_content(image_path)
        return f"Figure: {description}"

    def generate_transcript(self, image_path: str) -> Optional[str]:
        """Generate transcript for complex visual content"""
        if not self.is_complex_image(image_path):
            return None

        return f"""
        Transcript of visual content:
        [Detailed textual representation of complex visual information]

        This transcript provides the essential information conveyed by the
        visual element for users who cannot access the graphical representation.
        """

    def is_complex_image(self, image_path: str) -> bool:
        """Determine if image requires complex description"""
        complex_indicators = ['graph', 'chart', 'diagram', 'matrix', 'plot']
        return any(indicator in image_path.lower() for indicator in complex_indicators)

    def validate_image_accessibility(self, image_element: Dict[str, Any]) -> List[str]:
        """Validate image accessibility compliance"""
        issues = []

        # Check for alt text
        if not image_element.get('alt'):
            issues.append("Missing alt text attribute")

        # Check alt text quality
        alt_text = image_element.get('alt', '')
        if len(alt_text) < 5:
            issues.append("Alt text too short - should be descriptive")
        elif len(alt_text) > 125:
            issues.append("Alt text too long - should be concise")

        # Check for decorative images
        if image_element.get('decorative', False) and alt_text:
            issues.append("Decorative images should have empty alt text")

        # Check for complex images
        if self.is_complex_image(image_element.get('src', '')):
            if not image_element.get('long_description'):
                issues.append("Complex images require long descriptions")
            if not image_element.get('transcript'):
                issues.append("Complex images should have transcripts")

        return issues

class MediaAccessibilityHandler:
    """Handle audio and video accessibility"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize media accessibility handler"""
        self.config = config
        self.logger = logging.getLogger('MediaAccessibilityHandler')

    def create_audio_description(self, audio_content: Dict[str, Any]) -> Dict[str, str]:
        """Create audio descriptions and transcripts"""
        return {
            'transcript': self.generate_transcript(audio_content),
            'description': self.generate_audio_description(audio_content),
            'chapters': self.generate_chapters(audio_content)
        }

    def generate_transcript(self, audio_content: Dict[str, Any]) -> str:
        """Generate transcript for audio content"""
        # In practice, would use speech-to-text APIs
        # Placeholder implementation
        return f"""
        Audio Transcript:
        [Speaker]: {audio_content.get('transcript', 'Audio content description')}

        [Timestamps and speaker identification would be included here]
        """

    def generate_audio_description(self, audio_content: Dict[str, Any]) -> str:
        """Generate description of audio content"""
        duration = audio_content.get('duration', 'unknown')
        speakers = audio_content.get('speakers', ['unknown'])

        description = f"""
        Audio Content Description:
        - Duration: {duration}
        - Speakers: {', '.join(speakers)}
        - Content: {audio_content.get('description', 'Educational audio content')}
        - Language: {audio_content.get('language', 'English')}
        """

        return description.strip()

    def generate_chapters(self, audio_content: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate chapter markers for long audio content"""
        # Placeholder for chapter generation
        return [
            {'timestamp': '00:00', 'title': 'Introduction'},
            {'timestamp': '05:00', 'title': 'Main Content'},
            {'timestamp': '15:00', 'title': 'Conclusion'}
        ]

    def create_video_accessibility(self, video_content: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive video accessibility features"""
        return {
            'captions': self.generate_captions(video_content),
            'audio_description': self.generate_video_audio_description(video_content),
            'transcript': self.generate_video_transcript(video_content),
            'sign_language': self.check_sign_language_available(video_content),
            'text_alternatives': self.generate_text_alternatives(video_content)
        }

    def generate_captions(self, video_content: Dict[str, Any]) -> str:
        """Generate captions for video content"""
        # In practice, would integrate with speech recognition and timing
        return """
        [00:00] Speaker: Welcome to this educational video on Active Inference.
        [00:05] Speaker: Today we'll explore the fundamental principles...
        """

    def generate_video_audio_description(self, video_content: Dict[str, Any]) -> str:
        """Generate audio descriptions for video content"""
        return """
        Audio Description: The screen shows a complex diagram with interconnected nodes
        representing concepts in Active Inference. Arrows flow between nodes showing
        the direction of information processing...
        """

    def generate_video_transcript(self, video_content: Dict[str, Any]) -> str:
        """Generate full transcript for video content"""
        return """
        Full Video Transcript:

        [Visual Description]: Opening scene shows the Active Inference Knowledge Environment logo

        [00:00] Narrator: Welcome to the Active Inference Knowledge Environment.
        Today we explore how biological systems process information...

        [Visual Description]: Animation shows neural networks lighting up with activity

        [02:30] Narrator: At the core of Active Inference is the Free Energy Principle...
        """

    def check_sign_language_available(self, video_content: Dict[str, Any]) -> bool:
        """Check if sign language interpretation is available"""
        # Placeholder - would check for sign language tracks
        return video_content.get('sign_language_available', False)

    def generate_text_alternatives(self, video_content: Dict[str, Any]) -> str:
        """Generate text-based alternatives for video content"""
        return f"""
        Text Alternative for Video Content:

        Title: {video_content.get('title', 'Video Content')}

        Summary: {video_content.get('summary', 'Educational video content')}

        Key Points:
        - {video_content.get('key_point_1', 'Main educational concept')}
        - {video_content.get('key_point_2', 'Supporting information')}
        - {video_content.get('key_point_3', 'Practical applications')}

        For full interactive experience, enable video playback with captions.
        """
```

### Phase 2: Operable Interface Design

#### 2.1 Keyboard Navigation System
```python
from typing import Dict, List, Any, Optional, Callable, Tuple
import logging

class KeyboardNavigationManager:
    """Comprehensive keyboard navigation management system"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize keyboard navigation manager"""
        self.config = config
        self.logger = logging.getLogger('KeyboardNavigationManager')

        # Navigation state
        self.focusable_elements = []
        self.current_focus_index = 0
        self.navigation_history = []

        # Keyboard shortcuts
        self.keyboard_shortcuts = self.initialize_keyboard_shortcuts()

    def initialize_keyboard_shortcuts(self) -> Dict[str, Callable]:
        """Initialize keyboard shortcuts for navigation"""
        return {
            'Tab': self.navigate_next,
            'Shift+Tab': self.navigate_previous,
            'ArrowDown': self.navigate_next,
            'ArrowRight': self.navigate_next,
            'ArrowUp': self.navigate_previous,
            'ArrowLeft': self.navigate_previous,
            'Home': self.navigate_first,
            'End': self.navigate_last,
            'Enter': self.activate_current,
            'Space': self.activate_current,
            'Escape': self.escape_current,
            'h': self.show_help,  # With modifier key
            's': self.skip_to_content,  # Skip navigation
            'm': self.show_menu,
            'f': self.show_search,
            '?': self.show_help
        }

    def register_focusable_element(self, element_id: str, element_info: Dict[str, Any]) -> None:
        """Register a focusable element in the navigation system"""
        element_info['element_id'] = element_id
        element_info['tab_index'] = len(self.focusable_elements)

        self.focusable_elements.append(element_info)

        # Set appropriate ARIA attributes
        self.set_aria_attributes(element_id, element_info)

    def set_aria_attributes(self, element_id: str, element_info: Dict[str, Any]) -> None:
        """Set ARIA attributes for accessibility"""
        aria_attributes = {
            'role': element_info.get('role', 'button'),
            'tabindex': element_info.get('tab_index', 0),
            'aria-label': element_info.get('aria_label', element_info.get('label', '')),
            'aria-describedby': element_info.get('aria_describedby', ''),
            'aria-expanded': element_info.get('aria_expanded', False),
            'aria-selected': element_info.get('aria_selected', False)
        }

        # In practice, this would update the DOM element
        self.logger.debug(f"Setting ARIA attributes for {element_id}: {aria_attributes}")

    def navigate_next(self) -> Dict[str, Any]:
        """Navigate to next focusable element"""
        if not self.focusable_elements:
            return {'success': False, 'message': 'No focusable elements'}

        self.current_focus_index = (self.current_focus_index + 1) % len(self.focusable_elements)
        next_element = self.focusable_elements[self.current_focus_index]

        self.navigation_history.append(f"next_{next_element['element_id']}")

        return {
            'success': True,
            'element_id': next_element['element_id'],
            'element_info': next_element
        }

    def navigate_previous(self) -> Dict[str, Any]:
        """Navigate to previous focusable element"""
        if not self.focusable_elements:
            return {'success': False, 'message': 'No focusable elements'}

        self.current_focus_index = (self.current_focus_index - 1) % len(self.focusable_elements)
        prev_element = self.focusable_elements[self.current_focus_index]

        self.navigation_history.append(f"previous_{prev_element['element_id']}")

        return {
            'success': True,
            'element_id': prev_element['element_id'],
            'element_info': prev_element
        }

    def navigate_first(self) -> Dict[str, Any]:
        """Navigate to first focusable element"""
        if not self.focusable_elements:
            return {'success': False, 'message': 'No focusable elements'}

        self.current_focus_index = 0
        first_element = self.focusable_elements[0]

        return {
            'success': True,
            'element_id': first_element['element_id'],
            'element_info': first_element
        }

    def navigate_last(self) -> Dict[str, Any]:
        """Navigate to last focusable element"""
        if not self.focusable_elements:
            return {'success': False, 'message': 'No focusable elements'}

        self.current_focus_index = len(self.focusable_elements) - 1
        last_element = self.focusable_elements[-1]

        return {
            'success': True,
            'element_id': last_element['element_id'],
            'element_info': last_element
        }

    def activate_current(self) -> Dict[str, Any]:
        """Activate currently focused element"""
        if not self.focusable_elements or self.current_focus_index >= len(self.focusable_elements):
            return {'success': False, 'message': 'No current element'}

        current_element = self.focusable_elements[self.current_focus_index]

        # Perform activation based on element type
        activation_result = self.perform_activation(current_element)

        return {
            'success': True,
            'element_id': current_element['element_id'],
            'activation_type': activation_result.get('type', 'unknown'),
            'result': activation_result
        }

    def perform_activation(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Perform element activation"""
        element_type = element.get('type', 'unknown')

        if element_type == 'button':
            return {'type': 'click', 'action': 'button_clicked'}
        elif element_type == 'link':
            return {'type': 'navigation', 'url': element.get('href', '#')}
        elif element_type == 'menu_item':
            return {'type': 'menu_action', 'action': element.get('action', 'unknown')}
        elif element_type == 'form_field':
            return {'type': 'focus', 'field_type': element.get('field_type', 'text')}
        else:
            return {'type': 'unknown', 'element_info': element}

    def escape_current(self) -> Dict[str, Any]:
        """Escape from current context or element"""
        # Close menus, dialogs, etc.
        return {'success': True, 'action': 'escape', 'context_cleared': True}

    def skip_to_content(self) -> Dict[str, Any]:
        """Skip navigation and go directly to main content"""
        # Find main content area
        main_content = next(
            (elem for elem in self.focusable_elements if elem.get('role') == 'main'),
            None
        )

        if main_content:
            self.current_focus_index = main_content['tab_index']
            return {
                'success': True,
                'element_id': main_content['element_id'],
                'action': 'skip_to_content'
            }
        else:
            return {'success': False, 'message': 'Main content area not found'}

    def show_help(self) -> Dict[str, Any]:
        """Show keyboard navigation help"""
        help_content = """
        Keyboard Navigation Help:

        Navigation:
        - Tab / Arrow Keys: Move between elements
        - Shift+Tab: Move backwards
        - Home/End: First/Last element
        - Enter/Space: Activate element
        - Escape: Close menus/dialogs

        Shortcuts:
        - H: Show this help
        - S: Skip to main content
        - M: Show menu
        - F: Focus search
        - ?: Show help

        Current focus: {current_element}
        """

        current_element = ""
        if self.focusable_elements and self.current_focus_index < len(self.focusable_elements):
            current_element = self.focusable_elements[self.current_focus_index].get('aria_label', 'unknown')

        help_content = help_content.format(current_element=current_element)

        return {
            'success': True,
            'action': 'show_help',
            'help_content': help_content.strip()
        }

    def show_menu(self) -> Dict[str, Any]:
        """Show navigation menu"""
        menu_items = [elem for elem in self.focusable_elements if elem.get('role') == 'menuitem']

        return {
            'success': True,
            'action': 'show_menu',
            'menu_items': menu_items
        }

    def show_search(self) -> Dict[str, Any]:
        """Focus search input"""
        search_element = next(
            (elem for elem in self.focusable_elements if elem.get('role') == 'searchbox'),
            None
        )

        if search_element:
            self.current_focus_index = search_element['tab_index']
            return {
                'success': True,
                'element_id': search_element['element_id'],
                'action': 'focus_search'
            }
        else:
            return {'success': False, 'message': 'Search element not found'}

    def handle_keyboard_event(self, key_event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle keyboard event for navigation"""
        key = key_event.get('key', '')
        modifiers = key_event.get('modifiers', [])

        # Create combined key string
        if modifiers:
            combined_key = '+'.join(modifiers + [key])
        else:
            combined_key = key

        # Find matching shortcut
        if combined_key in self.keyboard_shortcuts:
            handler = self.keyboard_shortcuts[combined_key]
            return handler()
        else:
            return {'success': False, 'message': f'Unknown key combination: {combined_key}'}

    def get_navigation_state(self) -> Dict[str, Any]:
        """Get current navigation state"""
        current_element = None
        if self.focusable_elements and self.current_focus_index < len(self.focusable_elements):
            current_element = self.focusable_elements[self.current_focus_index]

        return {
            'current_focus_index': self.current_focus_index,
            'total_elements': len(self.focusable_elements),
            'current_element': current_element,
            'navigation_history': self.navigation_history[-10:],  # Last 10 actions
            'available_shortcuts': list(self.keyboard_shortcuts.keys())
        }

    def validate_keyboard_accessibility(self) -> List[str]:
        """Validate keyboard accessibility compliance"""
        issues = []

        # Check for focusable elements
        if not self.focusable_elements:
            issues.append("No focusable elements found")

        # Check for logical tab order
        tab_indices = [elem.get('tab_index', 0) for elem in self.focusable_elements]
        if tab_indices != sorted(tab_indices):
            issues.append("Tab order is not logical")

        # Check for skip links
        skip_links = [elem for elem in self.focusable_elements if elem.get('role') == 'link' and 'skip' in elem.get('aria_label', '').lower()]
        if not skip_links:
            issues.append("Missing skip navigation links")

        # Check for keyboard shortcuts documentation
        if not hasattr(self, 'keyboard_help_available') or not self.keyboard_help_available:
            issues.append("Keyboard shortcuts not documented")

        return issues

class FocusManagementSystem:
    """Advanced focus management for complex interfaces"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize focus management system"""
        self.config = config
        self.logger = logging.getLogger('FocusManagementSystem')

        # Focus state
        self.focus_stack = []  # Stack of focus contexts
        self.focus_traps = {}  # Modal dialogs and focus traps
        self.focus_history = []

    def manage_focus_context(self, context_id: str, context_type: str) -> None:
        """Manage focus within different contexts (modals, menus, etc.)"""
        context_info = {
            'context_id': context_id,
            'context_type': context_type,
            'focusable_elements': [],
            'entry_point': None,
            'exit_point': None
        }

        self.focus_stack.append(context_info)

    def set_focus_trap(self, trap_id: str, trap_elements: List[str]) -> None:
        """Set up a focus trap for modal dialogs"""
        self.focus_traps[trap_id] = {
            'elements': trap_elements,
            'active': True,
            'entry_focus': trap_elements[0] if trap_elements else None
        }

    def release_focus_trap(self, trap_id: str) -> None:
        """Release focus trap"""
        if trap_id in self.focus_traps:
            self.focus_traps[trap_id]['active'] = False

    def handle_focus_change(self, from_element: str, to_element: str, reason: str) -> None:
        """Handle focus changes and maintain history"""
        focus_change = {
            'from_element': from_element,
            'to_element': to_element,
            'reason': reason,
            'timestamp': self.get_current_time(),
            'context': self.get_current_context()
        }

        self.focus_history.append(focus_change)

        # Keep only recent history
        if len(self.focus_history) > 100:
            self.focus_history = self.focus_history[-100:]

    def validate_focus_flow(self) -> List[str]:
        """Validate focus flow accessibility"""
        issues = []

        # Check for focus traps without escape
        for trap_id, trap_info in self.focus_traps.items():
            if trap_info['active']:
                # Check if trap has proper escape mechanism
                if not trap_info.get('escape_handler'):
                    issues.append(f"Focus trap {trap_id} missing escape mechanism")

        # Check focus history for problematic patterns
        recent_changes = self.focus_history[-20:]
        unexpected_jumps = sum(1 for change in recent_changes
                             if change['reason'] == 'unexpected_jump')

        if unexpected_jumps > 5:
            issues.append("Excessive unexpected focus jumps detected")

        return issues

    def get_current_time(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()

    def get_current_context(self) -> str:
        """Get current focus context"""
        if self.focus_stack:
            return self.focus_stack[-1]['context_id']
        return 'global'
```

### Phase 3: Understandable Content Design

#### 3.1 Cognitive Accessibility Framework
```python
from typing import Dict, List, Any, Optional, Callable
import re
import logging

class CognitiveAccessibilityManager:
    """Cognitive accessibility and plain language support"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize cognitive accessibility manager"""
        self.config = config
        self.logger = logging.getLogger('CognitiveAccessibilityManager')

        # Readability metrics
        self.readability_algorithms = {
            'flesch_kincaid': self.calculate_flesch_kincaid,
            'gunning_fog': self.calculate_gunning_fog,
            'smog': self.calculate_smog_index,
            'coleman_liau': self.calculate_coleman_liau
        }

        # Plain language guidelines
        self.plain_language_rules = self.initialize_plain_language_rules()

    def initialize_plain_language_rules(self) -> Dict[str, Callable]:
        """Initialize plain language transformation rules"""
        return {
            'simplify_vocabulary': self.simplify_vocabulary,
            'shorten_sentences': self.shorten_sentences,
            'active_voice': self.convert_to_active_voice,
            'remove_jargon': self.remove_jargon,
            'add_examples': self.add_examples,
            'structure_content': self.structure_content
        }

    def assess_readability(self, text: str) -> Dict[str, Any]:
        """Assess readability of text using multiple metrics"""
        sentences = self.split_into_sentences(text)
        words = self.tokenize_words(text)
        syllables = sum(self.count_syllables(word) for word in words)

        readability_scores = {}

        for algorithm_name, algorithm_func in self.readability_algorithms.items():
            score = algorithm_func(sentences, words, syllables)
            readability_scores[algorithm_name] = score

        # Overall assessment
        avg_score = sum(readability_scores.values()) / len(readability_scores)

        assessment = {
            'scores': readability_scores,
            'average_score': avg_score,
            'grade_level': self.score_to_grade_level(avg_score),
            'readability_level': self.classify_readability(avg_score),
            'recommendations': self.generate_readability_recommendations(readability_scores)
        }

        return assessment

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - in practice, use NLP library
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"

        if word[0] in vowels:
            count += 1

        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1

        if word.endswith("e"):
            count -= 1

        if count == 0:
            count += 1

        return count

    def calculate_flesch_kincaid(self, sentences: List[str], words: List[str], syllables: int) -> float:
        """Calculate Flesch-Kincaid readability score"""
        if not sentences or not words:
            return 0.0

        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = syllables

        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

        return max(0, min(100, score))  # Clamp to 0-100

    def calculate_gunning_fog(self, sentences: List[str], words: List[str], syllables: int) -> float:
        """Calculate Gunning Fog index"""
        if not sentences or not words:
            return 0.0

        total_sentences = len(sentences)
        total_words = len(words)

        # Count complex words (3+ syllables)
        complex_words = sum(1 for word in words if self.count_syllables(word) >= 3)

        # Gunning Fog formula
        score = 0.4 * ((total_words / total_sentences) + 100 * (complex_words / total_words))

        return score

    def calculate_smog_index(self, sentences: List[str], words: List[str], syllables: int) -> float:
        """Calculate SMOG index"""
        if len(sentences) < 3:
            return 0.0

        # Count polysyllabic words (3+ syllables)
        polysyllabic = sum(1 for word in words if self.count_syllables(word) >= 3)

        # SMOG formula
        score = 1.043 * (polysyllabic ** 0.5) * (30 / len(sentences)) ** 0.5 + 3.1291

        return score

    def calculate_coleman_liau(self, sentences: List[str], words: List[str], syllables: int) -> float:
        """Calculate Coleman-Liau index"""
        if not sentences or not words:
            return 0.0

        total_sentences = len(sentences)
        total_words = len(words)

        # Count characters
        total_characters = sum(len(word) for word in words)

        # Coleman-Liau formula
        score = 0.0588 * (total_characters / total_words * 100) - 0.296 * (total_sentences / total_words * 100) - 15.8

        return score

    def score_to_grade_level(self, score: float) -> str:
        """Convert readability score to grade level"""
        # Using Flesch-Kincaid grade level approximation
        grade_level = 0.39 * (score) + 3.0

        if grade_level < 1:
            return "Kindergarten"
        elif grade_level < 2:
            return "1st Grade"
        elif grade_level < 3:
            return "2nd Grade"
        else:
            return f"{int(grade_level)}th Grade"

    def classify_readability(self, score: float) -> str:
        """Classify readability level"""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"

    def generate_readability_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate readability improvement recommendations"""
        recommendations = []

        avg_score = sum(scores.values()) / len(scores)

        if avg_score < 60:
            recommendations.append("Simplify vocabulary - use shorter, more common words")
            recommendations.append("Break down complex sentences into shorter ones")
            recommendations.append("Add examples and analogies to explain complex concepts")

        if scores.get('gunning_fog', 0) > 12:
            recommendations.append("Reduce use of complex words (3+ syllables)")

        if scores.get('flesch_kincaid', 0) < 50:
            recommendations.append("Use active voice instead of passive voice")

        if len(recommendations) == 0:
            recommendations.append("Content readability is good - consider adding visual aids")

        return recommendations

    def simplify_content(self, content: str, target_readability: str = "Standard") -> str:
        """Apply plain language transformations to content"""
        simplified = content

        for rule_name, rule_func in self.plain_language_rules.items():
            simplified = rule_func(simplified)

        return simplified

    def simplify_vocabulary(self, text: str) -> str:
        """Replace complex words with simpler alternatives"""
        # Simple vocabulary simplification (would use NLP in practice)
        replacements = {
            'utilize': 'use',
            'facilitate': 'help',
            'implement': 'do',
            'optimize': 'improve',
            'leverage': 'use',
            'paradigm': 'model',
            'methodology': 'method'
        }

        simplified = text
        for complex_word, simple_word in replacements.items():
            simplified = re.sub(r'\b' + complex_word + r'\b', simple_word, simplified, flags=re.IGNORECASE)

        return simplified

    def shorten_sentences(self, text: str) -> str:
        """Break long sentences into shorter ones"""
        sentences = self.split_into_sentences(text)
        shortened_sentences = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) > 20:
                # Split long sentences at conjunctions
                mid_point = len(words) // 2

                # Find good split point
                split_conjunctions = ['and', 'but', 'or', 'so', 'because', 'although']
                split_point = mid_point

                for i, word in enumerate(words[mid_point-5:mid_point+5]):
                    if word.lower() in split_conjunctions:
                        split_point = mid_point - 5 + i
                        break

                part1 = ' '.join(words[:split_point + 1])
                part2 = ' '.join(words[split_point + 1:])

                shortened_sentences.extend([part1, part2])
            else:
                shortened_sentences.append(sentence)

        return '. '.join(shortened_sentences)

    def convert_to_active_voice(self, text: str) -> str:
        """Convert passive voice to active voice where possible"""
        # Simple pattern-based conversion (would use NLP in practice)
        passive_patterns = [
            (r'(\w+) is (\w+) by (\w+)', r'\3 \2 \1'),
            (r'(\w+) was (\w+) by (\w+)', r'\3 \2 \1'),
            (r'(\w+) are (\w+) by (\w+)', r'\3 \2 \1')
        ]

        converted = text
        for pattern, replacement in passive_patterns:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)

        return converted

    def remove_jargon(self, text: str) -> str:
        """Identify and explain technical jargon"""
        # Simple jargon detection (would use domain-specific dictionaries)
        technical_terms = ['algorithm', 'neural network', 'bayesian', 'inference', 'optimization']

        # Add explanations in brackets
        explained = text
        for term in technical_terms:
            if re.search(r'\b' + term + r'\b', explained, re.IGNORECASE):
                explanation = self.get_jargon_explanation(term)
                explained = re.sub(
                    r'\b(' + term + r')\b',
                    r'\1 [' + explanation + ']',
                    explained,
                    flags=re.IGNORECASE
                )

        return explained

    def get_jargon_explanation(self, term: str) -> str:
        """Get simple explanation for technical term"""
        explanations = {
            'algorithm': 'step-by-step procedure',
            'neural network': 'computer system modeled after the brain',
            'bayesian': 'probability-based reasoning method',
            'inference': 'process of drawing conclusions',
            'optimization': 'process of finding the best solution'
        }

        return explanations.get(term.lower(), f'technical term: {term}')

    def add_examples(self, text: str) -> str:
        """Add examples to explain complex concepts"""
        # Simple example addition (would use NLP to identify complex sections)
        if 'Active Inference' in text and 'example' not in text.lower():
            example_addition = """

For example, when you reach for a cup of coffee, your brain uses Active Inference to predict where the cup will be and how to grasp it, constantly updating based on sensory feedback.

"""
            text += example_addition

        return text

    def structure_content(self, text: str) -> str:
        """Add structure and formatting for better comprehension"""
        # Add headings and lists where appropriate
        structured = text

        # Simple structuring (would use more sophisticated NLP)
        if 'Active Inference' in structured and '## ' not in structured:
            structured = '## What is Active Inference?\n\n' + structured

        return structured

    def create_content_summaries(self, content: str, summary_types: List[str] = None) -> Dict[str, str]:
        """Create multiple types of content summaries for different needs"""
        if summary_types is None:
            summary_types = ['brief', 'detailed', 'simple']

        summaries = {}

        for summary_type in summary_types:
            if summary_type == 'brief':
                summaries['brief'] = self.create_brief_summary(content)
            elif summary_type == 'detailed':
                summaries['detailed'] = self.create_detailed_summary(content)
            elif summary_type == 'simple':
                summaries['simple'] = self.create_simple_summary(content)

        return summaries

    def create_brief_summary(self, content: str) -> str:
        """Create brief, high-level summary"""
        sentences = self.split_into_sentences(content)
        if len(sentences) > 0:
            return sentences[0]  # First sentence as brief summary
        return "Content summary not available"

    def create_detailed_summary(self, content: str) -> str:
        """Create detailed summary with key points"""
        sentences = self.split_into_sentences(content)
        # Take first 3 sentences as detailed summary
        return '. '.join(sentences[:3]) + ('.' if len(sentences) > 3 else '')

    def create_simple_summary(self, content: str) -> str:
        """Create simplified summary for accessibility"""
        # Apply all simplification transformations
        simplified = self.simplify_content(content)
        sentences = self.split_into_sentences(simplified)

        # Take first 2 simplified sentences
        return '. '.join(sentences[:2]) + ('.' if len(sentences) > 2 else '')

class AccessibleContentGenerator:
    """Generate accessible content with multiple formats"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize accessible content generator"""
        self.config = config
        self.logger = logging.getLogger('AccessibleContentGenerator')

        # Content transformation pipelines
        self.transformation_pipelines = {
            'visual_impairment': ['text_alternatives', 'audio_description', 'tactile_graphics'],
            'cognitive_disability': ['plain_language', 'structured_content', 'supplemental_materials'],
            'motor_disability': ['voice_control', 'switch_access', 'eye_tracking'],
            'hearing_impairment': ['captions', 'transcripts', 'sign_language']
        }

    def generate_accessible_formats(self, content: Dict[str, Any], user_needs: List[str]) -> Dict[str, Any]:
        """Generate multiple accessible formats for content"""
        accessible_formats = {}

        for need in user_needs:
            if need in self.transformation_pipelines:
                pipeline = self.transformation_pipelines[need]
                transformed_content = self.apply_transformation_pipeline(content, pipeline)
                accessible_formats[need] = transformed_content

        return accessible_formats

    def apply_transformation_pipeline(self, content: Dict[str, Any], pipeline: List[str]) -> Dict[str, Any]:
        """Apply sequence of content transformations"""
        transformed = content.copy()

        for transformation in pipeline:
            if hasattr(self, f'apply_{transformation}'):
                transform_func = getattr(self, f'apply_{transformation}')
                transformed = transform_func(transformed)

        return transformed

    def apply_text_alternatives(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Add text alternatives for non-text content"""
        if 'images' in content:
            for image in content['images']:
                image['alt_text'] = f"Image: {image.get('description', 'visual content')}"
                image['long_description'] = image.get('detailed_description', image['alt_text'])

        return content

    def apply_audio_description(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Add audio descriptions"""
        if 'visual_elements' in content:
            for element in content['visual_elements']:
                element['audio_description'] = f"Visual element showing: {element.get('description', 'content')}"

        return content

    def apply_plain_language(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to plain language"""
        if 'text_content' in content:
            cognitive_manager = CognitiveAccessibilityManager({})
            content['plain_language_version'] = cognitive_manager.simplify_content(content['text_content'])

        return content

    def apply_structured_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Add structure for better comprehension"""
        if 'text_content' in content:
            content['structured_version'] = self.add_content_structure(content['text_content'])

        return content

    def add_content_structure(self, text: str) -> str:
        """Add headings, lists, and structure to content"""
        # Simple structuring
        structured = text

        # Add main heading if missing
        if not structured.startswith('#'):
            structured = '# Main Content\n\n' + structured

        # Split into sections (simplified)
        sections = structured.split('\n\n')
        structured_sections = []

        for section in sections:
            if len(section.split()) > 50:  # Long sections
                structured_sections.append('## Section\n\n' + section)
            else:
                structured_sections.append(section)

        return '\n\n'.join(structured_sections)

    def validate_accessibility_compliance(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate accessibility compliance of content"""
        validator = AccessibilityValidator({})
        issues = []

        # Check text content
        if 'text_content' in content:
            text_issues = validator.validate_text_accessibility(content['text_content'])
            issues.extend(text_issues)

        # Check images
        if 'images' in content:
            for image in content['images']:
                image_issues = validator.validate_image_accessibility(image)
                issues.extend(image_issues)

        # Check interactive elements
        if 'interactive_elements' in content:
            for element in content['interactive_elements']:
                element_issues = validator.validate_interactive_accessibility(element)
                issues.extend(element_issues)

        compliance_score = max(0, 100 - (len(issues) * 5))  # Deduct 5 points per issue

        return {
            'compliant': len(issues) == 0,
            'compliance_score': compliance_score,
            'issues': issues,
            'recommendations': self.generate_accessibility_recommendations(issues)
        }

    def generate_accessibility_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations to fix accessibility issues"""
        recommendations = []

        issue_types = {}
        for issue in issues:
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

        if 'missing alt text' in issue_types:
            recommendations.append("Add descriptive alt text to all images")

        if 'low contrast' in issue_types:
            recommendations.append("Increase color contrast to meet WCAG standards")

        if 'keyboard navigation' in issue_types:
            recommendations.append("Ensure all interactive elements are keyboard accessible")

        if 'complex language' in issue_types:
            recommendations.append("Simplify language and add plain language alternatives")

        if not recommendations:
            recommendations.append("Content appears accessible - regular audits recommended")

        return recommendations

class TestAccessibilityFramework:
    """Tests for accessibility framework"""

    @pytest.fixture
    def accessibility_validator(self):
        """Create accessibility validator for testing"""
        config = {'wcag_level': 'AA'}
        return AccessibilityValidator(config)

    def test_image_alt_text_validation(self, accessibility_validator):
        """Test image alt text validation"""
        # Test image with good alt text
        good_image = {'src': 'diagram.png', 'alt': 'Flowchart showing data processing steps'}
        issues = accessibility_validator.validate_image_accessibility(good_image)
        assert len(issues) == 0

        # Test image with missing alt text
        bad_image = {'src': 'diagram.png'}
        issues = accessibility_validator.validate_image_accessibility(bad_image)
        assert len(issues) > 0
        assert 'alt text' in issues[0].lower()

    def test_readability_assessment(self):
        """Test readability assessment"""
        cognitive_manager = CognitiveAccessibilityManager({})

        simple_text = "This is a simple sentence. It uses easy words."
        complex_text = "The utilization of sophisticated methodologies facilitates optimization of complex systems through advanced computational frameworks."

        simple_assessment = cognitive_manager.assess_readability(simple_text)
        complex_assessment = cognitive_manager.assess_readability(complex_text)

        assert simple_assessment['average_score'] > complex_assessment['average_score']

    def test_keyboard_navigation(self):
        """Test keyboard navigation functionality"""
        nav_manager = KeyboardNavigationManager({})

        # Register some elements
        nav_manager.register_focusable_element('button1', {'role': 'button', 'aria_label': 'Submit'})
        nav_manager.register_focusable_element('button2', {'role': 'button', 'aria_label': 'Cancel'})

        # Test navigation
        result = nav_manager.navigate_next()
        assert result['success'] == True
        assert result['element_id'] == 'button1'

        result = nav_manager.navigate_next()
        assert result['element_id'] == 'button2'

    def test_plain_language_transformation(self):
        """Test plain language transformations"""
        cognitive_manager = CognitiveAccessibilityManager({})

        complex_text = "The algorithm utilizes sophisticated methodologies."
        simplified = cognitive_manager.simplify_vocabulary(complex_text)

        assert 'utilize' not in simplified.lower()
        assert 'use' in simplified.lower()
```

---

**"Active Inference for, with, by Generative AI"** - Implementing comprehensive accessibility and inclusive design to ensure the Active Inference Knowledge Environment is usable by people with diverse abilities, following WCAG guidelines and universal design principles.
