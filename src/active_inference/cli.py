"""
Command Line Interface for Active Inference Knowledge Environment

Provides unified access to all platform features through a clean, intuitive CLI.
Supports learning, research, visualization, and development workflows.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .knowledge import KnowledgeRepository, KnowledgeRepositoryConfig
from .visualization import VisualizationEngine
from .research import ResearchFramework
from .applications import ApplicationFramework
from .llm import (
    OllamaClient,
    LLMConfig,
    PromptManager,
    ModelManager,
    ConversationManager,
    ActiveInferencePromptBuilder,
    ActiveInferenceTemplates
)


class ActiveInferenceCLI:
    """Command line interface for the Active Inference Knowledge Environment"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.knowledge_repo = None
        self.visualization_engine = None
        self.research_framework = None
        self.application_framework = None
        self.llm_client = None
        self.prompt_manager = None
        self.model_manager = None
        self.conversation_manager = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all platform components"""
        try:
            # Initialize knowledge repository
            config = KnowledgeRepositoryConfig(
                root_path=self.project_root / "knowledge",
                auto_index=True
            )
            self.knowledge_repo = KnowledgeRepository(config)

            # Initialize LLM components
            llm_config = LLMConfig()
            self.llm_client = OllamaClient(llm_config)
            self.prompt_manager = PromptManager()
            self.model_manager = ModelManager()
            self.conversation_manager = ConversationManager()

            # Initialize other components (placeholder for now)
            # self.visualization_engine = VisualizationEngine()
            # self.research_framework = ResearchFramework()
            # self.application_framework = ApplicationFramework()

        except Exception as e:
            print(f"Warning: Could not initialize all components: {e}")
            print("Some features may not be available.")

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser"""
        parser = argparse.ArgumentParser(
            description="Active Inference Knowledge Environment",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  ai-knowledge learn foundations          # Start foundations learning track
  ai-knowledge search "entropy"           # Search knowledge base
  ai-knowledge path show complete        # Show complete learning path
  ai-knowledge stats                     # Show repository statistics

  ai-knowledge llm chat "Explain Active Inference"  # Chat with AI
  ai-knowledge llm chat --template explanation    # Use explanation template
  ai-knowledge llm generate "What is FEP?" --template active_inference_explanation  # Generate explanation
  ai-knowledge llm models list           # List available models
  ai-knowledge llm models pull gemma3:2b # Pull model from registry
  ai-knowledge llm conversation list     # List saved conversations
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Knowledge commands
        knowledge_parser = subparsers.add_parser('knowledge', help='Knowledge repository operations')
        knowledge_subparsers = knowledge_parser.add_subparsers(dest='knowledge_command')

        # Learning commands
        learn_parser = knowledge_subparsers.add_parser('learn', help='Start learning')
        learn_parser.add_argument('track', choices=['foundations', 'research', 'applications', 'complete'],
                                help='Learning track to follow')

        # Search command
        search_parser = knowledge_subparsers.add_parser('search', help='Search knowledge base')
        search_parser.add_argument('query', help='Search query')
        search_parser.add_argument('--type', choices=['foundation', 'mathematics', 'implementation', 'application'],
                                 help='Filter by content type')
        search_parser.add_argument('--limit', type=int, default=10, help='Maximum results')

        # Path commands
        path_parser = knowledge_subparsers.add_parser('path', help='Learning path operations')
        path_subparsers = path_parser.add_subparsers(dest='path_command')

        path_show_parser = path_subparsers.add_parser('show', help='Show learning path')
        path_show_parser.add_argument('path_id', help='Path identifier')

        path_list_parser = path_subparsers.add_parser('list', help='List available paths')

        # Statistics command
        knowledge_subparsers.add_parser('stats', help='Show repository statistics')

        # Research commands
        research_parser = subparsers.add_parser('research', help='Research tools and experiments')
        research_subparsers = research_parser.add_subparsers(dest='research_command')

        research_subparsers.add_parser('experiments', help='Run experiments')
        research_subparsers.add_parser('simulations', help='Run simulations')
        research_subparsers.add_parser('analyze', help='Analyze results')

        # Visualization commands
        viz_parser = subparsers.add_parser('visualize', help='Visualization tools')
        viz_subparsers = viz_parser.add_subparsers(dest='viz_command')

        viz_subparsers.add_parser('concepts', help='Visualize concepts')
        viz_subparsers.add_parser('models', help='Visualize models')
        viz_subparsers.add_parser('dashboard', help='Start visualization dashboard')

        # Application commands
        app_parser = subparsers.add_parser('applications', help='Application framework')
        app_subparsers = app_parser.add_subparsers(dest='app_command')

        app_subparsers.add_parser('templates', help='Generate application templates')
        app_subparsers.add_parser('examples', help='Run example applications')

        # LLM commands
        llm_parser = subparsers.add_parser('llm', help='LLM and AI assistant operations')
        llm_subparsers = llm_parser.add_subparsers(dest='llm_command')

        # Chat command
        chat_parser = llm_subparsers.add_parser('chat', help='Chat with AI assistant')
        chat_parser.add_argument('message', nargs='?', help='Message to send (if not provided, starts interactive chat)')
        chat_parser.add_argument('--model', help='Model to use for chat')
        chat_parser.add_argument('--template', choices=['explanation', 'research', 'implementation', 'education'],
                               help='Use conversation template')

        # Generate command
        generate_parser = llm_subparsers.add_parser('generate', help='Generate text from prompt')
        generate_parser.add_argument('prompt', help='Prompt for text generation')
        generate_parser.add_argument('--model', help='Model to use')
        generate_parser.add_argument('--template', choices=['active_inference_explanation', 'research_question', 'code_implementation'],
                                   help='Use prompt template')
        generate_parser.add_argument('--concept', help='Concept for explanation template')
        generate_parser.add_argument('--audience-level', choices=['beginner', 'intermediate', 'advanced', 'expert'],
                                   default='intermediate', help='Audience level for explanations')

        # Model management commands
        model_parser = llm_subparsers.add_parser('models', help='Model management')
        model_subparsers = model_parser.add_subparsers(dest='model_command')

        model_subparsers.add_parser('list', help='List available models')
        model_subparsers.add_parser('info', help='Show model information').add_argument('model', help='Model name')

        pull_parser = model_subparsers.add_parser('pull', help='Pull model from registry')
        pull_parser.add_argument('model', help='Model to pull')

        # Conversation management
        conv_parser = llm_subparsers.add_parser('conversation', help='Conversation management')
        conv_subparsers = conv_parser.add_subparsers(dest='conv_command')

        conv_subparsers.add_parser('list', help='List conversations')
        conv_subparsers.add_parser('create', help='Create new conversation').add_argument('title', help='Conversation title')
        conv_subparsers.add_parser('delete', help='Delete conversation').add_argument('conversation_id', help='Conversation ID')
        conv_subparsers.add_parser('export', help='Export conversation').add_argument('conversation_id', help='Conversation ID')

        # Platform commands
        platform_parser = subparsers.add_parser('platform', help='Platform operations')
        platform_subparsers = platform_parser.add_subparsers(dest='platform_command')

        platform_subparsers.add_parser('serve', help='Start platform server')
        platform_subparsers.add_parser('status', help='Show platform status')

        return parser

    def run(self, args: Optional[list] = None) -> int:
        """Run the CLI with given arguments"""
        parser = self.create_parser()

        if args is None:
            args = sys.argv[1:]

        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return 0

        try:
            return self._dispatch_command(parsed_args)
        except Exception as e:
            print(f"Error: {e}")
            return 1

    def _dispatch_command(self, args) -> int:
        """Dispatch command to appropriate handler"""
        if args.command == 'knowledge':
            return self._handle_knowledge_command(args)
        elif args.command == 'research':
            return self._handle_research_command(args)
        elif args.command == 'visualize':
            return self._handle_visualization_command(args)
        elif args.command == 'applications':
            return self._handle_applications_command(args)
        elif args.command == 'llm':
            return self._handle_llm_command(args)
        elif args.command == 'platform':
            return self._handle_platform_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    def _handle_knowledge_command(self, args) -> int:
        """Handle knowledge repository commands"""
        if not self.knowledge_repo:
            print("Knowledge repository not available")
            return 1

        if args.knowledge_command == 'learn':
            return self._handle_learn_command(args)
        elif args.knowledge_command == 'search':
            return self._handle_search_command(args)
        elif args.knowledge_command == 'path':
            return self._handle_path_command(args)
        elif args.knowledge_command == 'stats':
            return self._handle_stats_command(args)
        else:
            print(f"Unknown knowledge command: {args.knowledge_command}")
            return 1

    def _handle_learn_command(self, args) -> int:
        """Handle learning track commands"""
        if args.track == 'foundations':
            print("🚀 Starting Foundations Learning Track")
            print("This track covers the theoretical foundations of Active Inference")
            print()

            if self.knowledge_repo:
                # Get foundation learning path
                path = self.knowledge_repo.get_learning_path('foundations_complete')
                if path:
                    print(f"📚 Learning Path: {path.name}")
                    print(f"📖 Description: {path.description}")
                    print(f"⏱️  Estimated time: {path.estimated_hours} hours")
                    print(f"🎯 Difficulty: {path.difficulty.value}")
                    print()

                    # Show path nodes
                    print("📋 Learning Modules:")
                    for i, node_id in enumerate(path.nodes, 1):
                        node = self.knowledge_repo.get_node(node_id)
                        if node:
                            print(f"  {i"2d"}. {node.title}")
                            print(f"      {node.description}")
                            print(f"      Type: {node.content_type.value} | Difficulty: {node.difficulty.value}")
                            print()

                    print("💡 To start learning, use: ai-knowledge search <topic>")
                    print("📖 For detailed content, explore individual modules")
                else:
                    print("Foundation learning path not found")

        elif args.track == 'complete':
            print("🎓 Starting Complete Learning Track")
            print("Comprehensive journey through all Active Inference concepts")
            # Similar implementation for complete track

        else:
            print(f"Learning track '{args.track}' not implemented yet")

        return 0

    def _handle_search_command(self, args) -> int:
        """Handle search commands"""
        if not self.knowledge_repo:
            print("Knowledge repository not available")
            return 1

        print(f"🔍 Searching for: '{args.query}'")
        print()

        # Prepare filters
        content_types = []
        if hasattr(args, 'type') and args.type:
            # Map string to enum (placeholder)
            content_types = [args.type]

        # Perform search
        results = self.knowledge_repo.search(
            query=args.query,
            limit=args.limit
        )

        if not results:
            print("No results found.")
            print("Try adjusting your search terms or filters.")
            return 0

        print(f"📚 Found {len(results)} results:")
        print()

        for i, node in enumerate(results, 1):
            print(f"{i}. {node.title}")
            print(f"   Type: {node.content_type.value} | Difficulty: {node.difficulty.value}")
            print(f"   Description: {node.description}")
            if node.tags:
                print(f"   Tags: {', '.join(node.tags)}")
            print()

        return 0

    def _handle_path_command(self, args) -> int:
        """Handle learning path commands"""
        if not self.knowledge_repo:
            print("Knowledge repository not available")
            return 1

        if args.path_command == 'show':
            path = self.knowledge_repo.get_learning_path(args.path_id)
            if not path:
                print(f"Learning path '{args.path_id}' not found")
                return 1

            print(f"📚 Learning Path: {path.name}")
            print(f"📖 {path.description}")
            print(f"⏱️  Estimated time: {path.estimated_hours} hours")
            print(f"🎯 Difficulty: {path.difficulty.value}")
            print()

            # Validate path
            validation = self.knowledge_repo.validate_learning_path(args.path_id)
            if validation['valid']:
                print("✅ Path validation: PASSED")
            else:
                print("⚠️  Path validation: ISSUES FOUND")
                for issue in validation['issues']:
                    print(f"   - {issue}")

            print()
            print("📋 Path Modules:")
            for i, node_id in enumerate(path.nodes, 1):
                node = self.knowledge_repo.get_node(node_id)
                if node:
                    print(f"  {i"2d"}. {node.title}")
                    print(f"      {node.description}")
                    print(f"      Prerequisites: {', '.join(node.prerequisites) if node.prerequisites else 'None'}")
                    print()

        elif args.path_command == 'list':
            paths = self.knowledge_repo.get_learning_paths()
            print("📚 Available Learning Paths:")
            print()

            for path in paths:
                print(f"• {path.name}")
                print(f"  {path.description}")
                print(f"  Duration: {path.estimated_hours}h | Difficulty: {path.difficulty.value}")
                print()

        return 0

    def _handle_stats_command(self, args) -> int:
        """Handle statistics command"""
        if not self.knowledge_repo:
            print("Knowledge repository not available")
            return 1

        stats = self.knowledge_repo.get_statistics()

        print("📊 Knowledge Repository Statistics")
        print("=" * 40)
        print()

        print("📈 Overview:")
        print(f"   Total knowledge nodes: {stats['total_nodes']}")
        print(f"   Total learning paths: {stats['total_paths']}")
        print()

        print("🏷️  Content Types:")
        for content_type, count in stats['content_types'].items():
            print(f"   {content_type}: {count}")
        print()

        print("📊 Difficulty Distribution:")
        for difficulty, count in stats['difficulties'].items():
            print(f"   {difficulty}: {count}")
        print()

        if stats['top_tags']:
            print("🏷️  Popular Tags:")
            for tag, count in stats['top_tags'][:10]:
                print(f"   {tag}: {count}")
            if len(stats['top_tags']) > 10:
                print(f"   ... and {len(stats['top_tags']) - 10} more")
            print()

        return 0

    def _handle_research_command(self, args) -> int:
        """Handle research commands (placeholder)"""
        print("🔬 Research tools coming soon!")
        print("This will include experiment management, simulation tools, and analysis frameworks.")
        return 0

    def _handle_visualization_command(self, args) -> int:
        """Handle visualization commands (placeholder)"""
        print("👁️  Visualization tools coming soon!")
        print("This will include interactive diagrams, concept maps, and simulation dashboards.")
        return 0

    def _handle_applications_command(self, args) -> int:
        """Handle application commands (placeholder)"""
        print("🛠️  Application framework coming soon!")
        print("This will include templates, case studies, and integration tools.")
        return 0

    def _handle_platform_command(self, args) -> int:
        """Handle platform commands (placeholder)"""
        print("🚀 Platform tools coming soon!")
        print("This will include server management, monitoring, and deployment tools.")
        return 0

    def _handle_llm_command(self, args) -> int:
        """Handle LLM commands"""
        if not self.llm_client or not self.prompt_manager:
            print("LLM components not available. Please ensure Ollama is installed and running.")
            print("Run: ollama serve")
            return 1

        if args.llm_command == 'chat':
            return self._handle_chat_command(args)
        elif args.llm_command == 'generate':
            return self._handle_generate_command(args)
        elif args.llm_command == 'models':
            return self._handle_models_command(args)
        elif args.llm_command == 'conversation':
            return self._handle_conversation_command(args)
        else:
            print(f"Unknown LLM command: {args.llm_command}")
            return 1

    def _handle_chat_command(self, args) -> int:
        """Handle chat command"""
        import asyncio

        async def chat():
            try:
                # Initialize client if needed
                if not self.llm_client._is_initialized:
                    await self.llm_client.initialize()

                model = args.model or "gemma3:2b"

                # Use template if specified
                if args.template:
                    if args.template == 'explanation':
                        template = ActiveInferenceTemplates.get_explanation_template()
                    elif args.template == 'research':
                        template = ActiveInferenceTemplates.get_research_template()
                    elif args.template == 'implementation':
                        template = ActiveInferenceTemplates.get_implementation_template()
                    elif args.template == 'education':
                        template = ActiveInferenceTemplates.get_education_template()

                    conversation = template.create_conversation(f"AI Chat - {args.template.title()}", self.conversation_manager)
                    print(f"💬 Started conversation with {args.template} template")
                else:
                    # Create simple conversation
                    conversation = self.conversation_manager.create_conversation("AI Chat")
                    print(f"💬 Started new conversation")

                if args.message:
                    # Single message chat
                    print(f"🤖 Generating response for: {args.message}")
                    print()

                    response = await self.llm_client.generate(args.message, model=model)
                    print(response)

                    # Save to conversation
                    self.conversation_manager.add_message(conversation.id, "user", args.message)
                    self.conversation_manager.add_message(conversation.id, "assistant", response)

                else:
                    # Interactive chat
                    print(f"💬 Interactive chat with {model}")
                    print("Type 'quit' or 'exit' to end conversation")
                    print()

                    while True:
                        try:
                            user_input = input("You: ").strip()
                            if user_input.lower() in ['quit', 'exit', 'bye']:
                                print("👋 Goodbye!")
                                break

                            if not user_input:
                                continue

                            print("🤖 Thinking...")
                            response = await self.llm_client.generate(user_input, model=model)
                            print(f"Assistant: {response}")
                            print()

                            # Save to conversation
                            self.conversation_manager.add_message(conversation.id, "user", user_input)
                            self.conversation_manager.add_message(conversation.id, "assistant", response)

                        except KeyboardInterrupt:
                            print("\n👋 Chat interrupted. Goodbye!")
                            break
                        except Exception as e:
                            print(f"❌ Error: {e}")

            except Exception as e:
                print(f"❌ Chat failed: {e}")
                return 1

        asyncio.run(chat())
        return 0

    def _handle_generate_command(self, args) -> int:
        """Handle generate command"""
        import asyncio

        async def generate():
            try:
                # Initialize client if needed
                if not self.llm_client._is_initialized:
                    await self.llm_client.initialize()

                model = args.model or "gemma3:2b"
                prompt = args.prompt

                # Use template if specified
                if args.template:
                    if args.template == 'active_inference_explanation':
                        variables = {
                            "concept": args.concept or "Active Inference",
                            "context": "general explanation",
                            "audience_level": args.audience_level,
                            "key_points": "core concepts, mathematical formulation, applications",
                            "response_type": "comprehensive explanation"
                        }
                        prompt = self.prompt_manager.generate_prompt(args.template, variables)

                    elif args.template == 'research_question':
                        variables = {
                            "topic": args.concept or "Active Inference",
                            "domain": "artificial intelligence",
                            "current_knowledge": "established foundations",
                            "research_gap": "practical applications",
                            "methodology": "theoretical and computational",
                            "num_questions": "5"
                        }
                        prompt = self.prompt_manager.generate_prompt(args.template, variables)

                    elif args.template == 'code_implementation':
                        variables = {
                            "algorithm": args.concept or "Active Inference",
                            "language": "Python",
                            "problem_description": "Implementation request",
                            "requirements": "comprehensive, well-documented",
                            "constraints": "educational, practical",
                            "codebase_context": "Active Inference platform"
                        }
                        prompt = self.prompt_manager.generate_prompt(args.template, variables)

                print(f"🤖 Generating with model: {model}")
                print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                print()

                response = await self.llm_client.generate(prompt, model=model)
                print(response)

            except Exception as e:
                print(f"❌ Generation failed: {e}")
                return 1

        asyncio.run(generate())
        return 0

    def _handle_models_command(self, args) -> int:
        """Handle model management commands"""
        import asyncio

        async def handle_models():
            try:
                if not self.llm_client._is_initialized:
                    await self.llm_client.initialize()

                if args.model_command == 'list':
                    models = await self.llm_client.list_models()
                    if models:
                        print("🧠 Available Models:")
                        print()
                        for model in models:
                            print(f"• {model}")
                    else:
                        print("No models available.")
                        print("Pull a model with: ai-knowledge llm models pull <model_name>")
                        print("Recommended: gemma3:2b, gemma3:4b")

                elif args.model_command == 'info':
                    info = await self.llm_client.get_model_info(args.model)
                    if info:
                        print(f"🧠 Model Information: {args.model}")
                        print("-" * 40)
                        for key, value in info.items():
                            print(f"{key}: {value}")
                    else:
                        print(f"Model {args.model} not found")

                elif args.model_command == 'pull':
                    print(f"📥 Pulling model: {args.model}")
                    success = await self.llm_client.pull_model(args.model)
                    if success:
                        print(f"✅ Successfully pulled {args.model}")
                    else:
                        print(f"❌ Failed to pull {args.model}")
                        return 1

            except Exception as e:
                print(f"❌ Model command failed: {e}")
                return 1

        asyncio.run(handle_models())
        return 0

    def _handle_conversation_command(self, args) -> int:
        """Handle conversation management commands"""
        if args.conv_command == 'list':
            conversations = self.conversation_manager.list_conversations()
            if conversations:
                print("💬 Conversations:")
                print()
                for conv in conversations:
                    print(f"• {conv.title} (ID: {conv.id[:8]}...)")
                    print(f"  Messages: {len(conv.messages)} | Updated: {conv.updated_at.strftime('%Y-%m-%d %H:%M')}")
                    if conv.metadata:
                        print(f"  Type: {conv.metadata.get('type', 'general')}")
                    print()
            else:
                print("No conversations found.")
                print("Create one with: ai-knowledge llm conversation create 'My Chat'")

        elif args.conv_command == 'create':
            conversation = self.conversation_manager.create_conversation(args.title)
            print(f"✅ Created conversation: {conversation.title}")
            print(f"ID: {conversation.id}")

        elif args.conv_command == 'delete':
            success = self.conversation_manager.delete_conversation(args.conversation_id)
            if success:
                print(f"🗑️  Deleted conversation: {args.conversation_id[:8]}...")
            else:
                print(f"❌ Conversation not found: {args.conversation_id}")

        elif args.conv_command == 'export':
            export_data = self.conversation_manager.export_conversation(args.conversation_id, "markdown")
            if export_data:
                print("📄 Conversation Export:")
                print("=" * 50)
                print(export_data)
            else:
                print(f"❌ Conversation not found: {args.conversation_id}")

        return 0


def main():
    """Main entry point for the CLI"""
    cli = ActiveInferenceCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
