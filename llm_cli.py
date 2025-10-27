#!/usr/bin/env python3
"""
Simple CLI for Active Inference LLM Module

Quick command-line interface for interacting with local LLMs via Ollama.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from active_inference.llm import OllamaClient, LLMConfig, PromptManager


async def main():
    """Simple interactive LLM CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Active Inference LLM CLI')
    parser.add_argument('prompt', nargs='?', help='Prompt to send to LLM')
    parser.add_argument('--model', default='gemma2:2b', help='Model to use (default: gemma2:2b)')
    parser.add_argument('--max-tokens', type=int, default=200, help='Maximum tokens to generate')
    parser.add_argument('--template', choices=['explanation', 'research', 'code'], help='Use prompt template')
    parser.add_argument('--chat', action='store_true', help='Start interactive chat session')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    config = LLMConfig(
        default_model=args.model,
        max_tokens=args.max_tokens,
        debug=False
    )
    
    async with OllamaClient(config) as client:
        # List models
        if args.list_models:
            print("📋 Available models:")
            models = await client.list_models()
            for model in models:
                print(f"   • {model}")
            return
        
        # Interactive chat
        if args.chat:
            print(f"💬 Starting chat with {args.model}")
            print("   Type 'exit' or 'quit' to end the session")
            print("-" * 50)
            
            messages = []
            while True:
                try:
                    user_input = input("\n👤 You: ")
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    messages.append({'role': 'user', 'content': user_input})
                    
                    print(f"🤖 {args.model}: ", end='', flush=True)
                    response = await client.chat(messages, max_tokens=args.max_tokens)
                    print(response)
                    
                    messages.append({'role': 'assistant', 'content': response})
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
            return
        
        # Template-based generation
        if args.template and args.prompt:
            prompt_manager = PromptManager()
            
            if args.template == 'explanation':
                prompt = prompt_manager.generate_prompt('active_inference_explanation', {
                    'concept': args.prompt,
                    'context': 'general',
                    'audience_level': 'intermediate',
                    'key_points': 'definition, examples, applications',
                    'response_type': 'clear explanation'
                })
            elif args.template == 'research':
                prompt = prompt_manager.generate_prompt('research_question', {
                    'topic': args.prompt,
                    'domain': 'cognitive science',
                    'current_knowledge': 'basic understanding',
                    'research_gap': 'to be determined',
                    'methodology': 'computational modeling',
                    'num_questions': '3'
                })
            elif args.template == 'code':
                prompt = prompt_manager.generate_prompt('code_implementation', {
                    'algorithm': args.prompt,
                    'language': 'Python',
                    'problem_description': 'implementation needed',
                    'requirements': 'clean, documented code',
                    'constraints': 'none',
                    'codebase_context': 'Active Inference framework'
                })
            
            response = await client.generate(prompt, max_tokens=args.max_tokens * 2)
            print(response)
            return
        
        # Simple generation
        if args.prompt:
            response = await client.generate(args.prompt, max_tokens=args.max_tokens)
            print(response)
            return
        
        # No arguments - show help
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

