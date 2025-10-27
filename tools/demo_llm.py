#!/usr/bin/env python3
"""
Comprehensive LLM Module Demo

Demonstrates all features of the Active Inference LLM module with Ollama integration.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from active_inference.llm import (
    OllamaClient,
    LLMConfig,
    PromptManager,
    ModelManager,
    ConversationManager,
    ActiveInferenceTemplates,
    ActiveInferencePromptBuilder,
)


async def main():
    """Run comprehensive LLM module demonstration"""
    
    print("🧠 Active Inference LLM Module Demo")
    print("=" * 70)
    
    # =========================================================================
    # 1. Basic LLM Client Demo
    # =========================================================================
    print("\n📡 1. LLM Client Initialization")
    print("-" * 70)
    
    config = LLMConfig(
        default_model='gemma2:2b',
        temperature=0.7,
        max_tokens=150,
        debug=False
    )
    
    async with OllamaClient(config) as client:
        print(f"✅ Client initialized with model: {config.default_model}")
        
        # Health check
        health = await client.health_check()
        print(f"✅ Service status: {health['status']}")
        print(f"   Ollama version: {health.get('version', 'unknown')}")
        print(f"   Available models: {len(health['available_models'])}")
        
        # =====================================================================
        # 2. Simple Text Generation
        # =====================================================================
        print("\n💬 2. Simple Text Generation")
        print("-" * 70)
        
        prompt = "Explain the Free Energy Principle in one sentence."
        print(f"Prompt: {prompt}")
        response = await client.generate(prompt, max_tokens=50)
        print(f"✅ Response: {response}")
        
        # =====================================================================
        # 3. Chat with Context
        # =====================================================================
        print("\n🗨️  3. Chat with Message History")
        print("-" * 70)
        
        messages = [
            {'role': 'user', 'content': 'What is active inference?'},
            {'role': 'assistant', 'content': 'Active inference is a framework for understanding perception and action based on minimizing prediction errors.'},
            {'role': 'user', 'content': 'How does it relate to machine learning?'}
        ]
        
        print("Conversation:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content'][:60]}...")
        
        chat_response = await client.chat(messages, max_tokens=100)
        print(f"\n✅ Response: {chat_response[:200]}...")
        
        # =====================================================================
        # 4. Model Management
        # =====================================================================
        print("\n\n🤖 4. Model Management")
        print("-" * 70)
        
        model_manager = ModelManager()
        print(f"Registered models: {len(model_manager.get_available_models())}")
        for model_name in model_manager.get_available_models():
            print(f"  - {model_name}")
        
        # Find best model for task
        best_model = model_manager.find_best_model('active inference explanation')
        if best_model:
            print(f"\n✅ Best model for Active Inference: {best_model.name}")
            print(f"   Family: {best_model.family}")
            print(f"   Context window: {best_model.context_window}")
            print(f"   Memory estimate: {best_model.get_memory_requirement():.1f} GB")
        
        # Models by capability
        math_models = model_manager.get_models_by_capability('mathematical_reasoning')
        print(f"\n✅ Models with mathematical reasoning ({len(math_models)}):")
        for model in math_models[:3]:
            print(f"   - {model.name}")
        
        # =====================================================================
        # 5. Prompt Templates
        # =====================================================================
        print("\n\n🎨 5. Prompt Template System")
        print("-" * 70)
        
        prompt_manager = PromptManager()
        print(f"Available templates: {', '.join(prompt_manager.list_templates())}")
        
        # Use Active Inference explanation template
        print("\n✅ Using Active Inference explanation template:")
        ai_prompt = prompt_manager.generate_prompt('active_inference_explanation', {
            'concept': 'variational free energy',
            'context': 'computational neuroscience',
            'audience_level': 'intermediate',
            'key_points': 'mathematical formulation, prediction errors, action selection',
            'response_type': 'clear explanation with examples'
        })
        
        print(f"   Generated prompt ({len(ai_prompt)} chars)")
        print(f"   First 150 chars: {ai_prompt[:150]}...")
        
        # Use with LLM
        print("\n   Sending to LLM...")
        template_response = await client.generate(ai_prompt, max_tokens=150)
        print(f"   ✅ Response: {template_response[:200]}...")
        
        # =====================================================================
        # 6. Prompt Builders
        # =====================================================================
        print("\n\n🏗️  6. Prompt Builder Pattern")
        print("-" * 70)
        
        builder = ActiveInferencePromptBuilder()
        complex_prompt = (builder
            .add_concept_explanation('expected free energy', 'policy selection', 'advanced')
            .set_variable('include_math', 'true')
            .set_variable('include_code', 'true')
            .build())
        
        print(f"✅ Built complex prompt ({len(complex_prompt)} chars)")
        print(f"   Content: {complex_prompt[:150]}...")
        
        # =====================================================================
        # 7. Conversation Management
        # =====================================================================
        print("\n\n💬 7. Conversation Management")
        print("-" * 70)
        
        conv_manager = ConversationManager()
        
        # Create conversation
        conversation = conv_manager.create_conversation("Learning Active Inference")
        print(f"✅ Created conversation: {conversation.title}")
        print(f"   ID: {conversation.id}")
        
        # Add messages
        conv_manager.add_message(conversation.id, 'user', 'Explain the free energy principle')
        conv_manager.add_message(conversation.id, 'assistant', 'The Free Energy Principle states...')
        conv_manager.add_message(conversation.id, 'user', 'How does this apply to perception?')
        
        print(f"   Messages: {len(conversation.messages)}")
        
        # Get context and continue conversation
        context = conv_manager.get_conversation_context(conversation.id)
        print(f"   Context messages: {len(context)}")
        
        print("\n   Continuing conversation with LLM...")
        response = await client.chat(context, max_tokens=100)
        conv_manager.add_message(conversation.id, 'assistant', response)
        print(f"   ✅ Added response: {response[:100]}...")
        
        # Export conversation
        json_export = conv_manager.export_conversation(conversation.id, 'json')
        markdown_export = conv_manager.export_conversation(conversation.id, 'markdown')
        print(f"\n   ✅ Exported conversation:")
        print(f"      JSON: {len(json_export)} chars")
        print(f"      Markdown: {len(markdown_export)} chars")
        
        # Search conversations
        results = conv_manager.search_conversations('perception')
        print(f"   ✅ Search for 'perception': {len(results)} results")
        
        # =====================================================================
        # 8. Conversation Templates
        # =====================================================================
        print("\n\n📋 8. Conversation Templates")
        print("-" * 70)
        
        # Use explanation template
        explanation_template = ActiveInferenceTemplates.get_explanation_template()
        print(f"✅ Explanation template: {explanation_template.name}")
        print(f"   System prompt: {explanation_template.system_prompt[:100]}...")
        
        # Create conversation from template
        conv = explanation_template.create_conversation("FEP Discussion", conv_manager)
        print(f"   ✅ Created templated conversation: {conv.title}")
        
        # List all templates
        print("\n✅ Available conversation templates:")
        for template_getter in [
            ActiveInferenceTemplates.get_explanation_template,
            ActiveInferenceTemplates.get_research_template,
            ActiveInferenceTemplates.get_implementation_template,
            ActiveInferenceTemplates.get_education_template
        ]:
            template = template_getter()
            print(f"   - {template.name}: {template.metadata.get('type', 'general')}")
        
        # =====================================================================
        # 9. Real-World Example: Generate Tutorial
        # =====================================================================
        print("\n\n📚 9. Real-World Example: Generate Tutorial")
        print("-" * 70)
        
        # Build tutorial generation prompt
        tutorial_prompt = prompt_manager.generate_prompt('active_inference_explanation', {
            'concept': 'active inference basics',
            'context': 'beginner tutorial for neuroscience students',
            'audience_level': 'beginner',
            'key_points': 'core concepts, biological examples, simple mathematics',
            'response_type': 'step-by-step tutorial'
        })
        
        print("Generating tutorial content...")
        tutorial = await client.generate(tutorial_prompt, max_tokens=200)
        print(f"✅ Generated tutorial ({len(tutorial)} chars)")
        print(f"\nTutorial preview:")
        print("-" * 70)
        print(tutorial[:400])
        print("...")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print("\n\n" + "=" * 70)
        print("✅ Demo Complete!")
        print("=" * 70)
        print("\n📊 Summary:")
        print(f"   • LLM Client: ✅ Working")
        print(f"   • Model Management: ✅ {len(model_manager.get_available_models())} models")
        print(f"   • Prompt Templates: ✅ {len(prompt_manager.list_templates())} templates")
        print(f"   • Conversations: ✅ {len(conv_manager.list_conversations())} active")
        print(f"   • Ollama Models: ✅ {len(health['available_models'])} available")
        
        print("\n🎉 All LLM module features working successfully!")
        print("\n💡 Next steps:")
        print("   • Explore more templates in PromptManager")
        print("   • Create custom conversation templates")
        print("   • Integrate with knowledge repository")
        print("   • Try different models for various tasks")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

