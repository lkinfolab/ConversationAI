"""
LangChain-based Gemini LLM service with conversation memory management.
Handles LLM interactions with proper memory context and token tracking.
"""

import logging
from typing import Optional
from django.conf import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from core.exceptions import LLMAPIError

logger = logging.getLogger('llm')

# Store conversation chains in memory (in production, use Redis)
_memory_store = {}


def initialize_llm_client():
    """Initialize Google Gemini LLM client."""
    api_key = settings.GEMINI_API_KEY
    if api_key == 'your-gemini-api-key-here':
        logger.warning("Gemini API key not configured. Using mock responses.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=api_key,
            temperature=settings.LANGCHAIN_SETTINGS.get('temperature', 0.7),
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        raise LLMAPIError(f"Failed to initialize LLM: {str(e)}")


def get_conversation_memory(conversation_id: int):
    """Get or create conversation memory for LangChain."""
    if conversation_id not in _memory_store:
        memory_type = settings.LANGCHAIN_SETTINGS.get('conversation_memory_type', 'buffer')

        if memory_type == 'summary':
            llm = initialize_llm_client()
            memory = ConversationSummaryMemory(
                llm=llm,
                max_token_limit=1000,
                memory_key="chat_history",
                human_prefix="User",
                ai_prefix="Assistant"
            )
        else:  # buffer is default
            k = settings.LANGCHAIN_SETTINGS.get('conversation_memory_k', 10)
            memory = ConversationBufferMemory(
                k=k,
                memory_key="chat_history",
                human_prefix="User",
                ai_prefix="Assistant"
            )

        _memory_store[conversation_id] = memory

    return _memory_store[conversation_id]


def load_conversation_history(conversation_id: int):
    """Load existing conversation messages into memory."""
    try:
        from apps.conversations.models import Message

        messages = Message.objects.filter(
            conversation_id=conversation_id
        ).order_by('timestamp')

        memory = get_conversation_memory(conversation_id)

        # Load previous messages into memory
        for message in messages:
            if message.role == 'user':
                memory.chat_memory.add_user_message(message.content)
            elif message.role == 'assistant':
                memory.chat_memory.add_ai_message(message.content)

        logger.debug(f"Loaded {messages.count()} messages for conversation {conversation_id}")

    except Exception as e:
        logger.error(f"Failed to load conversation history: {str(e)}")


def create_conversation_chain(llm):
    """Create LangChain conversation chain with custom prompt."""
    template = """You are a helpful AI assistant. Provide thoughtful, accurate, and concise responses.

Previous conversation:
{chat_history}

User: {input}
Assistant:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template=template
    )

    return ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=None,  # Will be set dynamically
        verbose=settings.LANGCHAIN_SETTINGS.get('verbose', False)
    )


def generate_llm_response(
    conversation_id: int,
    user_message: str,
    conversation_obj=None
) -> str:
    """
    Generate LLM response using LangChain with conversation memory.

    Args:
        conversation_id: ID of the conversation
        user_message: User's input message
        conversation_obj: Optional conversation model instance

    Returns:
        LLM generated response

    Raises:
        LLMAPIError: If LLM call fails
    """
    try:
        # Initialize LLM client
        llm = initialize_llm_client()

        # Check if using mock responses (for development without API key)
        if llm is None:
            return _generate_mock_response(user_message)

        # Load conversation history
        load_conversation_history(conversation_id)

        # Get or create memory for this conversation
        memory = get_conversation_memory(conversation_id)

        # Create conversation chain
        chain = create_conversation_chain(llm)
        chain.memory = memory

        # Generate response
        logger.debug(f"Generating LLM response for conversation {conversation_id}")
        response = chain.predict(input=user_message)

        # Log token usage
        if conversation_obj:
            _log_token_usage(conversation_id, conversation_obj, user_message, response)

        logger.info(f"Successfully generated response for conversation {conversation_id}")

        return response.strip()

    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise LLMAPIError(f"Failed to generate LLM response: {str(e)}")


def _generate_mock_response(user_message: str) -> str:
    """Generate a mock response for development/testing without API key."""
    mock_responses = {
        'hello': 'Hello! I\'m an AI assistant. How can I help you today?',
        'help': 'I can assist you with various topics. Feel free to ask me anything!',
        'what': 'I can provide information on a wide range of subjects. What would you like to know?',
    }

    # Check first word
    first_word = user_message.lower().split()[0] if user_message else ''

    if first_word in mock_responses:
        return mock_responses[first_word]

    return f"Thank you for your message: '{user_message}'. This is a mock response since the Gemini API key is not configured."


def _log_token_usage(conversation_id: int, conversation_obj, user_message: str, response: str):
    """Log token usage for monitoring and analytics."""
    try:
        from apps.integrations.models import GeminiLog

        # Rough token count (1 token ~= 4 characters)
        request_tokens = len(user_message) // 4
        response_tokens = len(response) // 4
        total_tokens = request_tokens + response_tokens

        GeminiLog.objects.create(
            conversation=conversation_obj,
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            model=settings.GEMINI_MODEL
        )

        logger.debug(f"Logged token usage: {total_tokens} tokens for conversation {conversation_id}")

    except Exception as e:
        logger.error(f"Failed to log token usage: {str(e)}")


def clear_conversation_memory(conversation_id: int):
    """Clear memory for a specific conversation (useful for cleanup)."""
    if conversation_id in _memory_store:
        del _memory_store[conversation_id]
        logger.debug(f"Cleared memory for conversation {conversation_id}")
