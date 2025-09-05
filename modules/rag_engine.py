import google.generativeai as genai
import os
from typing import List, Dict
from datetime import datetime

class RAGEngine:
    def __init__(self):
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.chat_history = []
            print("✓ Gemini Flash 1.5 model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            self.model = None
            self.chat_history = []
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        if not retrieved_chunks:
            return "No relevant document excerpts found."
        
        context = "RELEVANT DOCUMENT CONTEXT:\n\n"
        for i, chunk in enumerate(retrieved_chunks):
            # Truncate text for better readability
            text_preview = chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text']
            context += f"📄 Excerpt {i+1} (Page {chunk['page']}): {text_preview}\n"
            context += f"   Similarity: {chunk.get('similarity_score', 0.0):.3f}\n\n"
        return context
    
    def format_chat_history(self, chat_history: List[Dict]) -> str:
        """Format chat history for context"""
        if not chat_history:
            return "No previous conversation history."
        
        history_text = "CONVERSATION HISTORY:\n\n"
        for i, msg in enumerate(chat_history):
            role = "👤 USER" if msg['role'] == 'user' else "🤖 ASSISTANT"
            # Skip the current question if it's in history
            if i == len(chat_history) - 1 and msg['role'] == 'user':
                continue
            history_text += f"{role}: {msg['content']}\n"
            
            # Add sources for assistant messages
            if msg['role'] == 'assistant' and msg.get('sources'):
                history_text += "   📍 Sources: "
                source_pages = list(set([s['page'] for s in msg['sources'] if 'page' in s]))
                history_text += f"Pages {', '.join(map(str, sorted(source_pages)))}\n"
            
            history_text += "\n"
        
        return history_text
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict], chat_history: List[Dict] = None) -> Dict:
        """Generate response using RAG with proper chat history integration"""
        if self.model is None:
            return {
                'answer': "Error: AI model not initialized. Please check your API key.",
                'timestamp': datetime.now().isoformat(),
                'sources': []
            }
        
        # Format contexts
        document_context = self.format_context(retrieved_chunks)
        history_context = self.format_chat_history(chat_history or [])
        
        # Determine if this is a follow-up/about conversation question
        is_about_conversation = any(keyword in query.lower() for keyword in [
            'previous', 'before', 'earlier', 'first question', 'last question',
            'what did i ask', 'what was my', 'our conversation', 'chat history',
            'did i ask', 'have we discussed'
        ])
        
        # Create the prompt with enhanced instructions
        prompt = f"""
        ROLE: You are a helpful assistant that answers questions based on both the provided document content AND the conversation history.

        {history_context}

        {document_context}

        CURRENT QUESTION: {query}

        IMPORTANT INSTRUCTIONS:
        1. FIRST check if this question is about the CONVERSATION HISTORY (e.g., "what was my first question", "what did we discuss earlier")
        2. If it's about conversation history, answer using ONLY the conversation history above
        3. If it's about document content, answer using BOTH document context and conversation history when relevant
        4. If the answer requires document content but none is provided, say "I couldn't find relevant information in the document"
        5. ALWAYS cite page numbers when using document content [Page X]
        6. For conversation-based answers, be specific about what was discussed
        7. If you need to reference previous answers, mention they were based on specific pages

        CRITICAL: If the user asks about the conversation (e.g., "what was my first question"), you MUST use the conversation history, not the document.

        ANSWER:
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Extract sources - only include document sources, not conversation references
            sources = []
            if retrieved_chunks and not is_about_conversation:
                sources = [{'page': chunk['page'], 'text': chunk['text'][:200]} 
                          for chunk in retrieved_chunks[:3]]
            
            return {
                'answer': response.text,
                'timestamp': datetime.now().isoformat(),
                'sources': sources,
                'used_history': is_about_conversation
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return {
                'answer': "I'm having trouble generating a response at the moment. Please try again.",
                'timestamp': datetime.now().isoformat(),
                'sources': [],
                'used_history': False
            }
    
    def add_to_history(self, role: str, content: str, sources: List[Dict] = None):
        """Add message to chat history"""
        self.chat_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'sources': sources or []
        })
    
    def get_history(self) -> List[Dict]:
        """Get chat history"""
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []