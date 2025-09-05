import faiss
import numpy as np
import json
import os
from typing import List, Dict
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, index_path: str = "storage/faiss_index", metadata_path: str = "storage/metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        self.embedding_model = None
        self.embedding_dimension = None
        self.is_initialized = False
        self.used_gemini_for_index = False
        
    def initialize_embedding_model(self, use_gemini: bool = True):
        """Initialize embedding model with proper error handling"""
        try:
            # Try to use Gemini embeddings if requested and API key available
            api_key = os.getenv('GEMINI_API_KEY')
            if use_gemini and api_key:
                genai.configure(api_key=api_key)
                
                # Test the API with a simple embedding - ensure proper format
                test_result = genai.embed_content(
                    model="models/embedding-001",
                    content=["test embedding"],
                    task_type="retrieval_document"
                )
                
                if 'embedding' in test_result:
                    # Handle the embedding response format correctly
                    embedding_data = test_result['embedding']
                    if isinstance(embedding_data, list):
                        if len(embedding_data) > 0 and isinstance(embedding_data[0], list):
                            # Multiple embeddings returned (list of lists)
                            self.embedding_dimension = len(embedding_data[0])
                        else:
                            # Single embedding returned (flat list)
                            self.embedding_dimension = len(embedding_data)
                        
                        self.embedding_model = 'gemini'
                        self.used_gemini_for_index = True
                        print(f"✓ Using Gemini embeddings (dimension: {self.embedding_dimension})")
                        self.is_initialized = True
                        return True
                    else:
                        print("⚠️ Gemini returned non-list embedding")
                
            # Fall back to local model
            return self._initialize_local_model()
                
        except Exception as e:
            print(f"Gemini embedding failed: {e}, falling back to local model")
            return self._initialize_local_model()
    
    def _initialize_local_model(self):
        """Initialize local embedding model as fallback"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.used_gemini_for_index = False
            print(f"✓ Using local SentenceTransformer (dimension: {self.embedding_dimension})")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize local model: {e}")
            self.is_initialized = False
            return False
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts with proper error handling"""
        if not self.is_initialized:
            if not self.initialize_embedding_model():
                raise ValueError("Embedding model not initialized")
        
        if self.embedding_model == 'gemini':
            try:
                # Handle single text vs multiple texts
                if len(texts) == 1:
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=texts[0],
                        task_type="retrieval_document"
                    )
                else:
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=texts,
                        task_type="retrieval_document"
                    )
                
                # Handle the response format correctly
                if 'embedding' in result:
                    embedding_data = result['embedding']
                    if isinstance(embedding_data, list):
                        if len(embedding_data) > 0 and isinstance(embedding_data[0], list):
                            # Multiple embeddings returned (list of lists)
                            embeddings = np.array(embedding_data).astype('float32')
                        else:
                            # Single embedding returned (flat list)
                            embeddings = np.array([embedding_data]).astype('float32')
                    else:
                        raise ValueError("Unexpected embedding format from Gemini")
                else:
                    raise ValueError("No embedding in Gemini response")
                
                print(f"✓ Generated embeddings: {embeddings.shape}")
                return embeddings
                
            except Exception as e:
                print(f"Gemini embedding generation failed: {e}")
                print("Falling back to local model...")
                if not self._initialize_local_model():
                    raise ValueError("All embedding methods failed")
                # Fall through to local model
        
        # Use local model
        if isinstance(self.embedding_model, SentenceTransformer):
            embeddings = self.embedding_model.encode(texts).astype('float32')
            print(f"✓ Generated local embeddings: {embeddings.shape}")
            return embeddings
        else:
            raise ValueError("No valid embedding model available")
    
    def load_index(self):
        """Load existing index with proper initialization"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                print("Loading existing index...")
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                # Get index dimension
                index_dimension = self.index.d
                print(f"Index dimension: {index_dimension}")
                
                # Initialize appropriate embedding model based on index dimension
                if index_dimension == 768:
                    # Index was created with Gemini
                    print("Index requires Gemini embeddings")
                    if not self.initialize_embedding_model(use_gemini=True):
                        print("❌ Failed to initialize Gemini embeddings")
                        return False
                elif index_dimension == 384:
                    # Index was created with local model
                    print("Index requires local embeddings")
                    if not self._initialize_local_model():
                        return False
                else:
                    print(f"❌ Unknown index dimension: {index_dimension}")
                    return False
                
                # Final dimension check
                if self.embedding_dimension != self.index.d:
                    print(f"❌ Critical: Embedding dimension {self.embedding_dimension} doesn't match index dimension {self.index.d}")
                    return False
                
                print(f"✓ Index loaded with {len(self.metadata)} chunks")
                print(f"✓ Using: {'Gemini' if self.used_gemini_for_index else 'Local model'}")
                return True
                
            except Exception as e:
                print(f"Error loading index: {e}")
                return False
        return False
    

    def _reinitialize_for_index_compatibility(self):
        """Reinitialize embedding model to be compatible with existing index"""
        if hasattr(self, 'index') and self.index is not None:
            index_dimension = self.index.d
            
            # Try to use the same type of embedding that created the index
            if index_dimension == 768:  # Gemini dimension
                print("Index was created with Gemini, trying to use Gemini...")
                if not self.initialize_embedding_model(use_gemini=True):
                    raise ValueError("Index requires Gemini but API not available")
            elif index_dimension == 384:  # SentenceTransformer dimension
                print("Index was created with local model, using local model...")
                self._initialize_local_model()
            else:
                raise ValueError(f"Unknown index dimension: {index_dimension}")
    
    def create_index(self, chunks: List[Dict]):
        """Create FAISS index from chunks"""
        texts = [chunk['text'] for chunk in chunks]
        
        # Initialize embedding model (try Gemini first)
        if not self.initialize_embedding_model(use_gemini=True):
            raise ValueError("Failed to initialize embedding model")
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.metadata = chunks
        
        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ New index created with {len(chunks)} chunks")
        print(f"✓ Index dimension: {dimension}, Using: {'Gemini' if self.used_gemini_for_index else 'Local model'}")
    
    def load_index(self):
        """Load existing index with proper initialization"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                print("Loading existing index...")
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                # Initialize embedding model (try to match index type)
                index_dimension = self.index.d
                if index_dimension == 768:
                    # Index was created with Gemini
                    if not self.initialize_embedding_model(use_gemini=True):
                        print("⚠️  Index requires Gemini but failed to initialize, trying local...")
                        if not self._initialize_local_model():
                            return False
                else:
                    # Use appropriate model
                    if not self.initialize_embedding_model(use_gemini=False):
                        return False
                
                # Final dimension check
                if self.embedding_dimension != self.index.d:
                    print(f"❌ Critical: Embedding dimension {self.embedding_dimension} doesn't match index dimension {self.index.d}")
                    return False
                
                print(f"✓ Index loaded with {len(self.metadata)} chunks")
                print(f"✓ Index dimension: {self.index.d}, Using: {'Gemini' if self.used_gemini_for_index else 'Local model'}")
                return True
                
            except Exception as e:
                print(f"Error loading index: {e}")
                return False
        return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks with proper validation"""
        # Check if index is loaded
        if self.index is None:
            print("Index not loaded, attempting to load...")
            if not self.load_index():
                raise ValueError("Index not initialized and cannot be loaded")
        
        # Check if embedding model is initialized
        if not self.is_initialized:
            print("Embedding model not initialized, initializing...")
            if not self.initialize_embedding_model():
                raise ValueError("Embedding model not initialized")
        
        # Check if we have data to search
        if self.index.ntotal == 0:
            print("Warning: Index is empty")
            return []
        
        # Get query embedding
        query_embedding = self.get_embeddings([query])
        
        # Ensure k doesn't exceed number of vectors
        k = min(k, self.index.ntotal)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(1 / (1 + distance))
                results.append(result)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
    
    def is_ready(self) -> bool:
        """Check if the vector store is ready for searching"""
        return (self.index is not None and 
                self.is_initialized and 
                len(self.metadata) > 0)
    
    def clear_index(self):
        """Clear the current index and metadata"""
        self.index = None
        self.metadata = []
        
    def delete_storage_files(self):
        """Delete the index and metadata files from storage"""
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            print("✓ Storage files deleted")
        except Exception as e:
            print(f"Error deleting storage files: {e}")


    def search_structured_data(self, query: str, k: int = 5) -> List[Dict]:
        """Search specifically for structured data chunks"""
        if self.index is None or not self.metadata:
            return self.search(query, k)  # Fallback to regular search
        
        # Get regular results first
        regular_results = self.search(query, k * 2)  # Get more results to filter
        
        # Filter for structured data chunks
        structured_results = []
        for result in regular_results:
            if result.get('is_structured') and any(keyword in query.lower() for keyword in 
                                                ['percentage', 'percent', '%', 'chart', 
                                                'graph', 'table', 'data', 'number']):
                structured_results.append(result)
        
        # If we found structured data, return it prioritized
        if structured_results:
            return structured_results[:k]
        
        # Otherwise return regular results
        return regular_results[:k]