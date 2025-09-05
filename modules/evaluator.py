import json
import os
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from modules.vector_store import VectorStore
from modules.rag_engine import RAGEngine


class Evaluator:
    def __init__(self):
        self.vector_store = VectorStore()
        self.rag_engine = RAGEngine()
        self.results_file = "evaluation_results.json"
        self.test_questions_file = "test_questions.json"

    def load_test_questions(self) -> List[Dict]:
        """Load test questions from JSON file"""
        try:
            if os.path.exists(self.test_questions_file):
                with open(self.test_questions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Return default test questions
                return self.get_default_test_questions()
        except Exception as e:
            print(f"Error loading test questions: {e}")
            return self.get_default_test_questions()
        
    
    def get_default_test_questions(self) -> List[Dict]:
        """Get default test questions for evaluation"""
        return [
            {
                "id": 1,
                "question": "What is the main topic of this document?",
                "category": "general",
                "difficulty": "easy"
            },
            {
                "id": 2,
                "question": "What are the key findings or conclusions?",
                "category": "general", 
                "difficulty": "medium"
            },
            {
                "id": 3,
                "question": "On which pages is [important concept] discussed?",
                "category": "specific",
                "difficulty": "medium"
            },
            {
                "id": 4,
                "question": "What methodology was used in the research?",
                "category": "specific",
                "difficulty": "medium"
            },
            {
                "id": 5,
                "question": "What are the recommendations provided?",
                "category": "specific",
                "difficulty": "easy"
            },
            {
                "id": 6,
                "question": "Who are the authors and what are their affiliations?",
                "category": "factual", 
                "difficulty": "easy"
            },
            {
                "id": 7,
                "question": "What data sources were used?",
                "category": "specific",
                "difficulty": "hard"
            },
            {
                "id": 8,
                "question": "What limitations are mentioned in the document?",
                "category": "specific",
                "difficulty": "medium"
            },
            {
                "id": 9,
                "question": "How many sections does the document have?",
                "category": "structural",
                "difficulty": "easy"
            },
            {
                "id": 10,
                "question": "What future work is suggested?",
                "category": "specific",
                "difficulty": "medium"
            }
        ]
    

    def is_index_valid(self) -> bool:
        """Check if the current index is valid and matches the expected PDF"""
        try:
            # Check if index exists and has content
            if not self.vector_store.load_index():
                return False
            
            # Additional checks to ensure index is not stale
            if (not hasattr(self.vector_store, 'metadata') or 
                not self.vector_store.metadata or 
                not hasattr(self.vector_store, 'index') or 
                self.vector_store.index is None or 
                self.vector_store.index.ntotal == 0):
                return False
                
            return True
            
        except Exception as e:
            print(f"Error checking index validity: {e}")
            return False
        
    
    # def generate_answers(self, questions: List[Dict]) -> List[Dict]:
    #     """Generate answers for test questions"""
    #     if not self.vector_store.load_index():
    #         raise ValueError("No PDF index found. Please process a PDF first.")
        
    #     results = []
    #     for question in questions:
    #         try:
    #             # Search for relevant chunks
    #             retrieved_chunks = self.vector_store.search(question['question'], k=5)
                
    #             # Generate response
    #             response = self.rag_engine.generate_response(
    #                 question['question'], 
    #                 retrieved_chunks, 
    #                 []
    #             )
                
    #             result = {
    #                 **question,
    #                 "generated_answer": response['answer'],
    #                 "retrieved_chunks": retrieved_chunks[:3],  # Top 3 chunks
    #                 "timestamp": datetime.now().isoformat(),
    #                 "sources_used": [chunk['page'] for chunk in retrieved_chunks[:3]],
    #                 "manual_judgment": None,
    #                 "is_correct": None,
    #                 "confidence": None,
    #                 "notes": None
    #             }
                
    #             results.append(result)
                
    #         except Exception as e:
    #             print(f"Error generating answer for question {question['id']}: {e}")
    #             results.append({
    #                 **question,
    #                 "generated_answer": f"Error: {str(e)}",
    #                 "retrieved_chunks": [],
    #                 "timestamp": datetime.now().isoformat(),
    #                 "sources_used": [],
    #                 "manual_judgment": None,
    #                 "is_correct": False,
    #                 "confidence": 0,
    #                 "notes": f"Generation error: {e}"
    #             })
        
    #     return results

    def generate_answers(self, questions: List[Dict]) -> List[Dict]:
        """Generate answers for test questions using the SAME logic as app.py"""
        # First check if we have a valid index
        if not self.is_index_valid():
            raise ValueError("No valid PDF index found. Please process a PDF first in the main app.")
        
        results = []
        for question in questions:
            try:
                # Use the EXACT SAME logic as app.py
                # For conversation history questions, don't search the document
                is_about_chat = any(keyword in question['question'].lower() for keyword in [
                    'previous', 'before', 'earlier', 'first question', 
                    'last question', 'what did i ask', 'what was my',
                    'our conversation', 'chat history', 'did i ask'
                ])
                
                if is_about_chat:
                    # Don't search document for chat history questions
                    retrieved_chunks = []
                else:
                    # Search for relevant document chunks (same as app.py)
                    retrieved_chunks = self.vector_store.search(question['question'], k=5)
                
                # Generate response using the SAME method as app.py
                response = self.rag_engine.generate_response(
                    question['question'], 
                    retrieved_chunks, 
                    []  # Empty chat history for evaluation
                )
                
                result = {
                    **question,
                    "generated_answer": response['answer'],
                    "retrieved_chunks": retrieved_chunks[:3],  # Top 3 chunks
                    "timestamp": datetime.now().isoformat(),
                    "sources_used": [chunk['page'] for chunk in retrieved_chunks[:3]],
                    "manual_judgment": None,
                    "is_correct": None,
                    "confidence": None,
                    "notes": None,
                    "used_history": response.get('used_history', False)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error generating answer for question {question['id']}: {e}")
                results.append({
                    **question,
                    "generated_answer": f"Error: {str(e)}",
                    "retrieved_chunks": [],
                    "timestamp": datetime.now().isoformat(),
                    "sources_used": [],
                    "manual_judgment": "Error",
                    "is_correct": False,
                    "confidence": 0,
                    "notes": f"Generation error: {e}",
                    "used_history": False
                })
        
        return results


    def save_results(self, results: List[Dict]):
        """Save evaluation results to JSON file"""
        try:
            # Load existing results if any
            all_results = []
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            
            # Add new results
            all_results.extend(results)
            
            # Save back to file
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved {len(results)} evaluation results")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def load_results(self) -> List[Dict]:
        """Load evaluation results from JSON file"""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading results: {e}")
            return []
    
    def calculate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate accuracy metrics from results"""
        if not results:
            return {}
        
        total = len(results)
        correct = sum(1 for r in results if r.get('is_correct') is True)
        partially_correct = sum(1 for r in results if r.get('is_correct') == 'partial')
        incorrect = sum(1 for r in results if r.get('is_correct') is False)
        not_judged = sum(1 for r in results if r.get('is_correct') is None)
        
        accuracy = correct / total if total > 0 else 0
        partial_accuracy = (correct + partially_correct * 0.5) / total if total > 0 else 0
        
        # Calculate by category
        categories = {}
        for result in results:
            cat = result.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = {'total': 0, 'correct': 0, 'partial': 0}
            
            categories[cat]['total'] += 1
            if result.get('is_correct') is True:
                categories[cat]['correct'] += 1
            elif result.get('is_correct') == 'partial':
                categories[cat]['partial'] += 1
        
        # Calculate by difficulty
        difficulties = {}
        for result in results:
            diff = result.get('difficulty', 'unknown')
            if diff not in difficulties:
                difficulties[diff] = {'total': 0, 'correct': 0, 'partial': 0}
            
            difficulties[diff]['total'] += 1
            if result.get('is_correct') is True:
                difficulties[diff]['correct'] += 1
            elif result.get('is_correct') == 'partial':
                difficulties[diff]['partial'] += 1
        
        return {
            'total_questions': total,
            'correct_answers': correct,
            'partially_correct': partially_correct,
            'incorrect_answers': incorrect,
            'not_judged': not_judged,
            'accuracy': round(accuracy * 100, 2),
            'partial_accuracy': round(partial_accuracy * 100, 2),
            'categories': categories,
            'difficulties': difficulties,
            'evaluation_date': datetime.now().isoformat()
        }
    
    def generate_accuracy_table(self, results: List[Dict]) -> pd.DataFrame:
        """Generate a pandas DataFrame with accuracy metrics"""
        metrics = self.calculate_metrics(results)
        
        # Create summary table
        summary_data = {
            'Metric': [
                'Total Questions',
                'Correct Answers', 
                'Partially Correct',
                'Incorrect Answers',
                'Not Judged',
                'Accuracy (%)',
                'Partial Accuracy (%)'
            ],
            'Value': [
                metrics['total_questions'],
                metrics['correct_answers'],
                metrics['partially_correct'],
                metrics['incorrect_answers'],
                metrics['not_judged'],
                metrics['accuracy'],
                metrics['partial_accuracy']
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def generate_detailed_report(self, results: List[Dict]) -> str:
        """Generate a detailed text report"""
        metrics = self.calculate_metrics(results)
        
        report = [
            "PDF Chatbot Evaluation Report",
            "=" * 40,
            f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Questions: {metrics['total_questions']}",
            f"Correct Answers: {metrics['correct_answers']}",
            f"Partially Correct: {metrics['partially_correct']}",
            f"Incorrect Answers: {metrics['incorrect_answers']}",
            f"Accuracy: {metrics['accuracy']}%",
            f"Partial Accuracy: {metrics['partial_accuracy']}%",
            "",
            "By Category:"
        ]
        
        for category, stats in metrics['categories'].items():
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            report.append(f"  {category}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
        report.extend(["", "By Difficulty:"])
        for difficulty, stats in metrics['difficulties'].items():
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            report.append(f"  {difficulty}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
        return "\n".join(report)