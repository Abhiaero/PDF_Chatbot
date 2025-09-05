import json
from typing import List, Dict

def save_test_questions(questions: List[Dict], filename: str = "test_questions.json"):
    """Save test questions to JSON file"""
    with open(filename, 'w') as f:
        json.dump(questions, f, indent=2)

def load_test_questions(filename: str = "test_questions.json") -> List[Dict]:
    """Load test questions from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def calculate_accuracy(test_results: List[Dict]) -> float:
    """Calculate accuracy from test results"""
    correct = sum(1 for result in test_results if result.get('is_correct', False))
    total = len(test_results)
    return correct / total if total > 0 else 0