SUMMARY_PROMPT = """
You are an AI assistant. Summarize the following document in 150 words or fewer:

{content}
"""

LOGIC_QUESTION_GEN_PROMPT = """Based on the following document content, generate exactly 3 multiple choice questions that test logical reasoning and comprehension. Each question should be directly related to the content provided.

Document Content:
{context}

Please generate questions in the following EXACT JSON format:
[
  {{
    "question": "Your question here?",
    "options": [
      "A) Option 1",
      "B) Option 2", 
      "C) Option 3",
      "D) Option 4"
    ],
    "answer": "A",
    "explanation": "Brief explanation of why this is correct."
  }}
]

Requirements:
1. Questions must be based on the actual document content
2. Test comprehension, analysis, or logical reasoning
3. Include exactly 4 options (A, B, C, D)
4. Answer should be just the letter (A, B, C, or D)
5. Provide a clear explanation
6. Return ONLY the JSON array, no other text

JSON Response:
"""

EVALUATE_RESPONSE_PROMPT = """
You are evaluating a user's answer to a reasoning question from a document.

Document: {context}
Question: {question}
User's Answer: {response}

Evaluate the correctness of the user's answer, then briefly explain why it is right or wrong.
"""