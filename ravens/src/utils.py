import re
from typing import List

def extract_numbered_questions(content: str, marker: str = "</think>") -> List[str]:
    """
    Extracts numbered questions from the given content after the specified marker.

    Parameters:
        content (str): The full text to parse.
        marker (str): The marker after which to start extracting (default: "</think>").

    Returns:
        List[str]: A list of question strings in order.
    """
    # Step 1: Extract text after the marker
    idx = content.find(marker)
    text_to_parse = content[idx + len(marker):].strip() if idx != -1 else content

    # Step 2: Use a regex pattern to extract numbered questions
    pattern = r'\d+\.\s*(.*?)\s*(?=\d+\.|$)'
    raw_questions = re.findall(pattern, text_to_parse, flags=re.DOTALL)

    # Step 3: Clean up whitespace and return
    return [q.strip() for q in raw_questions]

# Example usage:
# questions = extract_numbered_questions(result.content)
# for i, q in enumerate(questions, start=1):
#     print(f"q{i} = \"{q}\"")
