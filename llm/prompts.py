"""
System and Task Prompts:
Supports Multilingual responses (English, Hindi), Python Code Generation,
Knowledge Graph Entity-Relation Extraction, HyDE, and Evaluation.
"""

# -------------------------------------------------------------
# System Prompts by Language and Mode
# -------------------------------------------------------------

SYSTEM_PROMPT_EN = """You are an advanced enterprise AI assistant powered by state-of-the-art Retrieval-Augmented Generation (RAG).
Your goal is to provide accurate, concise, and well-grounded answers based strictly on the provided context passages.
- Cite specific sources or document references where available.
- If the answer cannot be determined from the context, state that clearly rather than hallucinating.
- Maintain a professional, clear, and informative tone.
"""

SYSTEM_PROMPT_HI = """आप एक उन्नत एंटरप्राइज एआई सहायक हैं जो अत्याधुनिक रिट्रीवल-ऑगमेंटेड जेनरेशन (RAG) द्वारा संचालित है।
आपका लक्ष्य दिए गए संदर्भ (context) के आधार पर सटीक, स्पष्ट और विश्वसनीय उत्तर प्रदान करना है।
- कृपया अपना उत्तर शुद्ध, सरल और स्वाभाविक हिन्दी (Devanagari Hindi) में दें।
- यदि आवश्यक हो तो तकनीकी शब्दों के लिए सामान्य अंग्रेजी शब्दों का कोष्ठक में प्रयोग कर सकते हैं।
- संदर्भ में दी गई जानकारी के आधार पर ही उत्तर दें, अपनी ओर से कोई काल्पनिक तथ्य न जोड़ें।
"""

SYSTEM_PROMPT_PYTHON_CODE = """You are an expert Principal Python Engineer and Data Architect.
When providing coding answers:
1. Write clean, PEP 8 compliant, production-grade Python 3 code.
2. Include comprehensive type annotations (typing / typing_extensions).
3. Include informative Google-style or Sphinx docstrings.
4. Provide unit tests using `pytest` or `unittest` verifying the solution.
5. Provide a brief explanation of design choices and time/space complexity.
"""

GRAPH_EXTRACTION_PROMPT = """Analyze the following text and extract all key entities and their semantic relationships.
Return the output strictly in valid JSON format with the following structure:
{{
  "entities": [
    {{"name": "Entity Name", "type": "CONCEPT|ORGANIZATION|TECHNOLOGY|METRIC|PERSON", "description": "Brief description"}}
  ],
  "relations": [
    {{"source": "Entity Name 1", "target": "Entity Name 2", "relation": "USES|IMPLEMENTS|CONTAINS|PRODUCES|RELATED_TO", "description": "Short explanation"}}
  ]
}}

Text to extract from:
---
{text}
---
Ensure entity names are normalized (lowercase or standard capitalization). Return ONLY the raw JSON without markdown code fences.
"""

HYDE_PROMPT = """Given the question below, write a detailed, authoritative hypothetical paragraph that perfectly answers it.
Do not worry if some specific figures are assumed; focus on covering key domain concepts and terminology.

Question: {query}
Hypothetical Answer:"""

QUERY_EXPANSION_PROMPT = """You are an expert search optimization engine.
Given the user query, generate 3 high-relevance search query variations that expand acronyms, include synonyms, and optimize for vector/lexical retrieval.
Output strictly one query per line, without numbers or bullets.

Original Query: {query}
Variations:"""


def get_rag_prompt(
    query: str,
    context: str,
    language: str = "en",
    mode: str = "general"
) -> Tuple_Prompt:
    """
    Constructs the formatted prompt and matching system prompt.
    """
    lang = language.lower()
    if lang in ["hi", "hindi"]:
        system_prompt = SYSTEM_PROMPT_HI
        user_prompt = f"""नीचे दिए गए संदर्भ (Context) को ध्यानपूर्वक पढ़ें और पूछे गए प्रश्न का हिन्दी में सटीक उत्तर दें:

संदर्भ (Context):
---------------------
{context}
---------------------

प्रश्न (Question): {query}

उत्तर:"""
    elif mode.lower() == "code":
        system_prompt = SYSTEM_PROMPT_PYTHON_CODE
        user_prompt = f"""Context documentation and codebase context:
---------------------
{context}
---------------------

User Coding Task / Question:
{query}

Provide a complete, production-ready Python solution with explanation and test assertions:"""
    else:
        system_prompt = SYSTEM_PROMPT_EN
        user_prompt = f"""Use the following context passages to answer the user's question.
If the answer cannot be found in the context, state that you do not have enough information from the documents.

Context:
---------------------
{context}
---------------------

Question: {query}

Answer:"""

    return system_prompt, user_prompt


Tuple_Prompt = tuple[str, str]
