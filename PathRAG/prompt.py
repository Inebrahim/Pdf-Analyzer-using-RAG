GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

# --- Language and Delimiters ---
PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# --- Domain-Specific Entity Types ---
# This is a good place to customize the types of entities you want to extract.
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "requirement_id", "component", "signal", "interface", "function", "role", "document_section"
]

# --- Entity and relationship extraction Prompt ---
# This prompt instructs the LLM on how to find and format entities and relationships.
# It is model-agnostic and works well with instruction-tuned models like Mistral.
# In prompt.py

PROMPTS["entity_extraction"] = """Your task is to extract technical entities and their relationships from the following text.
Use {language} as the output language.

Follow these instructions STRICTLY:
1. First, identify all technical entities. For each entity, write a line in this exact format:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<brief_description>)

2. Second, identify all relationships between those entities. For each relationship, write a line in this exact format:
("relationship"{tuple_delimiter}<source_entity_name>{tuple_delimiter}<target_entity_name>{tuple_delimiter}<relationship_description>{tuple_delimiter}<keywords>{tuple_delimiter}<strength_score_1.0_to_5.0>)

3. Separate every line with a **{record_delimiter}**.
4. When you are completely finished, end your entire response with **{completion_delimiter}**.

Here is the text to analyze:
{input_text}
"""

# --- Summarize entity descriptions Prompt ---
PROMPTS["summarize_entity_descriptions"] = """Summarize the descriptions for one or more AUTOSAR entities.
Resolve contradictions and combine all details into a single, clear description.
Entities: {entity_name}
Descriptions: {description_list}
Output:
"""

# --- RAG response Prompt ---
# This is the final prompt that uses the retrieved context to answer the user's question.
# In prompt.py, replace the existing rag_response with this one.

PROMPTS["rag_response"] = """---Role---
You are a technical assistant answering questions about AUTOSAR requirements and design documents.

---Instructions---
1.  Your primary goal is to provide a concise and accurate answer using ONLY the information found in the supplied "Context" section, which is derived from PDF extractions, embeddings, and knowledge graphs.
2.  **You are strictly forbidden from using any external knowledge or making assumptions beyond the provided context.**
3.  Do not provide Python code examples unless the code is explicitly present in the context.
4.  If the context does not contain enough information to answer the question, you MUST respond with a clear statement like "The provided document context does not contain a specific answer to this question."
5.  When relevant, highlight specific requirements (like SRS_... or SWS_...), components, or images mentioned in the context.

---Target response length and format---
{response_type}

---Context---
{context_data}
"""

# --- Keyword Extraction from User Query ---
# This prompt is used to analyze the user's question and extract key terms.
PROMPTS["keywords_extraction"] = """Your task is to extract high-level concepts and low-level entities from a user's question about a technical document.

- **High-level keywords** are broad concepts (e.g., "purpose", "dependencies", "scope").
- **Low-level keywords** are specific, named entities (e.g., "SWS_Rte_01001", "ECU").

User Question: "{query}"

Respond ONLY with a single, valid JSON object in the following format. Do not add any other text or explanations.
{{
    "high_level_keywords": ["list", "of", "concepts"],
    "low_level_keywords": ["list", "of", "entities"]
}}
"""

# --- A Default Fail Response ---
# This is a fallback message if the RAG process fails at any point.
PROMPTS["fail_response"] = "I'm sorry, I was unable to find a definitive answer in the provided document. Please try rephrasing your question."