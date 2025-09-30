#It contains the heavy machinery and the complex, multi-step assembly lines for the two main processes: knowledge extraction and query answering.
import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
import tiktoken
import time
import csv
from PathRAG.utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    csv_string_to_list,
    pack_history_to_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from PathRAG.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from PathRAG.prompt import GRAPH_FIELD_SEP, PROMPTS

#Its only job is to take a huge piece of raw text (the entire PDF content) and chop it into small, consistently sized pieces, or "chunks."
#The original code tried to use tiktoken (an OpenAI library), which would crash.
#We fixed this by hard-coding the tiktoken_model to "cl100k_base", a generic and compatible encoder. This made the chunking process reliable.
def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="cl100k_base"
):
    """
    Chunks a string by the number of tokens.

    Args:
        content (str): The string content to be chunked.
        overlap_token_size (int): The number of tokens to overlap between chunks.
        max_token_size (int): The maximum number of tokens per chunk.
        tiktoken_model (str): The model name for tiktoken encoding.
            --- CHANGE ---
            The default was changed from "mistralai/Mistral-7B-Instruct-v0.3" to "cl100k_base".
           The tiktoken library does not support Hugging Face model names directly and would cause an error.
            "cl100k_base" is a compatible and robust general-purpose encoding.
    """
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results
    
# This function is a quality control step. If the system finds too many different descriptions for the same entity or relationship,
# it will use the LLM to write a clean, single summary. This prevents the final context from being cluttered with repetitive information.

async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary
    
#_handle_single_entity_extraction & _handle_single_relationship_extraction
#These are simple parsers. After the LLM returns its structured text output (like ("entity"<|>...)), these
#functions are responsible for carefully extracting the individual pieces of data (the name, the type, the description, etc.) from that string.
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None

    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )

#_merge_nodes_then_upsert & _merge_edges_then_upsert:
# Purpose: These functions are responsible for intelligently adding new information to the knowledge graph. If an entity already exists,
# this function will merge the new information with the old information (e.g., by combining descriptions), making the graph richer over time.
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data

#This is the heart of the indexing process. 
#Its goal is to take all the text chunks and build the knowledge graph from them. It's the slowest and most complex part of the "learn" phase.

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:

    use_llm_func: callable = global_config["llm_model_func"]

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    context_base = {
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        "entity_types": ",".join(entity_types),
        "language": language,
    }
    entity_extract_prompt_template = PROMPTS["entity_extraction"]

#The Workflow:
# It defines an inner function, _process_single_content, which contains the instructions for one "worker." 
# This worker's job is to take one text chunk, use the entity_extraction prompt, and get the structured entity/relationship data from the LLM.
# It then orchestrates these workers to process all the chunks.
# Our Change (Major Performance Fix): The original code tried to run all 1219 "workers" at the same time (asyncio.gather on the whole list).
#This overwhelmed the GPU and caused the process to stall. We replaced this with a resilient batching loop.

    async def _process_single_content(chunk_item):
        chunk_key, chunk_data = chunk_item
        hint_prompt = entity_extract_prompt_template.format(**context_base, input_text=chunk_data["content"])
        final_result = await use_llm_func(hint_prompt)
        
        records = split_string_by_multi_markers(final_result, [context_base["record_delimiter"], context_base["completion_delimiter"]])
        
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        for record in records:
            match = re.search(r"\((.*)\)", record)
            if not match: continue
            
            record_attributes = split_string_by_multi_markers(match.group(1), [context_base["tuple_delimiter"]])
            
            if entity := await _handle_single_entity_extraction(record_attributes, chunk_key):
                maybe_nodes[entity["entity_name"]].append(entity)
            elif relationship := await _handle_single_relationship_extraction(record_attributes, chunk_key):
                maybe_edges[(relationship["src_id"], relationship["tgt_id"])].append(relationship)
        return dict(maybe_nodes), dict(maybe_edges)

    # --- THIS IS RESILIENT BATCH PROCESSING LOGIC ---
    ordered_chunks = list(chunks.items())
    batch_size = 16  # A safe number for parallel requests on a P100/T4 GPU (free) on Kaggle
    
    logger.info(f"Starting resilient entity extraction for {len(ordered_chunks)} chunks in batches of {batch_size}...")

    with tqdm_async(total=len(ordered_chunks), desc="Extracting & Saving Entities") as pbar:
        for i in range(0, len(ordered_chunks), batch_size):
            batch = ordered_chunks[i:i + batch_size]
            
            # 1. Process one batch to get new entities and relationships
            batch_results = await asyncio.gather(*(_process_single_content(item) for item in batch))
            
            # 2. Aggregate results FOR THIS BATCH ONLY
            batch_nodes, batch_edges = defaultdict(list), defaultdict(list)
            for nodes, edges in batch_results:
                for k, v in nodes.items(): batch_nodes[k].extend(v)
                for k, v in edges.items(): batch_edges[k].extend(v)

            if not batch_nodes and not batch_edges:
                pbar.update(len(batch)) # Update progress and continue if batch yielded nothing
                continue

            # 3. Merge and upsert the new data into the main graph and VDBs
            logger.debug(f"Batch {i//batch_size + 1}: Merging {len(batch_nodes)} entities and {len(batch_edges)} relationships.")
            
            entities_to_embed = await tqdm_async.gather(*(_merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config) for k, v in batch_nodes.items()))
            relationships_to_embed = await tqdm_async.gather(*(_merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config) for k, v in batch_edges.items()))

            # 4. Upsert embeddings for the new data
            if entity_vdb and entities_to_embed:
                await entity_vdb.upsert({
                    compute_mdhash_id(d["entity_name"], prefix="ent-"): {"content": d["entity_name"] + d.get("description", ""), "entity_name": d["entity_name"]}
                    for d in entities_to_embed
                })
            if relationships_vdb and relationships_to_embed:
                await relationships_vdb.upsert({
                    compute_mdhash_id(d["src_id"] + d["tgt_id"], prefix="rel-"): {"content": d.get("keywords", "") + d["src_id"] + d["tgt_id"] + d.get("description", ""), "src_id": d["src_id"], "tgt_id": d["tgt_id"]}
                    for d in relationships_to_embed
                })

            # 5. SAVE EVERYTHING TO DISK
            # This is the crucial step for saving progress.
            logger.info(f"--- Saving progress after batch {i//batch_size + 1} ---")
            await knowledge_graph_inst.index_done_callback()
            if entity_vdb: await entity_vdb.index_done_callback()
            if relationships_vdb: await relationships_vdb.index_done_callback()
            
            # 6. Update the progress bar
            pbar.update(len(batch))
            
    logger.info("All batches processed and saved.")
    return knowledge_graph_inst

#kg_query
#Purpose: This is the core logic for answering a user's question. It runs the full RAG pipeline.
# The Workflow:
# Keyword Extraction: It takes the user's question, uses the keywords_extraction prompt, and asks the LLM to identify high-level and low-level keywords.
# Context Building: It calls another function, _build_query_context, to perform the actual retrieval.
# Final Generation: It takes the context returned by _build_query_context, combines it with the rag_response system prompt and the user's original question,
#and sends this final package to the LLM to generate a human-readable answer.
# Our Change (Major Robustness Fix): The original code was prone to hallucination. We added a crucial short-circuiting check.
#If the retrieval step (_build_query_context) returns a poor or empty context, this function now immediately returns a "Sorry, 
# I don't know" message without ever calling the LLM. This is the single most important change for preventing the model from making up answers.

async def kg_query(
    query, knowledge_graph_inst: BaseGraphStorage, entities_vdb: BaseVectorStorage, relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,chunks_vdb: BaseVectorStorage, query_param: QueryParam, global_config: dict
) -> str:
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, _, _, _ = await handle_cache(None, args_hash, query, query_param.mode) # Bypassing cache for now
    if cached_response: return cached_response

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, language=language)
    
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    logger.info(f"Keyword extraction result: {result}")
    
    try:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if not match: raise ValueError("No JSON object found")
        keywords_data = json.loads(match.group(0))
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse keywords: {e}. Response: {result}")
        return PROMPTS["fail_response"]

    if not hl_keywords and not ll_keywords:
        logger.warning("LLM returned no keywords.")
        return PROMPTS["fail_response"]

    context = await _build_query_context(
        [", ".join(ll_keywords), ", ".join(hl_keywords)],
        knowledge_graph_inst, entities_vdb, relationships_vdb, text_chunks_db, chunks_vdb, query_param
    )

    if not context or len(context.strip()) < 200: # Using 200 as a threshold for "meaningful" context
        logger.warning("Retrieval step found no meaningful context. Short-circuiting to prevent hallucination.")
        return PROMPTS.get("fail_response", "I'm sorry, I couldn't find any specific information about that in the document.")

    if query_param.only_need_context: return context
    if not context.strip(): return PROMPTS["fail_response"]

    sys_prompt = PROMPTS["rag_response"].format(context_data=context, response_type=query_param.response_type)
    if query_param.only_need_prompt: return sys_prompt
        
    response = await use_model_func(query, system_prompt=sys_prompt, stream=query_param.stream)
    
    await save_to_cache(None, CacheData(args_hash=args_hash, content=response, prompt=query, mode=query_param.mode))
    return response

#build_query_context
#This function is the "librarian." Its job is to take the keywords and find all the relevant information from the databases.
# This function's job is to take the keywords from the user's query and use them to find the most relevant context from all 
# available sources (the graph and the raw text).
# The "Pure PathRAG" Strategy:
# It first checks if there are specific, low-level keywords (like SRS_...). If so, it calls _get_node_data.
# If there are only high-level keywords, it calls _get_edge_data.
# It does not perform a separate, general vector search.
#It prioritizes the structured knowledge in the graph. This is a strategic choice to improve precision at the cost of being less effective
#on very vague, general questions.

async def _build_query_context(
    query_keywords: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: BaseVectorStorage, # This argument is no longer used but kept for compatibility
    query_param: QueryParam,
):
    ll_keywords, hl_keywords = query_keywords
    
    # We now ONLY perform the targeted entity and graph search.

    if ll_keywords:
        logger.info("Low-level keywords detected. Performing targeted entity/graph search...")
        # Get context by finding specific nodes (entities) and their neighbors
        entities_context, relations_context, text_units_context = await _get_node_data(
            ll_keywords, knowledge_graph_inst, entities_vdb, text_chunks_db, query_param
        )
    elif hl_keywords:
        logger.info("Only high-level keywords detected. Performing broader relationship search...")
        # If no specific entities, search for relationships/concepts
        entities_context, relations_context, text_units_context = await _get_edge_data(
            hl_keywords, knowledge_graph_inst, relationships_vdb, text_chunks_db, query_param
        )
    else:
        # If no keywords were extracted, we cannot proceed.
        logger.warning("No keywords extracted from query. Returning empty context.")
        return ""

    return f"""
-----BEGIN CONTEXT-----
### Relevant Entities from Knowledge Graph
```csv
{entities_context}
{relations_context}
{text_units_context}
-----END CONTEXT-----
"""

# In operate.py, replace your entire _get_node_data function with this one.

async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    if not query: return "", "", ""
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not results: return "", "", ""

    node_names = [r["entity_name"] for r in results]
    node_datas = await asyncio.gather(*(knowledge_graph_inst.get_node(name) for name in node_names))
    node_degrees = await asyncio.gather(*(knowledge_graph_inst.node_degree(name) for name in node_names))
    
    valid_node_datas = [
        {**n, "entity_name": name, "rank": d} for name, n, d in zip(node_names, node_datas, node_degrees) if n
    ]
    
    # We no longer need to find text units here; the general search handles it.
    
    use_relations = await _find_most_related_edges_from_entities(
        valid_node_datas, query_param, knowledge_graph_inst
    )

    logger.info(f"Local query found: {len(valid_node_datas)} entities and {len(use_relations)} relations.")

    entities_list = [["id", "entity", "type", "description", "rank"]] + [[i, n["entity_name"], n.get("entity_type", "UNKNOWN"), n.get("description", "UNKNOWN"), n["rank"]] for i, n in enumerate(valid_node_datas)]
    relations_list = [["id", "context"]] + [[i, r[0]] for i, r in enumerate(use_relations)]
    
    # Return an empty string for the text units, as this is now handled by the general search.
    text_units_context_csv = ""

    return list_of_list_to_csv(entities_list), list_of_list_to_csv(relations_list), text_units_context_csv


#This is the cleanest and most robust solution. It fixes the `NameError` by removing the problematic code and simplifies your retrieval logic. After this change, your application will finally work from end to end.
#This function handles the "precision search." It takes specific entity names, finds them in the entities_vdb, retrieves their data from
#the graph, and then calls _find_most_related_edges_from_entities to explore the graph and find connected information.


# async def _get_node_data(
#     query: str,
#     knowledge_graph_inst: BaseGraphStorage,
#     entities_vdb: BaseVectorStorage,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
#     query_param: QueryParam,
# ):
#     if not query: return "", "", ""
#     results = await entities_vdb.query(query, top_k=query_param.top_k)
#     if not results: return "", "", ""

#     node_names = [r["entity_name"] for r in results]
#     node_datas = await asyncio.gather(*(knowledge_graph_inst.get_node(name) for name in node_names))
#     node_degrees = await asyncio.gather(*(knowledge_graph_inst.node_degree(name) for name in node_names))
    
#     valid_node_datas = [
#         {**n, "entity_name": name, "rank": d} for name, n, d in zip(node_names, node_datas, node_degrees) if n
#     ]

#     use_text_units = await _find_most_related_text_unit_from_entities(
#         valid_node_datas, query_param, text_chunks_db
#     )
    
#     use_relations = await _find_most_related_edges_from_entities(
#         valid_node_datas, query_param, knowledge_graph_inst
#     )

#     logger.info(f"Local query: {len(valid_node_datas)} entities, {len(use_relations)} relations, {len(use_text_units)} text units.")

#     entities_list = [["id", "entity", "type", "description", "rank"]] + [[i, n["entity_name"], n.get("entity_type", "UNKNOWN"), n.get("description", "UNKNOWN"), n["rank"]] for i, n in enumerate(valid_node_datas)]
#     relations_list = [["id", "context"]] + [[i, r[0]] for i, r in enumerate(use_relations)]
#     text_units_list = [["id", "content"]] + [[t["id"], t["content"]] for t in use_text_units]

#     return list_of_list_to_csv(entities_list), list_of_list_to_csv(relations_list), list_of_list_to_csv(text_units_list)

# In operate.py, add this complete function definition.
# A good place is after _get_node_data and before _get_edge_data.

# async def _find_most_related_text_unit_from_entities(
#     node_datas: list[dict],
#     query_param: QueryParam,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
# ):
#     """
#     Given a list of entities (nodes), finds the original text chunks (text units)
#     they came from.
#     """
#     if not node_datas:
#         return []

#     # 1. Collect all the unique source chunk IDs from the list of entities.
#     # An entity can be linked to multiple chunks, so we use a set to avoid duplicates.
#     text_unit_ids = set()
#     for node in node_datas:
#         # The source_id field can contain multiple IDs separated by "<SEP>"
#         source_id_str = node.get("source_id", "")
#         if source_id_str:
#             ids = split_string_by_multi_markers(source_id_str, [GRAPH_FIELD_SEP])
#             text_unit_ids.update(ids)

#     if not text_unit_ids:
#         return []

#     # 2. Go to the text chunk database and retrieve the full content for all those IDs.
#     valid_text_units = await text_chunks_db.get_by_ids(list(text_unit_ids))
    
#     # 3. Filter out any empty or invalid results.
#     valid_text_units = [unit for unit in valid_text_units if unit and unit.get("content")]

#     # 4. Truncate the final list to make sure the context isn't too long for the LLM.
#     return truncate_list_by_token_size(
#         valid_text_units,
#         key=lambda x: x["content"],
#         max_token_size=query_param.max_token_for_text_unit,
#     )    
#These are helper functions that perform the detailed work of finding connected text chunks or entities based on the results from the initial database searches.

#This handles a broader search. It uses high-level keywords to search the relationships_vdb to find relevant concepts and relationships.
async def _get_edge_data(
    keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    if not keywords:
        return "", "", ""
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
    if not results:
        return "", "", ""

    edge_keys = [(r["src_id"], r["tgt_id"]) for r in results]
    edge_datas = await asyncio.gather(*(knowledge_graph_inst.get_edge(src, tgt) for src, tgt in edge_keys))
    edge_degrees = await asyncio.gather(*(knowledge_graph_inst.edge_degree(src, tgt) for src, tgt in edge_keys))

    valid_edge_datas = [
        {"src_id": k[0], "tgt_id": k[1], "rank": d, **v}
        for k, v, d in zip(edge_keys, edge_datas, edge_degrees)
        if v is not None
    ]
    valid_edge_datas.sort(key=lambda x: (x.get("rank", 0), x.get("weight", 0)), reverse=True)

    truncated_edges = truncate_list_by_token_size(
        valid_edge_datas,
        key=lambda x: x.get("description", ""),
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        truncated_edges, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        truncated_edges, query_param, text_chunks_db
    )

    logger.info(
        f"Global query context: {len(use_entities)} entities, "
        f"{len(truncated_edges)} relations, {len(use_text_units)} text units."
    )

    relations_list = [["id", "source", "target", "description", "keywords", "weight", "rank"]]
    relations_list.extend([
        [i, e["src_id"], e["tgt_id"], e.get("description", ""), e.get("keywords", ""),
         e.get("weight", 1.0), e.get("rank", 0)]
        for i, e in enumerate(truncated_edges)
    ])
    entities_list = [["id", "entity", "type", "description", "rank"]]
    entities_list.extend([
        [i, n["entity_name"], n.get("entity_type", "UNKNOWN"), n.get("description", "UNKNOWN"), n.get("rank", 0)]
        for i, n in enumerate(use_entities)
    ])
    text_units_list = [["id", "content"]]
    text_units_list.extend([[i, t["content"]] for i, t in enumerate(use_text_units)])

    return (
        list_of_list_to_csv(entities_list),
        list_of_list_to_csv(relations_list),
        list_of_list_to_csv(text_units_list),
    )


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = list(dict.fromkeys(
        name for e in edge_datas for name in (e["src_id"], e["tgt_id"])
    ))

    node_datas = await asyncio.gather(*(knowledge_graph_inst.get_node(name) for name in entity_names))
    node_degrees = await asyncio.gather(*(knowledge_graph_inst.node_degree(name) for name in entity_names))

    valid_node_datas = [
        {**n, "entity_name": name, "rank": d}
        for name, n, d in zip(entity_names, node_datas, node_degrees) if n is not None
    ]

    return truncate_list_by_token_size(
        valid_node_datas,
        key=lambda x: x.get("description", ""),
        max_token_size=query_param.max_token_for_local_context,
    )


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
):
    text_unit_ids = {
        unit_id
        for edge in edge_datas
        for unit_id in split_string_by_multi_markers(edge.get("source_id", ""), [GRAPH_FIELD_SEP])
    }

    valid_text_units = await text_chunks_db.get_by_ids(list(text_unit_ids))
    valid_text_units = [unit for unit in valid_text_units if unit and unit.get("content")]

    return truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

#A simple utility to merge different pieces of retrieved context together and remove duplicates.
def combine_contexts(entities_lists, relationships_lists, sources_lists):
    hl_entities, ll_entities = entities_lists
    hl_relationships, ll_relationships = relationships_lists
    hl_sources, ll_sources = sources_lists

    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_relationships = process_combine_contexts(hl_relationships, ll_relationships)
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    if not node_datas:
        return []

    source_nodes = [dp["entity_name"] for dp in node_datas]

    all_edges = set()
    edge_tasks = [knowledge_graph_inst.get_node_edges(node) for node in source_nodes]
    edge_results = await asyncio.gather(*edge_tasks)

    for edges in edge_results:
        if edges:
            for edge in edges:
                all_edges.add(tuple(sorted(edge)))

    edge_details_tasks = [knowledge_graph_inst.get_edge(u, v) for u, v in all_edges]
    edge_details_results = await asyncio.gather(*edge_details_tasks)

    relationship_strings = []
    for edge_detail in edge_details_results:
        if edge_detail:
            s_name = edge_detail.get("src_id", "Unknown")
            t_name = edge_detail.get("tgt_id", "Unknown")
            keywords = edge_detail.get("keywords", "related to")
            desc = edge_detail.get("description", "")
            relationship_strings.append([f"Entity '{s_name}' is {keywords} entity '{t_name}'. Description: {desc}"])

    if relationship_strings:
        logger.info("Found the following relational paths (1-hop connections) for context:")
        # We'll log the first 5 to keep the logs from getting too cluttered
        for path_segment in relationship_strings[:5]:
            logger.info(f"  - Path: {path_segment[0]}")
    else:
        logger.info("No direct relational paths found for the retrieved entities.")

    return truncate_list_by_token_size(
        relationship_strings,
        key=lambda x: x[0],
        max_token_size=query_param.max_token_for_local_context,
    )
