#The purpose of this file is to define a single, easy-to-use Python class (PathRAG) that brings together all the complex pieces of the system
#(language models, embedding models, databases, prompts, etc.) into one object.
#It manages the configuration and orchestrates the two main workflows: indexing new information and querying (answering questions).
import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .llm import ollama_turbo_complete, sentence_transformer_embedding #The AI models
from .operate import chunking_by_token_size, extract_entities, kg_query #The heavy-lifting functions
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
) #The helper functions
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
) #he database "blueprints"
from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
) #database implementations

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """Ensures that there is always an event loop available."""
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop
    except RuntimeError:
        logger.info("Creating a new event loop in the main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop

#The entire class is defined as a @dataclass. This is a modern Python feature that makes it incredibly easy to 
#define classes that are primarily used for storing data and configuration. It automatically handles the __init__ method and other boilerplate.
@dataclass
class PathRAG:
    working_dir: str = field(
        default_factory=lambda: f"./PathRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    #which database code to use from storage.py
    kv_storage: str = field(default="JsonKVStorage") 
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    log_level: str = field(default="INFO") #Sets the detail level for the log file. "INFO" is good for general progress; "DEBUG" would be for intense troubleshooting.

    chunk_token_size: int = 512
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "cl100k_base" #a stable, universal choice.

    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    node_embedding_algorithm: str = "node2vec" #feature to create embeddings from the graph structure itself. node2vec is a famous algorithm for this.
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 384, #384 to exactly match the output dimension of our sentence-transformers embedding model. 
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 42,
        }
    )

    embedding_func: EmbeddingFunc = field(default_factory=lambda: sentence_transformer_embedding) #model to use for converting text to numbers(embeddings)
# sets the default function for creating embeddings. The default_factory=lambda: ... syntax is used because we are assigning the function itself, not the result of calling it.
    embedding_batch_num: int = 32 #process texts in batches of 32 and to run a maximum of 16 of these "jobs" in parallel.
    embedding_func_max_async: int = 16

    llm_model_func: callable = ollama_turbo_complete
    llm_model_name: str = "gpt-oss:120b"    #language model to use for thinking and answering
    llm_model_max_token_size: int = 4096  #context limit
    llm_model_max_async: int = 16 #number of parallel calls to the LLM to 16
    llm_model_kwargs: dict = field(default_factory=dict) #A flexible "pass-through" dictionary that lets you add extra, custom parameters to the LLM call without changing the code.

    enable_llm_cache: bool = True
    convert_response_to_json_func: callable = convert_response_to_json #Points to the helper function in utils.py that is responsible for safely extracting a JSON object from the LLM's text output.
    addon_params: dict = field(default_factory=dict) #It's a flexible dictionary that allows us to pass extra parameters (like the list of entity types for extraction) deep into the application's logic.


    #The __post_init__ method is a special dataclass method that runs immediately after 
    #the object is created. We use it to set up all the different storage components (the knowledge graph, the vector databases, etc.).
    
    def __post_init__(self):
        #Sets up Logging: Initializes the system logger to save information to PathRAG.log.
        log_file = os.path.join("PathRAG.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        
#Selects Storage Classes: It looks at the kv_storage and other string fields and selects the correct 
#Python class from the dictionary provided by _get_storage_class(). For example, it chooses the NanoVectorDBStorage class.
        storage_classes = self._get_storage_class()
        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = storage_classes[self.kv_storage]
        self.vector_db_storage_cls: Type[BaseVectorStorage] = storage_classes[self.vector_storage]
        self.graph_storage_cls: Type[BaseGraphStorage] = storage_classes[self.graph_storage]

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

#Initializes All Databases: This is the most important step.
#It creates an instance of each storage class, effectively "powering on" all databases.
#For example, self.entities_vdb = ... creates the vector database for entities and loads any existing data from its corresponding .json file. 

#This is where the assembly line actually builds the parts using the blueprints.
#*   `self.embedding_func = limit_async_func_call(...)`: This takes embedding model function and wraps it with a "limiter" from `utils.py`.
#This prevents the application from trying to create too many embeddings at the same time, which would crash it. It ensures a maximum of 16 concurrent embedding tasks.
#*   `self.llm_response_cache = self.key_string_value_json_storage_cls(...)`: This line uses the blueprint we selected earlier (`JsonKVStorage`) and calls it to create an **actual object**. It passes in configuration details like the `namespace` ("llm_response_cache"), which determines the filename (`kv_store_llm_response_cache.json`). When this object is created, its own `__post_init__` method runs, and it **loads any existing data from that file**.
#*   The next six lines do the exact same thing for every other database the system needs, creating a separate object for each and loading its data from disk. After this block, your `PathRAG` object is fully equipped with live, data-loaded database connections.

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)
        
        self.llm_response_cache = self.key_string_value_json_storage_cls(
            namespace="llm_response_cache", global_config=asdict(self), embedding_func=None
        ) if self.enable_llm_cache else None

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self), embedding_func=self.embedding_func
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self), embedding_func=self.embedding_func
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self), embedding_func=self.embedding_func
        )
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities", global_config=asdict(self), embedding_func=self.embedding_func, meta_fields={"entity_name"}
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships", global_config=asdict(self), embedding_func=self.embedding_func, meta_fields={"src_id", "tgt_id"}
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks", global_config=asdict(self), embedding_func=self.embedding_func
        )

#Prepares the LLM: It wraps the llm_model_func with a cache and other parameters, making it ready for use.
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)( 
#Just like with the embedding function,this wraps the LLM function to ensure we don't make more than 16 parallel calls to the AI, which would exhaust memory.
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
#partial(...): This is a clever Python tool. It creates a new, pre-configured version of hf_model_complete function.
#It "bakes in" the hashing_kv=self.llm_response_cache argument. This means that 
#later on, when other parts of the code call the LLM, they don't need to manually pass the cache database every time; it's already included.
        )

    def _get_storage_class(self) -> dict:
        return {
            "JsonKVStorage": JsonKVStorage,
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "NetworkXStorage": NetworkXStorage,
        }

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

# ainsert
# This is the workflow that runs when you tell the system to learn new information (e.g., when you click "Index PDF").
# The Workflow:
# It takes the raw text from the PDF.
# It calls chunking_by_token_size (from operate.py) to break the text into small, manageable pieces.
# It calls self.chunks_vdb.upsert(), which uses the sentence-transformer to create embeddings for these chunks and saves them.
# It then calls the most complex function, extract_entities (from operate.py), which uses the LLM to read all the chunks and build the knowledge graph.
# Finally, it calls _on_write_complete() to save all the new data (the graph, the chunks, the embeddings) to the files in the working_dir.

    #ainsert takes new text and runs the entire indexing pipeline.
    async def ainsert(self, string_or_strings): #This defines an asynchronous function named ainsert. The async keyword means this function can perform long-running tasks (like calling an AI model or writing to a file)
        #without freezing the entire application. The a prefix is a common convention for asynchronous versions of functions.
#string_or_strings: This is the input, which is expected to be either a single giant string of text (from one PDF) or a list of strings (if you were indexing multiple documents at once).
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
#Purpose: This makes the code flexible. It checks if the input is just a single string.
#Action: If it is, it wraps it in a list []. This ensures that the rest of the function can always assume it's working with a list of documents, even if there's only one.         

        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
        }#Purpose: This block prepares the raw document text for storage.
# for c in string_or_strings: It loops through the list of documents.
# c.strip(): It removes any leading or trailing whitespace from the document text.
# compute_mdhash_id(...): This is a helper function from utils.py. It takes the entire document's text and calculates a unique, consistent ID for it (an MD5 hash).
# This ID acts like a fingerprint. Using a hash ensures that if you try to index the exact same document again, it will have the same ID.
# "doc-": This prefix is added to the ID to make it clear that this is a document ID.
# {"content": c.strip()}: It creates a simple dictionary to hold the document's content.
# new_docs = {...}: The final result is a dictionary where keys are the unique document IDs and values are dictionaries containing the content. Example: {'doc-abc123...': {'content': 'The full text of the PDF...'}}.
               
        # 1. Chunk the text
        
        inserting_chunks = {}
        for doc_key, doc in new_docs.items():
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {**dp, "full_doc_id": doc_key}
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
            }
            inserting_chunks.update(chunks)
#inserting_chunks = {}: Initializes an empty dictionary to hold all the small text chunks we're about to create.
# for doc_key, doc in new_docs.items(): It loops through the documents we just prepared.
# chunking_by_token_size(...): This is the crucial call to the function in operate.py. 
# It takes the full document content (doc["content"]) and all the chunking parameters from our configuration (self.chunk_token_size, etc.) and returns a list of small chunk dictionaries.
# for dp in ...: It loops through each chunk dictionary returned by chunking_by_token_size.
# compute_mdhash_id(dp["content"], prefix="chunk-"): Just like with the full document, it creates a unique, content-based ID for every single chunk.
# {**dp, "full_doc_id": doc_key}: This is a clever way to create a new dictionary. It takes all the key-value pairs from the original chunk dictionary (dp) and adds one new key, full_doc_id, which links the chunk back to the full document it came from.
# inserting_chunks.update(chunks): It adds all the newly created chunks for this document to the main inserting_chunks dictionary. After this loop, inserting_chunks holds all the chunks from all the documents.
        if not inserting_chunks:
            logger.warning("No new chunks were generated from the input.")
            return
            
        logger.info(f"Generated {len(inserting_chunks)} chunks for indexing.")
        
                # 2. Create embeddings for chunks
        
        await self.chunks_vdb.upsert(inserting_chunks) #calls the upsert method of our NanoVectorDBStorage object (self.chunks_vdb). Inside that method (in storage.py), the sentence-transformer model is called to create an embedding vector for every single chunk, and then both the text and the vector are saved to the vdb_chunks.json file.

        logger.info("Extracting entities and relationships from chunks...")

                # 3. Extract the knowledge graph

        self.chunk_entity_relation_graph = await extract_entities(
            inserting_chunks,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            global_config=asdict(self),
        )
#await extract_entities(...): This is the slowest and most important step of the entire indexing process. It calls the main knowledge extraction function from operate.py.
# It passes all the necessary components: the dictionary of chunks (inserting_chunks), the current knowledge graph object (self.chunk_entity_relation_graph), the entity and relationship vector databases (self.entities_vdb, self.relationships_vdb), and the global configuration.
# The extract_entities function then performs its long, resilient batching process, using the LLM to analyze the chunks and populate the graph and other databases.
# self.chunk_entity_relation_graph = ...: When extract_entities is finished, it returns the final, updated knowledge graph object, which we save back to our self.
                         # 4. Save everything to disk
        
        await self.full_docs.upsert(new_docs) #This saves the raw text of the full document into the kv_store_full_docs.json file.
        await self.text_chunks.upsert(inserting_chunks) #This saves the raw text of all the individual chunks into the kv_store_text_chunks.json file.
        await self._on_write_complete() #This is a final "cleanup" function. It calls the index_done_callback() method on all the storage objects. For our file-based system, this is the command that explicitly tells each database object to write its in-memory data to its corresponding file on disk, ensuring all progress is saved permanently.

    async def _on_write_complete(self): 
        storages_to_save = [
            self.full_docs, self.text_chunks, self.llm_response_cache,
            self.entities_vdb, self.relationships_vdb, self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]
        tasks = [
            s.index_done_callback() for s in storages_to_save if s and hasattr(s, 'index_done_callback')
        ]
        await asyncio.gather(*tasks)
        logger.info("All storage backends have been saved.")

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))
        
#aquery        
# This is the workflow that runs when a user asks a question.
# The Workflow:
# It takes the user's question string.
# It calls the main kg_query function (from operate.py).
# It passes all the assembled components to that function: the knowledge graph (self.chunk_entity_relation_graph), the entity database (self.entities_vdb),
#the relationship database, etc.
# The kg_query function then performs the complex RAG pipeline (keyword extraction, retrieval, context building, final generation) and returns the final text answer.
# This aquery method simply receives that final answer and passes it back to the user.    

    # aquery takes a user's question and runs the entire retrieval and generation pipeline.
    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "hybrid": #Purpose:This line checks the `mode` parameter from the `QueryParam` object.This application is designed to support different query strategies. Right now, `"hybrid"` is the only one we've fully implemented.
#Action: If the mode is "hybrid," it proceeds to the main query logic.
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                self.chunks_vdb,
                param,
                asdict(self),
            )
#Purpose: This is the most important line in the function. It delegates all the hard work to the main query engine, kg_query, which lives in operate.py.
# await kg_query(...): It calls the kg_query function and waits for it to complete. This is where the entire RAG pipeline (keyword extraction, retrieval, context building, final generation) happens.
# Arguments Passed: It carefully passes all the necessary "tools" and "data" to the kg_query function:
# query: The user's original question.
# self.chunk_entity_relation_graph: The live knowledge graph object.
# self.entities_vdb, self.relationships_vdb, self.chunks_vdb: The three live vector database objects.
# self.text_chunks: The key-value store for retrieving the raw text of chunks.
# param: The query parameters object, telling the function how much context to retrieve.
# asdict(self): This is a helper that converts the entire PathRAG configuration (all the @dataclass fields) into a simple dictionary. This gives the kg_query function easy access to all the application's settings (like llm_model_name, prompt templates, etc.).
# response = ...: When kg_query is finished, it returns the final, human-readable answer as a string. This string is stored in the response variable.
        else:
            raise ValueError(f"Unsupported query mode: {param.mode}")
#This is a safety check. If a user somehow managed to ask for a query mode that doesn't exist (e.g., "graph_only"), this code would run.        
#This block is for saving any new entries to the LLM response cache
        if self.llm_response_cache: #checks if the cache is actually enabled in the configuration.
            await self.llm_response_cache.index_done_callback()
#If the cache is enabled, this line calls the "save to disk" method for the cache database. This ensures that if the same question is asked again in the future, the answer can be retrieved instantly from the kv_store_llm_response_cache.json file.           
        return response