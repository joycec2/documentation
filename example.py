import datetime
import os
import argparse
import re
os.environ["HF_HOME"] = "model/"

from pathlib import Path
from typing import Dict

import ray
import mysql.connector
from langchain_community.llms import LlamaCpp, Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne

from schema_objects import table_schema_objs
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent
)
from typing import List
from llama_index.llms.ollama import Ollama
from llama_index.llms.vllm import Vllm
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema
)
from llama_index.core import VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader
from llama_index.core.schema import BaseNode
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.core.service_context import ServiceContext
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.llms import ChatResponse
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from sqlalchemy import (
    create_engine,
    MetaData,
    text
)

from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.core.objects import ObjectIndex
from utils import connect_to_DB, execute_query

class QueryExecutor:
    def __init__(self, llm_type: str, db_config):
        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.db_config = db_config

        # Connect to database
        self.engine = create_engine(f"mysql://{db_config['user']}:{db_config['password']}@localhost:3306/{db_config['database']}")
        self.sql_database = SQLDatabase(self.engine)
        print("[INFO] ENGINE DIALECT: ", self.engine.dialect.name)

        # # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Query Executor")
        parser.add_argument("--llm", type=str, default="ollama", choices=["vllm", "ollama"],
                            help="Choose the LLM model (vllm or ollama)")
        args = parser.parse_args()

        if args.llm is None:
            args.llm = llm_type
        
        print(args.llm)

        if args.llm == "vllm":
            self.llm = Vllm(
                model="microsoft/Orca-2-7b",
                dtype="float16",
                tensor_parallel_size=1,
                temperature=0,
                max_new_tokens=100,
                vllm_kwargs={
                    "swap_space": 0,
                    "gpu_memory_utilization": 0.5,
                    "max_model_len": 4096,
                },
            )
        else:
            self.llm = Ollama(model="phi3", request_timeout=500000)
        
        # Initialize LLM model settings
        Settings.llm = self.llm
        Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large", embed_batch_size=512)
        
        self.table_node_mapping = SQLTableNodeMapping(self.sql_database)

        if not Path("indexes").exists():
            # Get table node mappings

            self.obj_index = ObjectIndex.from_objects(
                table_schema_objs,
                self.table_node_mapping,
                VectorStoreIndex,
            )
            self.obj_index.persist(persist_dir="indexes")
        else:
            print("[INFO] Found existing indexes. Loading from disk...\nNote: Delete the indexes directory to create new indexes\n")
            storage_context = StorageContext.from_defaults(persist_dir="indexes")
            # load index
            self.obj_index = ObjectIndex.from_persist_dir("indexes", self.table_node_mapping)

        # 'similarity_top_k' sets the closest number of tables to the user query.
        # Reducing this would increase accuracy given that the CORRECT related tables are being retrieved!
        self.obj_retriever = self.obj_index.as_retriever(similarity_top_k=5)
        self.sql_retriever = SQLRetriever(self.sql_database)

        self.SQLQuery = ""

    def remove_collections(self, chroma_client):
        """
        Remove all the collections present in ChromaDB

        Arguments:
            chroma_client -> The chromaDB client
        """
        collections = chroma_client.list_collections()
        collection_names = [collection.name for collection in collections]
        for collection_name in collection_names:
            chroma_client.delete_collection(collection_name)


    def index_all_tables(self, table_index_dir: str = "table_index_dir") -> Dict[str, VectorStoreIndex]:
        """
        Index all the tables in the database

        Arguments:
            table_index_dir -> Directory name for storing table indexes locally
        """
        if not Path(table_index_dir).exists():
            os.makedirs(table_index_dir)

        vector_index_dict = {}
        engine = self.sql_database.engine
        print(f"[LOG] Started indexing at {datetime.datetime.now()}")

        chroma_client = chromadb.PersistentClient()

        for table_name in self.sql_database.get_usable_table_names():
            print(f"[LOG] Indexing rows in table: {table_name}")

            # if not os.path.exists(f"{table_index_dir}/{table_name}"):
            if f"{table_name}" not in [c.name for c in chroma_client.list_collections()]:
                chroma_collection = chroma_client.create_collection(name=f"{table_name}")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

                # get all rows from table
                print(f"[INFO] Cannot find index directory for table [{table_name}]")
                with engine.connect() as conn:
                    cursor = conn.execute(text(f'SELECT * FROM {table_name} LIMIT 3;'))
                    result = cursor.fetchall()
                    row_tups = []
                    for row in result:
                        row_tups.append(tuple(row))

                # index each row, put into vector store index
                nodes = [TextNode(text=str(t)) for t in row_tups]

                self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # put into vector store index (use OpenAIEmbeddings by default)
                print(f"[INFO] Writing indexes to chromaDB for {table_name}")
                index = VectorStoreIndex(nodes, service_context=self.service_context, storage_context=self.storage_context)
            else:
                print(f"[INFO] Found existing indexes in chromaDB..")

                chroma_collection = chroma_client.get_collection(name=f"{table_name}")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
 
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store
                )

            vector_index_dict[table_name] = index

        return vector_index_dict

    def get_table_context_and_rows_str(
        self, query_str: str, table_schema_objs: List[SQLTableSchema]
    ):
        """
        Get table context string with 'k' rows. The 'k' variable can be set using 'similarity_top_k' argument

        Arguments:
            query_str -> User query
            table_schema_objs -> A list containing the table_name and context_str for each table in the database
        """
        print(f"\n<-------------[MATCHED_TABLES]-------------->")
        for table_schema_obj in table_schema_objs:
            print(f"[INFO] {table_schema_obj.table_name}")
        print("\n\n")
        context_strs = []
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        
        for table_schema_obj in table_schema_objs:
            table = metadata.tables[table_schema_obj.table_name]
            columns = table.columns

            table_info = ""
            table_info += f"CREATE TABLE `{table_schema_obj.table_name}` ("
            for column in columns:
                # Format column information
                column_info = f"`{column.name}` {column.type}"
                # Add information about constraints (e.g., primary key, not null)
                if column.primary_key:
                    column_info += " PRIMARY KEY"
                if not column.nullable:
                    column_info += " NOT NULL"
                table_info += f"{column_info}, "
            table_info += ")\n"

            print(f"[SCHEMA_INFO] {table_schema_obj.table_name}: {table_info}")

            if table_schema_obj.context_str:
                # table_opt_context = " The table description is: "
                table_opt_context = table_schema_obj.context_str
                table_info += table_opt_context

            # also lookup vector index to return relevant table rows
            relevant_nodes = self.vector_index_dict[
                table_schema_obj.table_name
            ].as_retriever(similarity_top_k=5).retrieve(query_str)
            if len(relevant_nodes) > 0:
                print(f"[LOG] relevant_nodes: {relevant_nodes[0]}")
                table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
                for node in relevant_nodes:
                    table_row_context += str(node.get_content()) + "\n"
                table_info += table_row_context

            context_strs.append(table_info)
        return "\n\n".join(context_strs)

    def get_table_context_str(self, table_schema_objs: List[SQLTableSchema]):
        """
        Get table context string without any rows

        Arguments:
            table_schema_objs -> A list containing the table_name and context_str for each table in the database
        """
        context_strs = []
        for table_schema_obj in table_schema_objs:
            table_info = self.sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            context_strs.append(table_info)
        return "\n\n".join(context_strs)

    def parse_response_to_sql(self, response: ChatResponse) -> str:
        """
        Parse the response from the LLM to get the SQL query

        Arguments:
            response -> an Object of type 'ChatResponse' containing the response from the LLM
        """
        response = response.message.content
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        # Modified --> introduced SQLQuery variable
        self.SQLQuery = response.strip("\n\t ").replace("\n", " ").strip("```").strip()
        if self.SQLQuery.startswith("sql "):
            self.SQLQuery = self.SQLQuery[4:]
        pattern = r';.*'
        self.SQLQuery = re.sub(pattern, ';', self.SQLQuery)

    def run_query(self, query: str):
        """
        Run the user query against the Query Pipeline.

        Arguments:
            query -> user query
        """
        print(f"[LOG] Started processing at {datetime.datetime.now()}")
        response = self.qp.run(query=query)
        db_config = self.db_config
        query = self.SQLQuery
        connector = connect_to_DB(db_config)
        data = execute_query(connection=connector, query=query)
        return self.SQLQuery, response, data

    def setup_query_pipeline(self):
        """
        Setup the Query Pipeline
        """
        # Define functional components
        table_parser_component = FnComponent(fn=self.get_table_context_and_rows_str)
        sql_parser_component = FnComponent(fn=self.parse_response_to_sql)

        text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
            dialect=self.engine.dialect.name
        )

        response_synthesis_prompt_str = (
            "Given an input question, synthesize a response from the query results. Only create a response from the SQL response itself and do not make up any information.\n"
            "Query: {query_str}\n"
            "SQL: {sql_query}\n"
            "SQL Response: {context_str}\n"
            "Response: "
        )
        response_synthesis_prompt = PromptTemplate(
            response_synthesis_prompt_str,
        )

        # Define Query Pipeline
        self.qp = QP(
            modules={
                "input": InputComponent(),
                "table_retriever": self.obj_retriever,
                "table_output_parser": table_parser_component,
                "text2sql_prompt": text2sql_prompt,
                "text2sql_llm": self.llm,
                "sql_output_parser": sql_parser_component,
                "sql_retriever": self.sql_retriever,
                "response_synthesis_prompt": response_synthesis_prompt,
                "response_synthesis_llm": self.llm
            },
            verbose=True
        )

        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=Settings.embed_model, callback_manager=self.qp.callback_manager)
        self.vector_index_dict = self.index_all_tables()

        # Add Chains and Links between modules
        self.qp.add_link("input", "table_retriever")
        self.qp.add_link("input", "table_output_parser", dest_key="query_str")
        self.qp.add_link(
            "table_retriever", "table_output_parser", dest_key="table_schema_objs"
        )
        self.qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        self.qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
        self.qp.add_chain(
            ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
        )
        self.qp.add_link(
            "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
        )
        self.qp.add_link(
            "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
        )
        self.qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        self.qp.add_link("response_synthesis_prompt", "response_synthesis_llm")