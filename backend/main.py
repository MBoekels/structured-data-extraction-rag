import logging
import json
import numpy as np
import os
from pathlib import Path
import subprocess
import time
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config.settings import settings
from .database import DatabaseManager
from .llm_services import get_embedding_client, get_llm
from .vector_store import VectorStoreManager
from .processing import (
    parse_document_with_docling,
    chunk_document,
    format_context_from_evaluated_chunks,
    save_docling_document,
    load_docling_document,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# --- Qdrant Docker Management ---
QDRANT_CONTAINER_NAME = "qdrant-rag-app"
qdrant_started_by_app = False

def start_qdrant():
    """Starts the Qdrant Docker container if not already running."""
    global qdrant_started_by_app
    try:
        # Check if Docker is running
        subprocess.run(["docker", "info"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker is not running or not installed. Please start Docker and try again.")
        return

    try:
        # Check if Qdrant is already running by checking the container
        container_id = subprocess.check_output(["docker", "ps", "-q", "-f", f"name={QDRANT_CONTAINER_NAME}"]).decode().strip()
        if container_id:
            logger.info(f"Qdrant container '{QDRANT_CONTAINER_NAME}' is already running.")
            return

        # Check if the container exists but is stopped
        container_id = subprocess.check_output(["docker", "ps", "-aq", "-f", f"name={QDRANT_CONTAINER_NAME}"]).decode().strip()
        if container_id:
            logger.info(f"Starting existing Qdrant container '{QDRANT_CONTAINER_NAME}'...")
            subprocess.run(["docker", "start", QDRANT_CONTAINER_NAME], check=True)
            qdrant_started_by_app = True
        else:
            logger.info(f"Creating and starting new Qdrant container '{QDRANT_CONTAINER_NAME}'...")
            subprocess.run([
                "docker", "run", "-d",
                "-p", f"{settings.QDRANT_PORT}:{settings.QDRANT_PORT}",
                "--name", QDRANT_CONTAINER_NAME,
                "qdrant/qdrant"
            ], check=True)
            qdrant_started_by_app = True
        
        # Wait for the container to be ready
        time.sleep(5)
        logger.info("Qdrant container started successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Qdrant container: {e.stderr.decode()}")
    except FileNotFoundError:
        logger.error("Could not find 'docker' executable. Please ensure Docker is installed and in your PATH.")

def stop_qdrant():
    """Stops the Qdrant Docker container if it was started by this application."""
    if qdrant_started_by_app:
        logger.info(f"Stopping Qdrant container '{QDRANT_CONTAINER_NAME}'...")
        try:
            subprocess.run(["docker", "stop", QDRANT_CONTAINER_NAME], check=True, capture_output=True)
            logger.info(f"Qdrant container '{QDRANT_CONTAINER_NAME}' stopped successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop Qdrant container: {e.stderr.decode()}")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Application",
    description="An application for retrieving information from documents using RAG.",
    # on_startup=[start_qdrant],
    # on_shutdown=[stop_qdrant],
)

# --- Globals ---
# It's better to initialize these once and reuse them.
db_manager = DatabaseManager()
embedding_client = get_embedding_client()
logger.info(f"Embedding model initialized: {embedding_client}")
vector_store_manager = VectorStoreManager(embedding_client=embedding_client)

def _extract_json_from_llm_output(text: str) -> dict | None:
    """Extracts a JSON object from a string, handling markdown code blocks and raw JSON."""
    start_tag = "```json"
    end_tag = "```"
    json_str = None

    # Try to extract from markdown code block first
    start_index = text.find(start_tag)
    if start_index != -1:
        content_start_index = start_index + len(start_tag)
        end_index = text.rfind(end_tag, content_start_index) # Ensure end_tag is after start_tag
        if end_index != -1:
            json_str = text[content_start_index:end_index].strip()
        else:
            # If start tag found but no end tag, assume rest of string is JSON
            json_str = text[content_start_index:].strip()
    else:
        # If no markdown block, try to find the first JSON object by brace matching
        brace_level = 0
        json_start = -1
        for i, char in enumerate(text):
            if char == '{':
                if json_start == -1:
                    json_start = i
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and json_start != -1:
                    # Found a potential JSON object
                    potential_json_str = text[json_start : i + 1]
                    try:
                        return json.loads(potential_json_str)
                    except json.JSONDecodeError:
                        # Not a valid JSON, continue searching
                        pass
        # If no valid JSON object found by brace matching, try to parse the whole string
        json_str = text.strip()

    # If json_str was extracted from markdown, try to parse it
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from string: {json_str}")
            return None
    return None # No JSON found or parsed

@app.post("/debug/reset-database", summary="Reset the entire database")
async def reset_database_endpoint():
    """
    (Debug) Drops all tables and recreates them. This is a destructive operation.
    """
    try:
        db_manager.reset_database()
        return JSONResponse(
            status_code=200,
            content={"message": "Database has been successfully reset."}
        )
    except Exception as e:
        logger.error(f"Error resetting database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/company/", summary="Create a new company profile")
async def create_company(company_name: str = Form(...), description: str = Form(None)):
    """
    Creates a new company profile in the database.
    This is a necessary first step before ingesting documents for a company.
    """
    try:
        company_id = db_manager.get_or_create_company(company_name, description)
        return JSONResponse(
            status_code=201,
            content={
                "message": "Company created successfully.",
                "company_id": company_id,
                "company_name": company_name,
                "description": description,
            },
        )
    except Exception as e:
        logger.error(f"Error creating company: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_all_companies():
    """Retrieve all companies."""
    try:
        companies = db_manager.get_all_companies()
        return JSONResponse(status_code=200, content=companies)
    except Exception as e:
        logger.error(f"Error getting companies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/company/", summary="Get all companies")
async def get_all_companies_route():
    return await get_all_companies()

@app.get("/company/{company_id}", summary="Get a company by ID")
async def get_company(company_id: int):
    """Retrieve a single company by its ID."""
    try:
        company = db_manager.get_company_by_id(company_id)
        if company is None:
            raise HTTPException(status_code=404, detail="Company not found")
        return JSONResponse(status_code=200, content=company)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting company: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _ingest_pipeline_sync(company_id: int, file: UploadFile):
    start_time = time.time()
    # 1. Save file to a persistent location
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    logging.info(f"File '{file.filename}' uploaded to '{file_path}'.")

    pdf_id = None  # Initialize pdf_id to handle potential errors before its assignment
    docling_df = None

    try:
        # 2.5 Check for existing PDF and its status
        existing_pdf_info = db_manager.get_pdf_info_by_name(company_id, file.filename)

        if existing_pdf_info:
            logger.info(f"PDF '{file.filename}' has already been successfully ingested. Loading from cache directly.")
            pdf_id = existing_pdf_info['id'] # Use existing pdf_id

            try:
                # Load existing DoclingDocument
                docling_df = load_docling_document(existing_pdf_info['docling_json_path'])
                # TODO Make sure that if load_docling_document was unsuccessful, that there is an error thrown here!
            except:
                # If DoclingDocument cannot be retrieved from cache
                # Clean up the old record before creating a new one.
                logger.info(f"Found previous incomplete ingestion for '{file.filename}'. Cleaning up and re-ingesting.")
                db_manager.delete_pdf_info(existing_pdf_info['id'])
                # Clean up any old chunks from the vector store before re-ingesting
                vector_store_manager.delete_chunks_by_pdf_id(company_id, pdf_id)
        
                # 3. Process document with Docling
                t1 = time.time()
                docling_df = parse_document_with_docling(str(file_path))
                logger.info(f"Docling parsing took {time.time() - t1:.2f} seconds.")

                # Store the original filename for reference, not the temp path
                pdf_id = db_manager.save_pdf_info(company_id, file.filename) # Save new record for re-ingested PDF

                # 3.5 Save the parsed document to a file and update the DB
                docling_json_path = save_docling_document(docling_df, pdf_id)
                db_manager.update_pdf_docling_path(pdf_id, docling_json_path)

        else:
            # Store the original filename for reference, not the temp path
            pdf_id = db_manager.save_pdf_info(company_id, file.filename)

            # 3. Process document with Docling
            t1 = time.time()
            docling_df = parse_document_with_docling(str(file_path))
            logger.info(f"Docling parsing took {time.time() - t1:.2f} seconds.")

            # 3.5 Save the parsed document to a file and update the DB
            docling_json_path = save_docling_document(docling_df, pdf_id)
            db_manager.update_pdf_docling_path(pdf_id, docling_json_path)

        # 4. Chunk the document
        t1 = time.time()
        chunks_df = chunk_document(docling_df)
        logger.info(f"Document chunking took {time.time() - t1:.2f} seconds.")

        # 5. Ingest chunks into vector store
        t1 = time.time()
        vector_store_manager.ingest_chunks(chunks_df, company_id, pdf_id)
        logger.info(f"Vector store ingestion took {time.time() - t1:.2f} seconds.")

        # 6. Mark ingestion as successful
        db_manager.update_pdf_status(pdf_id, 'SUCCESS')
        logger.info(f"Successfully ingested and marked PDF id {pdf_id} as SUCCESS. Total time: {time.time() - start_time:.2f} seconds.")
        return {
            "message": "Document ingested successfully.",
            "company_id": company_id,
            "pdf_id": pdf_id,
            "num_chunks": len(chunks_df),
        }
    except HTTPException as e:
        # If an error occurs after the PDF record is created, mark it as FAILED.
        if pdf_id:
            db_manager.update_pdf_status(pdf_id, 'FAILED')
            logger.error(f"Ingestion failed for PDF id {pdf_id} during an HTTP exception. Marked as FAILED.")
        raise e # Re-raise the HTTP exception
    except Exception as e:
        if pdf_id:
            db_manager.update_pdf_status(pdf_id, 'FAILED')
            logger.error(f"Ingestion failed for PDF id {pdf_id}. Marked as FAILED.")
        logging.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/", summary="Ingest a PDF for a company")
async def ingest_pipeline(
    company_id: int = Form(...),
    file: UploadFile = File(...)
):
    """
    Ingestion pipeline to process and store a document.
    """
    try:
        result = await asyncio.to_thread(_ingest_pipeline_sync, company_id, file)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/company/{company_id}/pdfs/", summary="Get all PDFs for a company")
async def get_all_pdfs_for_company(company_id: int):
    """Retrieve all PDFs for a given company."""
    try:
        pdfs = db_manager.get_all_pdfs_for_company(company_id)
        return JSONResponse(status_code=200, content=pdfs)
    except Exception as e:
        logger.error(f"Error getting PDFs for company: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdfs/{pdf_id}", summary="Get a PDF by ID")
async def get_pdf(pdf_id: int):
    """Retrieve a single PDF by its ID."""
    try:
        pdf = db_manager.get_pdf_by_id(pdf_id)
        if pdf is None:
            raise HTTPException(status_code=404, detail="PDF not found")
        return JSONResponse(status_code=200, content=pdf)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _initiate_query_pipeline_sync(query_text: str):
    start_time = time.time()
    try:
        # --- Part 1: Generate JSON schema for the answer ---
        logging.info(f"Generating schema for query: '{query_text}'")
        schema_prompt_template = PromptTemplate.from_template(
            "You are a data science expert that only knows how to reply in json format. "
            "Generate a simple and flat JSON schema. Avoid deep nesting unless absolutely necessary. "
            "You generate suitable data formats for the requested inputs. Whenever possible, "
            "you also account for the units of the values in your returned schema.\n\n"
            "Your response must contain ONLY the JSON object, wrapped in ```json and ``` tags. "
            "Based on the following question, I want you to generate a data structure "
            "in json format that allows to store the information that is requested "
            "in a structured format with the respective datatypes.\n\n{user_query}"
        )
        schema_llm = get_llm()
        schema_generation_chain = schema_prompt_template | schema_llm
        t1 = time.time()
        schema_str = schema_generation_chain.invoke({"user_query": query_text}).content
        logger.info(f"Schema generation LLM call took {time.time() - t1:.2f} seconds.")
        
        schema_json = _extract_json_from_llm_output(schema_str)
        if not schema_json:
            raise HTTPException(status_code=500, detail="LLM failed to generate a valid JSON schema.")
        logging.info(f"Generated schema: {schema_json}")

        # --- Part 2: Embed the query text (although not explicitly used here, it's good practice) ---
        # This embedding can be stored for future use, e.g., semantic query caching.
        t1 = time.time()
        embedding_response = embedding_client.embeddings.create(input=query_text, model=settings.EMBEDDING_MODEL_ID)
        logger.info(f"Embedding creation took {time.time() - t1:.2f} seconds.")
        query_embedding = embedding_response.data[0].embedding

        # --- Part 3: Save query, schema, and embedding to the database ---
        t1 = time.time()
        query_id = db_manager.save_query_and_schema(query_text, schema_json, query_embedding)
        logger.info(f"Database save took {time.time() - t1:.2f} seconds.")
        logging.info(f"Query initiated with ID: {query_id}. Total time: {time.time() - start_time:.2f} seconds.")

        return {
            "message": "Query initiated successfully. Ready for RAG analysis.",
            "query_id": query_id,
            "query_text": query_text,
            "generated_schema": schema_json,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error during query initiation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", summary="Initiate a query and generate schema")
async def initiate_query_pipeline(query_text: str = Form(...)):
    """
    Initiates a query by generating a schema and saving it.
    This is the first step and is independent of any company.
    """
    try:
        result = await asyncio.to_thread(_initiate_query_pipeline_sync, query_text)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/", summary="Get all queries")
async def get_all_queries():
    """Retrieve all queries."""
    try:
        queries = db_manager.get_all_queries()
        return JSONResponse(status_code=200, content=queries)
    except Exception as e:
        logger.error(f"Error getting queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/{query_id}", summary="Get a query by ID")
async def get_query(query_id: int):
    """Retrieve a single query by its ID."""
    try:
        query = db_manager.get_query_info(query_id)
        if query is None:
            raise HTTPException(status_code=404, detail="Query not found")
        return JSONResponse(status_code=200, content=query)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _company_rag_analysis_sync(query_id: int, company_id: int):
    start_time = time.time()
    try:
        # --- Part 1: Retrieve query information from the database ---
        query_info = db_manager.get_query_info(query_id)
        if not query_info:
            raise HTTPException(status_code=404, detail=f"Query with ID {query_id} not found.")
        
        query_text = query_info['query_text']
        schema_json = query_info['schema_json']
        
        query_embedding = query_info['query_embedding']
        
        logging.info(f"Performing RAG analysis for query_id {query_id} and company_id {company_id}")

        # --- Part 2: RAG to answer the query ---
        logging.info(f"Searching for relevant documents for company_id {company_id}")
        # 1. Retrieve relevant documents
        t1 = time.time()
        doc_vector_pairs = vector_store_manager.search(query_text, company_id, k=10, query_vector=query_embedding)
        logger.info(f"Vector store search took {time.time() - t1:.2f} seconds.")
        if not doc_vector_pairs:
            raise HTTPException(status_code=404, detail="No relevant documents found for the query.")

        # Initialize LLM for evaluation
        evaluator_llm = get_llm()

        # Prepare initial messages for the evaluation conversation
        messages = [
            SystemMessage(content="You are an expert document evaluator. Your task is to assess the relevance and utility of document chunks for answering a given query based on a specified JSON schema. Your response for each chunk must be a JSON object, wrapped in ```json and ``` tags. For each chunk, provide a JSON object indicating its relevance, potential information extracted, and a confidence score."),
            HumanMessage(content=f"Query: {query_text}\n\nDesired JSON Schema: {json.dumps(schema_json, indent=2)}\n\nNow, I will provide document chunks one by one. For each chunk, tell me if it contains information relevant to the query and schema, and if so, what specific data points it could contribute. Your response for each chunk must be a JSON object with the following keys: \"relevance_score\" (integer 1-5), \"potential_data_points\" (object mapping schema fields to brief descriptions of relevant info), \"confidence\" (string: \"low\", \"medium\", \"high\"), and \"comment\" (string). If a chunk is not relevant, set \"relevance_score\" to 1 and \"potential_data_points\" to an empty object.")
        ]

        evaluated_chunks = []
        total_eval_time = 0
        for doc, vec in doc_vector_pairs:
            chunk_content = doc.page_content
            chunk_metadata = doc.metadata

            # Add the current chunk to the conversation
            messages.append(HumanMessage(content=f"Document Chunk:\n{chunk_content}"))

            # Invoke the LLM with the current conversation history
            t1 = time.time()
            evaluation_response = evaluator_llm.invoke(messages)
            eval_time = time.time() - t1
            total_eval_time += eval_time
            logger.info(f"LLM evaluation for chunk took {eval_time:.2f} seconds.")
            evaluation_result_str = evaluation_response.content

            # Extract JSON evaluation from the LLM's response
            chunk_evaluation_json = _extract_json_from_llm_output(evaluation_result_str)

            if chunk_evaluation_json:
                # Add the LLM's evaluation to the chunk's metadata or a new structure
                evaluated_chunks.append((chunk_evaluation_json, chunk_content, chunk_metadata))
            else:
                logger.warning(f"Failed to get valid LLM evaluation for chunk: {chunk_content[:100]}...")
                # Optionally, you might want to skip this chunk or handle it differently

            # Add the AI's response to the messages for the next turn
            messages.append(AIMessage(content=evaluation_result_str))
        logger.info(f"Total LLM evaluation time for all chunks: {total_eval_time:.2f} seconds.")

        # Now, filter and re-rank chunks based on LLM evaluation
        # Sort evaluated_chunks by relevance_score in descending order
        evaluated_chunks.sort(key=lambda x: x[0].get("relevance_score", 0), reverse=True)

        # Filter documents based on a relevance threshold
        RELEVANCE_THRESHOLD = 3 # Example threshold (1-5 scale)
        relevant_chunks = []
        for evaluation, chunk_content, chunk_metadata in evaluated_chunks:
            if evaluation.get("relevance_score", 0) >= RELEVANCE_THRESHOLD:
                relevant_chunks.append((evaluation, chunk_content, chunk_metadata))

        if not relevant_chunks:
            return {
                "query_id": query_id,
                "company_id": company_id,
                "answer": None,
                "sources": [],
                "status": "INSUFFICIENT_DOCUMENT_CONTEXT",
                "message": "No relevant chunks found for the query after LLM evaluation.",
            }

        # --- Final Answer Generation using Conversational Context ---
        # Add a final instruction to the conversation, asking the LLM to synthesize the answer.
        synthesis_instruction = (
            "Based on all the document chunks you have evaluated in our conversation, "
            "please now provide a final, consolidated answer. Compile all the relevant data points "
            "you identified into the single JSON object we discussed at the beginning. "
            "Pay close attention to the schema and ensure all units are correct. "
            "If information for a field was found in multiple chunks, synthesize it appropriately. "
            "If a field has no information across all chunks, use null. "
            "Your response must contain ONLY the JSON object, wrapped in ```json and ``` tags."
        )
        messages.append(HumanMessage(content=synthesis_instruction))

        # Invoke the LLM one last time to get the final synthesized answer
        logging.info("Invoking LLM to generate final synthesized answer.")
        t1 = time.time()
        final_synthesis_response = evaluator_llm.invoke(messages)
        logger.info(f"Final LLM synthesis call took {time.time() - t1:.2f} seconds.")
        llm_output_str = final_synthesis_response.content
        
        llm_output_json = _extract_json_from_llm_output(llm_output_str)
        if not llm_output_json:
            raise HTTPException(status_code=500, detail="LLM failed to generate a valid JSON response.")
        logging.info(f"LLM output: {llm_output_json}")

        # --- Save the result to the database with ranked sources and final answer ---
        t1 = time.time()
        ranked_sources = []
        for evaluation_json, chunk_content, chunk_metadata in relevant_chunks:
            # Create a new dictionary that combines original metadata and LLM evaluation
            combined_metadata = {
                "original_metadata": chunk_metadata,
                "llm_evaluation": evaluation_json,
                "chunk_content_preview": chunk_content[:200] # Optional: store a preview of the chunk content
            }
            ranked_sources.append(combined_metadata)
        db_manager.save_llm_result(query_id, company_id, llm_output_json, ranked_sources)
        logger.info(f"Database save for results took {time.time() - t1:.2f} seconds.")

        logger.info(f"RAG analysis for query_id {query_id} and company_id {company_id} completed in {time.time() - start_time:.2f} seconds.")
        return {
            "query_id": query_id,
            "company_id": company_id,
            "answer": llm_output_json, # Re-add the final answer
            "sources": ranked_sources, # These are the LLM-evaluated and ranked chunks
            "status": "Success",
            "message": "Results of the RAG Analysis have been written to the database.",
        }
    
    except Exception as e:
        logging.error(f"Error during RAG analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/rag_analysis", summary="Perform RAG analysis for a company")
async def company_rag_analysis(
    query_id: int = Form(...),
    company_id: int = Form(...)
):
    """
    Performs RAG analysis for a specific company using a pre-initiated query.
    """
    try:
        result = await asyncio.to_thread(_company_rag_analysis_sync, query_id, company_id)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _delete_pdf_data(pdf_id: int):
    """Helper function to delete all data associated with a PDF."""
    pdf_info = db_manager.get_pdf_by_id(pdf_id)
    if not pdf_info:
        raise HTTPException(status_code=404, detail=f"PDF with ID {pdf_id} not found.")

    company_id = pdf_info['company_id']
    
    # Delete docling cache file
    docling_path = pdf_info.get('docling_json_path')
    if docling_path and os.path.exists(docling_path):
        os.remove(docling_path)
        logger.info(f"Deleted docling cache file: {docling_path}")

    # Delete chunks from vector store
    vector_store_manager.delete_chunks_by_pdf_id(company_id, pdf_id)
    logger.info(f"Deleted vector store chunks for PDF ID: {pdf_id}")

    # Delete PDF file from uploads
    upload_file_path = Path("uploads") / pdf_info['file_path']
    if os.path.exists(upload_file_path):
        os.remove(upload_file_path)
        logger.info(f"Deleted uploaded file: {upload_file_path}")

    # Delete PDF info from database
    db_manager.delete_pdf_info(pdf_id)
    logger.info(f"Deleted PDF info from database for ID: {pdf_id}")

@app.delete("/pdf/{pdf_id}", summary="Delete a PDF and all its associated data")
async def delete_pdf(pdf_id: int):
    try:
        await _delete_pdf_data(pdf_id)
        return JSONResponse(
            status_code=200,
            content={"message": f"Successfully deleted PDF {pdf_id} and all associated data."}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting PDF {pdf_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/query/{query_id}", summary="Delete a query and all its associated results")
async def delete_query(query_id: int):
    try:
        # Check if query exists
        if not db_manager.get_query_info(query_id):
            raise HTTPException(status_code=404, detail=f"Query with ID {query_id} not found.")

        # Delete associated results
        db_manager.delete_results_by_query_id(query_id)
        logger.info(f"Deleted results for query ID: {query_id}")

        # Delete the query itself
        db_manager.delete_query(query_id)
        logger.info(f"Deleted query with ID: {query_id}")

        return JSONResponse(
            status_code=200,
            content={"message": f"Successfully deleted query {query_id} and all associated results."}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting query {query_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/company/{company_id}/", summary="Delete a company and all its associated data")
async def delete_company(company_id: int):
    try:
        # Check if company exists
        if not db_manager.get_company_by_id(company_id):
            raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found.")

        # Delete all associated PDFs and their data
        pdfs = db_manager.get_all_pdfs_for_company(company_id)
        for pdf in pdfs:
            await _delete_pdf_data(pdf['id'])
        
        # Delete associated results
        db_manager.delete_results_by_company_id(company_id)
        logger.info(f"Deleted results for company ID: {company_id}")

        # Delete the company itself
        db_manager.delete_company(company_id)
        logger.info(f"Deleted company with ID: {company_id}")

        return JSONResponse(
            status_code=200,
            content={"message": f"Successfully deleted company {company_id} and all associated data."}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting company {company_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))