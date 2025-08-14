# Thesis Outline: A Hybrid RAG System with Conversational Chunk Evaluation for Structured Data Extraction from Corporate Reports

## Abstract

*   **Problem Statement:** State the core challenge: the difficulty and inefficiency of manually extracting specific, structured data points from long, unstructured, and semi-structured documents like corporate sustainability and financial reports.
*   **Proposed Solution:** Briefly introduce the design and implementation of a full-stack, Retrieval-Augmented Generation (RAG) application built to address this problem.
*   **Key Innovations & Features:**
    *   Highlight the novel, two-stage conversational RAG pipeline that first uses an LLM to evaluate the relevance of retrieved document chunks and then uses the same conversational context to synthesize a final, structured JSON answer.
    *   Mention the dynamic generation of a target JSON schema from a user's natural language query.
    *   Emphasize the integration of the `docling` library for advanced document layout analysis, enabling robust handling of complex elements like tables.
*   **Results & Conclusion:** Summarize the findings, noting the system's effectiveness in providing accurate, structured data and present a cost-benefit analysis comparing it to manual methods.

---

## 1. Introduction

*   **1.1. Motivation and Problem Statement**
    *   The growing demand for reliable ESG (Environmental, Social, and Governance) and financial data from corporate disclosures.
    *   The inherent challenges of unstructured data in PDF format: varied layouts, dense text, and complex tables.
    *   The limitations of traditional methods: keyword search (lacks semantic understanding) and manual analysis (slow, expensive, error-prone, not scalable).
    *   The potential of Large Language Models (LLMs) and the RAG paradigm as a modern solution.
*   **1.2. Research Questions**
    *   How can a RAG system be architected to reliably extract data conforming to a dynamically generated, structured JSON schema?
    *   How can the quality of retrieved context be programmatically assessed and filtered to improve the accuracy and reduce hallucinations of the final generated answer?
    *   What is the impact of a layout-aware document parsing and chunking strategy, particularly for tables, on the overall performance of a RAG system for data extraction?
    *   How does the operational cost and performance of the developed automated system compare to the traditional approach of manual analysis by a human expert?
*   **1.3. Key Contributions**
    *   The design and implementation of an end-to-end application for structured data extraction.
    *   A novel two-stage conversational RAG process (Evaluate, then Synthesize) to enhance answer quality.
    *   A practical demonstration of integrating a sophisticated document parsing library (`docling`) to overcome common chunking challenges, especially with tables.
    *   A detailed performance and cost-benefit analysis, contextualizing the system's value in a real-world scenario.
*   **1.4. Thesis Structure**
    *   A brief overview of the chapters to follow.

---

## 2. Background and Related Work

*   **2.1. Foundational Concepts in NLP**
    *   From word embeddings to contextual embeddings (Transformer architecture, attention mechanism).
*   **2.2. Large Language Models (LLMs)**
    *   Overview of model families (e.g., GPT, DeepSeek).
    *   The concept of in-context learning as the basis for RAG.
*   **2.3. Retrieval-Augmented Generation (RAG)**
    *   The standard RAG architecture: Indexing, Retrieval, Generation.
    *   How RAG mitigates LLM limitations (knowledge cut-offs, hallucinations).
    *   Brief overview of advanced RAG techniques (e.g., re-ranking, query expansion) and positioning of this work's conversational evaluation method within that landscape.
*   **2.4. Data Storage Paradigms in AI Systems**
    *   **Relational Databases (e.g., SQLite):** Their role in storing structured metadata, application state, and final results. Justification for its use in this project.
    *   **Vector Databases (e.g., Qdrant):** The necessity for a specialized database for efficient similarity search on high-dimensional vectors. Explanation of concepts like collections and distance metrics (e.g., Cosine Similarity).
*   **2.5. Document Layout Analysis and Parsing**
    *   The unique challenges of the PDF format.
    *   Comparison of basic text extraction tools versus layout-aware parsing libraries like `docling`.

---

## 3. System Design and Architecture

*   **3.1. High-Level System Architecture**
    *   **Architecture Map:** A detailed diagram illustrating the interaction between the Frontend (Streamlit), Backend (FastAPI), Databases (SQLite, Qdrant), File Storage, and external LLM/Embedding services.
    *   Description of the main components and the flow of data during both the ingestion and query analysis processes.
*   **3.2. Frontend-Backend Communication Strategy**
    *   **REST API vs. Direct SQL:** A critical discussion of the architectural choice.
        *   **Why direct SQL reads?** Explain the use of direct `sqlite3` connections in the Streamlit frontend for read-only operations (fetching companies, queries, results) to achieve simplicity and low-latency page loads.
        *   **Why a REST API for writes/actions?** Explain the use of the FastAPI backend for all state-changing (create, delete) and computationally intensive operations (ingestion, RAG analysis), promoting decoupling, security, and proper task management.
*   **3.3. Data Model and Schemas**
    *   **SQLite Database Schema:** Detailed breakdown of the `companies`, `pdfs`, `queries`, and `results` tables, including their columns and foreign key relationships.
    *   **Qdrant Vector Schema:** Description of a "point" in a Qdrant collection, detailing the vector and the payload structure (text content and metadata like `pdf_id`, `pages`, `bboxes`).

---

## 4. Implementation Details

*   **4.1. Backend Development with FastAPI**
    *   Overview of the API endpoints and their responsibilities.
    *   **Performance and Concurrency:**
        *   The choice of an asynchronous framework (FastAPI).
        *   **Use of Threading:** Explanation of using `asyncio.to_thread` to run synchronous, blocking code (like file I/O or CPU-bound parsing from `docling`) without blocking the main server event loop, making the application responsive and usable.
*   **4.2. The Document Ingestion Pipeline**
    *   **Parsing with `docling`:** How PDFs are converted into a structured `DoclingDocument` object.
    *   **Context Caching:** The implementation of a caching mechanism for `docling` outputs.
        *   How serializing the parsed document object to a file avoids costly reprocessing.
        *   Implications on performance and operating costs (fewer CPU cycles).
        *   Logic for handling cache hits, misses, and failures.
    *   **Advanced Chunking Strategy:**
        *   **The Difficulties of Table Chunking:** A dedicated section on this challenge. Discuss how naive splitting destroys table semantics.
        *   The solution: Using `docling_core`'s `HybridChunker` with a `MarkdownTableSerializer` to convert tables into a machine-readable text format, preserving their structure within a chunk.
    *   **Vectorization and Storage:** The process of creating embeddings in batches and upserting them into a company-specific Qdrant collection.
*   **4.3. The Query and RAG Analysis Pipeline**
    *   **Step 1: Dynamic Schema Generation:** The prompt engineering used to instruct an LLM to create a JSON schema based on the user's query.
    *   **Step 2: The Conversational RAG Process:**
        *   **Retrieval:** Fetching candidate chunks from Qdrant.
        *   **Conversational Evaluation:** Detailing the prompt structure (`SystemMessage`, `HumanMessage`) that tasks the LLM to act as a "document evaluator." Explain how each chunk is presented to the LLM within a persistent conversation, and how its structured JSON evaluation is captured.
        *   **Filtering:** Applying a relevance score threshold to the evaluated chunks.
        *   **Conversational Synthesis:** Explaining the final prompt in the *same conversation*, which asks the LLM to synthesize a consolidated answer based on the chunks it has just evaluated, leveraging its conversational memory.
*   **4.4. Frontend Development with Streamlit**
    *   Structuring the multi-page application.
    *   Using `st.cache_data` to optimize data loading and prevent redundant database calls.
    *   Managing interactive UI state with `st.session_state` for a smooth user experience (e.g., showing/hiding sources).
    *   Rendering PDF page images with bounding box overlays using `pypdfium2` and `Pillow` to provide visual source verification.

---

## 5. Evaluation and Discussion

*   **5.1. Experimental Setup**
    *   Hardware/Software Environment (local machine specs, cloud services, model versions).
    *   Dataset description (e.g., number and type of sustainability reports used).
    *   Evaluation metrics (e.g., field-level F1-score for structured data, human assessment of answer quality).
*   **5.2. Performance and Latency Analysis**
    *   Measuring end-to-end times for ingestion and query pipelines.
    *   Identifying performance bottlenecks (e.g., sequential LLM API calls during evaluation).
*   **5.3. Qualitative Analysis of Results**
    *   Showcasing successful extraction examples.
    *   A frank analysis of failure modes: Where does the system fail? (e.g., extremely convoluted tables, ambiguous queries, LLM non-compliance).
    *   The impact of the chunk evaluation step: Did it successfully filter out irrelevant or misleading context?
*   **5.4. Cost-Benefit Analysis**
    *   **Calculating System Operating Costs:**
        *   A model for estimating the cost per query based on token usage for schema generation, N chunk evaluations, and final synthesis.
        *   Discussing the cost-saving impact of the `docling` cache.
    *   **Comparison with Manual Labor:**
        *   Estimating the time and fully-loaded cost of a human analyst to perform the same data extraction tasks.
        *   Discussion on the trade-offs: speed, scalability, and consistency of the automated system versus the nuanced understanding of a human.
        *   Answering the question: "Was it actually worth it?" from a business perspective.
*   **5.5. Limitations and Future Work**
    *   The system's dependency on the capabilities of the chosen LLM.
    *   Potential for parallelizing the chunk evaluation LLM calls to reduce latency.
    *   Exploring more advanced retrieval or re-ranking strategies.
    *   Expanding the system to support other document formats.

---

## 6. Conclusion

*   **6.1. Summary of Work**
    *   Recap the problem, the developed system, and its novel architectural components.
*   **6.2. Answering the Research Questions**
    *   Provide direct, evidence-based answers to the research questions from Section 1.2.
*   **6.3. Final Remarks**
    *   The broader implications of this work for automating knowledge work in fields like finance, legal, and compliance.
    *   Concluding thoughts on the future of hybrid AI systems that combine retrieval with sophisticated, multi-step reasoning.

---

## Appendices

*   **A. Key Source Code Snippets**
*   **B. API Endpoint Reference**
*   **C. Example Prompts**

## References

*   List of all cited academic papers, articles, and software documentation.
