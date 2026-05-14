### What's all this then?

This repo contains my work for boot.dev's RAG (Retrieval Augmented Generation) course.

RAG is a technique of Augmenting the Generated output of an LLM by first Retrieving relevant documents and providing them to the LLM as additional context on top of the user and system queries. 

##### Topics covered in this course

- Keyword Search: Finding relevant documents by the keywords they contain. Includes methodologies such as preprocessing & tokenization of queries and documents, inverse indexing, search relevance formulars such as BM 25.
- Semantic Search: Finding relevant documents by the semantic similarity between their contents and the query. Includes methodologies such as embedding of text into vector spaces, chunked embedding, overlapped embedding, sentence based embedding.
- Hybrid Search: Combinding both search approaches to create a combined relevance ranking. Includes Score normalization, weighted combination & reciprocal rank fusion (rrf).
- Query Preprocessing: Letting an LLM preprocess a query, e.g. by correcting spelling, rewriting or expanding it.
- Re-Ranking: After performing initial search (in this case via rrf hybrid search), use an LLM to re-rank the preliminary results. 
- Evaluation: How to evaluate the performance of a search method. Covered topics: Precision and Recall metrics based on a golden model dataset, plus combined F1 score. Using an LLM to judge how well search results fit the query. 
- Actual Augmented Generation: Some basic practice examples of querying the Gemini API with search results and various queries. 
- Multimodal Search: Search across media types. In this case, between text and image. Covers the concepts of multimodal embeddings/contrastive learning, where an LLM is trained on paired data of images and text and embeds them in the same vector space.