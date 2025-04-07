# Character-Based-Custom-Chatbot-using-RAG
This project builds a custom AI chatbot that answers questions based on fictional character descriptions using the Retrieval-Augmented Generation (RAG) approach. The dataset contains synthetic character data (names, descriptions, medium, and settings) and helps evaluate how well a chatbot can retrieve and generate contextually grounded responses.

---

## 🚀 Project Overview

**Goal:**  
Create a question-answering chatbot that can answer user queries by grounding responses in a predefined set of character descriptions.

**Key Features:**
- Uses a synthetic dataset (`character_descriptions.csv`) for controlled evaluation.
- Implements RAG to retrieve relevant text snippets based on the user's question.
- Leverages OpenAI's `text-embedding-ada-002` for embedding.
- Generates responses using `gpt-3.5-turbo-instruct`.

---

## 📦 Dataset

- `character_descriptions.csv`  
  Contains fictional character descriptions with the following fields:
  - `Name`
  - `Description`
  - `Medium` (TV, Film, Play)
  - `Setting`

These are combined into a single `text` column used for embeddings and retrieval.

---

## 🛠️ How It Works

1. **Data Preparation**
   - Load and preprocess the character dataset.
   - Create a `text` column combining character metadata.
   - Generate embeddings using OpenAI’s embedding API.

2. **Retrieval-Augmented Generation (RAG)**
   - Encode the user's question.
   - Compute cosine distances between the question and the dataset.
   - Select the most relevant texts within a token budget.
   - Format a prompt with selected contexts.
   - Send the prompt to the completion model for an answer.

3. **Demonstration**
   - Compare chatbot's performance with and without retrieval.
   - Observe hallucinations in generic queries versus grounded, contextual replies.

---

## 📚 Acknowledgments
OpenAI
Vocareum API environment
Udacity AI Chatbot Course (template inspiration)


