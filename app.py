# app.py

import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# -------------------------------------------
# Load environment variables
# -------------------------------------------
# Steps to Get a Hugging Face Access Token
#Go to https://huggingface.co/join, Create a free account
#Click on your Profile Picture, Select Settings from the dropdown, Inside Settings, click on Access Tokens
#Click the New token button, Give your token a name, Set the Role to: "Read" access (enough for using models), Click Generate Token.
# Your Access Token will appear
# Warning: For security, you won't see the token again after you leave the page. If you lose it, you can create a new one.

load_dotenv(dotenv_path="token.env")
hf_token = os.getenv("HF_TOKEN")

# Setup Hugging Face Inference Client

client = InferenceClient(
    model="google/flan-t5-small",
    token=hf_token
)


# -------------------------------
# Load and Prepare Dataset
# -------------------------------
df = pd.read_csv("character_descriptions.csv", index_col=False)

# Create descriptive text field
df["text"] = df.apply(
    lambda row: f"{row['Name']} is {row['Description']}. "
                f"This character is shown in the {row['Medium']} in {row['Setting']}.",
    axis=1
)

# -------------------------------
# Load Local Embedding Model
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight & offline
df["embeddings"] = df["text"].apply(lambda text: embedding_model.encode(text))


# -------------------------------
# Utility Functions
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(text):
    return embedding_model.encode(text)


def distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine"):
    if distance_metric == "cosine":
        return [1 - cosine_similarity(query_embedding, e) for e in embeddings]


def get_cosine_distance(question, df):
    question_embeddings = get_embedding(question)
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(question_embeddings,
                                                     df_copy["embeddings"].tolist(),
                                                     distance_metric="cosine")
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy


def get_relevant_context(prompt_template, question, df, max_token_count):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    current_token_count = len(tokenizer.encode(prompt_template)) + len(tokenizer.encode(question))
    context = []
    for text in get_cosine_distance(question, df)["text"].values:
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break
    return context


def prompt_and_context(question, df, max_token_count):
    prompt_template = """
    You are a smart assistant to answer the question based on provided context. \
    If the question can not be answered based on the provided contexts, only say \
    "The question is out of scope. Could you please check your question or ask another question". Do not try to \
    answer the question out of the provided contexts.
    Context: 

    {}

    ---

    Question: {}
    Answer:"""

    context = get_relevant_context(prompt_template, question, df, max_token_count)
    prompt_template = prompt_template.format("\n\n###\n\n".join(context), question)
    return prompt_template


# -------------------------------
# Gradio Interface Logic
# -------------------------------
def generate_response(user_question):
    prompt = prompt_and_context(user_question, df, max_token_count=2000)
    # Use Hugging Face hosted model to generate the output
    output = client.text_generation(prompt, max_new_tokens=200)
    return output



# -------------------------------
# Launch Gradio App
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🧙 Character-Based RAG Chatbot")
    gr.Markdown("Ask questions about fictional characters. The bot retrieves context locally and answers!")

    with gr.Row():
        user_input = gr.Textbox(label="Ask a question", placeholder="e.g. Who is a middle-aged man?")
        output = gr.Textbox(label="Bot's Answer", lines=8)

    send_btn = gr.Button("Send")
    send_btn.click(fn=generate_response, inputs=user_input, outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
