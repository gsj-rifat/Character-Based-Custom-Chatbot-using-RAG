# # Custom Chatbot Project

# ## Data Description
# The 'character_descriptions.csv' dataset is the file which containes the characters information such as name, short description, medium and setting. This project will implement the RAG approach to customize the chatbot using this dataset.
# 

# ## Reason for Dataset Selection
# 
# I have choosen this character_descriptions.csv dataset because this is an excellent choice for this RAG project due to its synthetic nature, evaluation clarity, structured format, and domain relevance. We can accurately measure the effectiveness of retrieval and grounding, Trace hallucinations and fine-tune prompts, Simulate real-world chatbot scenarios in a safe and experimental way.

# ## Data Wrangling


import pandas as pd
import tiktoken
import numpy as np
from embeddings_utils import get_embedding, distances_from_embeddings


df = pd.read_csv("character_descriptions.csv", index_col=False)
df.head(5)



#Create the text column that describe the data
pd.options.display.max_colwidth = 300

df["text"] = df.apply(
    lambda row: f"{row['Name']} is {row['Description']}. "
                f"This character is shown in the {row['Medium']} in {row['Setting']}.",
    axis=1
)

df.head(2)


# Embedding the text column
import openai

openai.api_base = "https://openai.vocareum.com/v1"

# For security reason, I omited the API KEY after completing the project.

openai.api_key = ""


EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
batch_size = 100
embeddings = []
for i in range(0, len(df), batch_size):
    # Send text data to OpenAI model to get embeddings
    response = openai.Embedding.create(
        input=df.iloc[i:i+batch_size]["text"].tolist(),
        engine=EMBEDDING_MODEL_NAME
    )
    
    # Add embeddings to list
    embeddings.extend([data["embedding"] for data in response["data"]])

# Add embeddings list to dataframe
df["embeddings"] = embeddings
df.head(2)


# ## Custom Query Completion



def get_cosine_distance(question, df):
    """
    This function do as following:
    First, it generates an embedding for the user's question.
    Next, it creates a copy of the original DataFrame.
    Then, it calculates a distances column that measures how similar each row's text is to the user’s question.
    Finally, it sorts the DataFrame in ascending order of distance — placing the most relevant texts ( those closest in meaning) at the top.
    """

    # Get the embedding for question text
    question_embeddings = get_embedding(question, model=EMBEDDING_MODEL_NAME)

    # Copy the current dataframe. Create distances column
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(question_embeddings,
                                                df_copy["embeddings"].values,
                                                distance_metric="cosine")

    # Order by ascending order. The closer distance mean better relevant
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy



def get_relevant_context(prompt_template, question, df, max_token_count):
    # count the total token by tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model. 
    
    """
    
    # Count total token
    current_token_count = len(tokenizer.encode(prompt_template)) + len(tokenizer.encode(question))

    # List of contexts to send to Openai
    context = []
    for text in get_cosine_distance(question, df)["text"].values:
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count
        # if not exceed max tokens, append to context
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break
    return context


def prompt_and_context(question, df, max_token_count):
    """
    Format the prompt template, add relevant contexts to guide chatbot to answer user questions.
    This is no-shot example.
    """

    # Prompt template to instruct the chatbot
    prompt_template = """
    You are a smart assistant to answer the question based on provided context. \
    If the question can not be answered based on the provided contexts, only say \ 
    "The question is out of scope. Could you please check your question or ask another question". Do not try to \
    answer the question out of the provide contexts.
    Context: 

    {}

    ---

    Question: {}
    Answer:"""

    # Get the relevant context
    context = get_relevant_context(prompt_template = prompt_template, question = question, 
                                   df = df, max_token_count = max_token_count)
    # Format the prompt template
    prompt_template = prompt_template.format("\n\n###\n\n".join(context), question)

    return prompt_template


# ## Custom Performance Demonstration

# ### Question 1


question_1 = "Who is the mother of Emily?"


# The general question "question_1" is sent to Openai
# Thus, the response is unknown
answer1_without_context = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=question_1,
    max_tokens=150
)
answer1_without_context["choices"][0]["text"]



df['text'].iloc[0]



# The question is sent along with relevant contexts
# Thus, the response is as expected (Emily is in a relationship with George)
answer1_customized = openai.Completion.create(
  model="gpt-3.5-turbo-instruct",
  prompt=prompt_and_context(question_1, df, 2000),
    max_tokens=150
)
answer1_customized["choices"][0]["text"]


# ### Question 2


question_2 = "Who is a middle-aged man?"



# The general question "question_2" is sent to Openai
# Thus, the response is hallucinated
answer2_without_context = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=question_2,
    max_tokens=150
)
answer2_without_context["choices"][0]["text"]


df['text'].iloc[1]



# The question is sent along with relevant contexts
# Thus, the response is as expected (Jack married to Alice and appears in the Play in England)
answer2_customized = openai.Completion.create(
  model="gpt-3.5-turbo-instruct",
  prompt=prompt_and_context(question_2, df, 2000),
    max_tokens=150
)
print(answer2_customized["choices"][0]["text"])





