import click
import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEndpointEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from pprint import pprint

load_dotenv("../.env")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

SUPPORTED_MODELS = ("hugging_face", "google")


def validate_path(ctx, param, value):
    if not os.path.exists(value):
        raise click.BadParameter(
            "Database path doesn't exist. Run the create_database.py script first."
        )
    return value


def validate_model(ctx, param, value):
    if value not in SUPPORTED_MODELS:
        raise click.BadParameter(f"Model '{value}' not supported.")
    return value


@click.command()
@click.option(
    "--db_path",
    default="db/",
    callback=validate_path,
    help="Folder location of the database.",
)
@click.option(
    "--model_type",
    default="hugging_face",
    callback=validate_model,
    help="Embedding model to use [hugging_face|google].",
)
def rag(db_path: str, model_type: str):
    # add more models
    if model_type == "hugging_face":
        embedding_function = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            task="feature-extraction",
        )

        llm = HuggingFaceEndpoint(
            repo_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            max_new_tokens=512,
        )
        model = ChatHuggingFace(llm=llm, verbose=True)
    elif model_type == "google":
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    else:
        raise click.BadParameter(f"model_type {model_type} not supported.")

    db = Chroma(embedding_function=embedding_function, persist_directory=db_path)

    print("write your query: ")
    query = input()
    while query != "" or query != "exit":
        results = db.similarity_search_with_relevance_scores(query, k=3)
        print([score for _doc, score in results])

        context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query)
        print(prompt)

        response = model.invoke(prompt)
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response}\nSources: {sources}"
        print(formatted_response)

        print("write your query: ")
        query = input()


if __name__ == "__main__":
    rag()
