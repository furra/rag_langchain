from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import click
import os
import shutil
import sys

load_dotenv("../.env")


SUPPORTED_EMBEDDINGS = {
    "hugging_face": HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        task="feature-extraction",
    ),
    "google": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
}


def validate_path(ctx, param, value):
    if not os.path.exists(value):
        raise click.BadParameter(
            "Data path doesn't exist. Run the create_data.py script first."
        )
    return value


def validate_model(ctx, param, value):
    if value not in SUPPORTED_EMBEDDINGS:
        raise click.BadParameter(f"Model '{value}' not supported.")
    return value


@click.command()
@click.option(
    "--data_path",
    default="data/",
    callback=validate_path,
    help="Folder location of the data files.",
)
@click.option(
    "--db_path",
    default="db",
    show_default=True,
    help="Location where to store the database.",
)
@click.option(
    "--model_type",
    default="hugging_face",
    callback=validate_model,
    show_default=True,
    help="Embedding model to use [hugging_face|google].",
)
def create_db(data_path: str, db_path: str, model_type: str):
    embeddings = SUPPORTED_EMBEDDINGS[model_type]

    # loads documents
    loader = DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()

    # TODO: try semantic chunks
    # https://python.langchain.com/docs/how_to/semantic-chunker/
    # split texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if os.path.exists(db_path):
        delete = input(f"Found folder {db_path}. Delete? [Y|N] (Default N):").lower()
        if delete == "y":
            shutil.rmtree(db_path)
        else:
            click.echo("Aborting.")
            return

    # Create DB
    db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    print(f"Saved {len(chunks)} chunks to {db_path}.")


if __name__ == "__main__":
    create_db()
