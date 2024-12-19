import click
import os
import pandas
import sys

from tqdm import tqdm


# File needs to be a csv with the following columns: 'Release Year', 'Origin/Ethnicity', 'Plot', and 'Title'.
def file_exists(ctx, param, value):
    if not os.path.exists(value):
        raise click.BadParameter("Data file doesn't exist.")
    return value


def check_empty_data_folder(ctx, param, value):
    if os.path.exists(value) and len(os.listdir(value)) > 0:
        raise click.BadParameter(
            "Data folder is not empty, please delete the folder or its contents."
        )
    return value


@click.command()
@click.option(
    "--data_file",
    required=True,
    callback=file_exists,
    help="Folder location of the data file.",
)
@click.option(
    "--data_path",
    default="data/",
    callback=check_empty_data_folder,
    help="Folder location of the output data files.",
)
def create_data(data_file: str, data_path: str):
    data = pandas.read_csv(data_file)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    for _, row in tqdm(data.iterrows()):
        title = (
            row["Title"]
            .replace("'", "")
            .replace(" ", "_")
            .replace("/", "_slash_")
            .lower()
        )
        # check and create data folder
        file_name = f"{data_path}/{row['Release Year']}_{title}.txt"
        if (
            not os.path.exists(file_name)
            and row["Release Year"] > 2016
            and row["Origin/Ethnicity"] == "American"
        ):
            with open(file_name, "w") as file:
                file.write(row["Plot"])


if __name__ == "__main__":
    create_data()
