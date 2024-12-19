# Small RAG test implementation with langchain.

## Data
Movie summary data was used from [kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots).

## Preparing the Data
The file `create_data.py` reads a csv file and creates a folder with each movie summary as a plain text file using the release year and the title for the name.

Note: every script uses click; use the `--help` for usage info.
```
python create_data.py --help
Usage: create_data.py [OPTIONS]

Options:
  --data_file TEXT  Folder location of the data file.  [required]
  --data_path TEXT  Folder location of the output data files.
  --help            Show this message and exit.
```

## Creating the Database
ChromaDB is used for the vector store database. Uses hugging_face embeddings as default
```
python create_database.py --help
Usage: create_database.py [OPTIONS]

Options:
  --data_path TEXT   Folder location of the data files.
  --db_path TEXT     Location where to store the database.  [default: db]
  --model_type TEXT  Embedding model to use [hugging_face|google].  [default:
                     hugging_face]
  --help             Show this message and exit.
```

## Running the RAG program
```
python rag.py --help
Usage: rag.py [OPTIONS]

Options:
  --db_path TEXT     Folder location of the database.
  --model_type TEXT  Embedding model to use [hugging_face|google].
  --help             Show this message and exit.
```

Example:
```
python rag.py --db_path db_google --model_type google
write your query:
what happened at the end of Thor: Ragnarok?
[0.6148830874380088, 0.6078887168285054, 0.5997808600921912]
Human:
Answer the question based only on the following context:

Two years after the battle of Sokovia, Thor has been unsuccessfully searching for the Infinity Stones, and is now imprisoned by the fire demon Surtur in Muspelheim. Surtur reveals that Thor's father Odin is no longer on Asgard, and that the realm will soon be destroyed in the prophesied Ragnarök, once Surtur unites his crown with the Eternal Flame that burns in Odin's vault. Thor defeats Surtur and claims his crown, believing he has prevented Ragnarök.

---

allow their escape. Thor, facing Hela, loses his right eye and then has a vision of Odin that helps him realize only Ragnarök can stop her. While Hela is distracted, Loki locates Surtur's crown and places it in the Eternal Flame. Surtur is reborn and destroys Asgard, seemingly killing Hela.

---

Thor returns to Asgard to find his brother Loki posing as Odin. Thor forces Loki to help him find their father, and with directions from Stephen Strange on Earth, they locate Odin in Norway. Odin explains that he is dying, and that his passing will allow his firstborn child, Hela, to escape from a prison she was sealed in long ago. Hela had been the leader of Asgard's armies, and had conquered the Nine Realms with Odin, but she had been imprisoned and written out of history after Odin feared that she had become too ambitious. Odin subsequently dies, and Hela, released from her imprisonment,

---

Answer the question based on the above context: what happened at the end of Thor: Ragnarok?

Response: content='Surtur was reborn when Loki placed his crown in the Eternal Flame. Surtur then destroyed Asgard, seemingly killing Hela in the process.\n' additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-c753dc33-287f-4149-a7bd-f580803ea61b-0' usage_metadata={'input_tokens': 333, 'output_tokens': 30, 'total_tokens': 363, 'input_token_details': {'cache_read': 0}}
Sources: ['data_small/2017_thor:_ragnarok.txt', 'data_small/2017_thor:_ragnarok.txt', 'data_small/2017_thor:_ragnarok.txt']
```