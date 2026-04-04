This set of scripts will allow you to scrape any publicly accessible website to create a RAG store for use
with a pretrained LLM, enabling you to create a chatbot that is a subject matter expert on content from that
website.

Requires python3.12 as of this writing, in order to install torch with cuda. I recommend running python3.12
in a virtual environment for this. I highly recommend running this with cuda enabled as it will drastically
reduce the time that embed_documents takes to complete.

Recommended system requirements:
8 GB dedicated GPU memory, cuda capable GPU
32 GB of RAM

For reference, I built and tested this on a website with over 65,000 unique URLs. Total runtime for that
website was approximately 21 hours, which is why I have implemented checkpointing and resuming.

Ensure your venv is activated.  
run pip install -r requirements.txt  
In crawler.py, set START_URL and BASE_DOMAIN  
Run crawler to create urls_list.py  
Run load_prep to create docs.json  
Run chunk to create chunks.json  
run embed_documents to create chroma_db_nomic database

Once embed_documents has completed, you will have a directory named chroma_db_nomic that you can use as a RAG
vector store for your program utilizing a pretrained LLM. I recommend removing the checkpointing files once
complete, as well as the docs.json file. My docs.json file was ove 2.5GB from the above referenced website.
