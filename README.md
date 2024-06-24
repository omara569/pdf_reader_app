# pdf_reader_app

This is a project that uses [Google Gemini Pro](https://ai.google.dev/) in tandem with the [Langchain](https://python.langchain.com/v0.2/docs/introduction/) framework to read PDFs, extract the text of those PDFs, and then answer queries relative to the PDF. The app was developed using [streamlit](https://streamlit.io/).

## Installation

It is recommended that a virual environment is used and the path to the directory containing these files is added to the path in the virtual environment.

These libraries and their dependencies can be installed using the "requirements.txt" file using [pip](https://pip.pypa.io/en/stable/installation/) package manager:
```bash
pip install requirements.txt
```

## API key and dotenv

- An API key is needed to work with Google's gemini locally. Create a ".env" file with ```GEM_API_KEY = "API_KEY"``` where API_KEY is replaced with the actual API key generated. Be sure to maintain the double quotes.

## Vector Storage Method

This pertains to how the text is stored in a way that is speed efficient when looking up a query. Text is stored as an embedding, which a vector of values representing the word or token. In this embedding space, words associated more closely together have more similar embeddings. 

A query will be converted into an embedding and then used to query the text to find the most relavant text based on similarity. The resulting text is then fed to our LLM to provide the context that is used to respond to the query.

The method I used was [FISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/). One can also use other storage methods found [here](https://python.langchain.com/v0.2/docs/integrations/vectorstores/).

## Running Locally

To run locally, simply write in terminal (with virtual environment active, if applicable):
`streamlit run app.py`

From there, connect to the localhost address that the application is running on and it's available for your use!