[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# gt-policy-bot

This is an LLM powererd application developed to answer questions about the Georgia Tech student code of conduct. The primary goal is develop a Question Answering system that is grounded in evidence. It is shown that supplementing an LLM with appropriate evidence reduces its tendency to hallucinate when faced with a question it does not know the answer to. 

So, when users ask our system a question, we retrieve relevant chunks of the code of conduct, and provide it to the the LLM as evidence. 

**Disclaimer**: LLMs are prone to make mistakes, and adding evidence doesn't reduce it to zero. So, this is still an academic experiment. Answers from the application is not definitive legal advice. Please consult the official code of conduct for authoritative information. 

Demo - [https://huggingface.co/spaces/mohit-raghavendra/gt-policy-bot](https://huggingface.co/spaces/mohit-raghavendra/gt-policy-bot)

Student Code of Conduct - [https://policylibrary.gatech.edu/student-life/student-code-conduct](https://policylibrary.gatech.edu/student-life/student-code-conduct)

## Running it locally

### Building and running the current demo locally


1. Clone the repository.
```
git clone https://github.com/mohit-raghavendra/gt-policy-bot
```

2. Install the requirements
```
pip install -r requirements.txt
```

3. Create an index on pinecone, and get the API key and environment. Also create an API key on Google's MakerSuite. 

4. Set the environment variables as follows:

```
export PINECONE_KEY=<YOUR KEY>
export PINECONE_ENV=<YOUR KEY>
export GOOGLE_PALM_KEY=<YOUR KEY>
```

5. Set the correct pinecone index name in ```config.yml```.


6. Run the following commands to chunk the document, vectorize it and embed it. 

```
python3 utils/parsedocument.py
python3 vectorize.py
```

7. Run the demo and navigate to [http://127.0.0.1:7860](http://127.0.0.1:7860)

```
gradio app.py
```