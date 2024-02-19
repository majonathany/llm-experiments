import glob, logging, torch
import os
import getpass

from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage


from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import TextLoader
from faker import Faker
from langchain_community.embeddings import HuggingFaceEmbeddings


from .LLM import initialize_model

CONNECTION_STRING = "postgresql+psycopg2://postgres@localhost:5432/langchain2"
password = 'ornsoft1'
faker = Faker()
        
def collection_name():
    return faker.city() + f"{faker.zipcode()}"

def initialize(textlines: list, collection_name = None):
    if not collection_name:
        collection_name = collection_name()


COLLECTION_NAME = "test_8324"

def db_name():
    DB_NAME = faker.first_name() + faker.zipcode()
    return DB_NAME

def get_embeddings(model, documents):
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    # tokenizer = model.tokenizer
    # model.eval()

    embeddings = []

    for doc in documents:
        # Add special tokens adds [CLS], [SEP], etc.
        # inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = model.generate(doc)

        # Get hidden states features for each token
        with torch.no_grad():
            outputs = model(**inputs)

        # Only use the token embeddings from the last hidden state
        last_hidden_states = outputs.last_hidden_state

        # Pool the embeddings for the whole document
        # Here, simply mean of token embeddings is used. You can use more sophisticated pooling strategies.
        doc_embedding = torch.mean(last_hidden_states, dim=1)
        embeddings.append(doc_embedding)

    return embeddings


def initialize(folder_directory=None):

    # if not folder_directory:
    #     sample = "/home/ubuntu/llm_experiments/input/2024-02-09 22:05:45.900486.txt"
    #     textlines = open(sample, 'r').readlines()
    # else:
    #     textfile = glob.glob(folder_directory)
    #     textlines = textfile.split("-----");

    textlines = ['/Users/jonathan/work/llm_experiments/single.txt']
    db = None

    collection_name_elem = collection_name()

    text_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=384, chunk_overlap=100)
    db = PGVector.from_documents(
        embedding=HuggingFaceEmbeddings(),
        documents=[],
        collection_name=collection_name_elem,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True
    )
    for t in textlines:
        embeddings = []
        if t.endswith('.txt'):
            loader = TextLoader(t)
            documents = loader.load()

            docs = text_splitter.split_documents(documents)

            db = PGVector.from_documents(
                embedding=HuggingFaceEmbeddings(),
                documents=docs,
                collection_name=collection_name_elem,
                connection_string=CONNECTION_STRING,
            )

    if db:
        return db.collection_metadata

def create_db(db_name):
    import psycopg2

    #establishing the connection
    conn = psycopg2.connect(
        database="postgres", user='postgres', password=password, host='127.0.0.1', port= '5432'
    )
    conn.autocommit = True
    
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    
    #Preparing query to create a database
    sql = f'CREATE database {db_name}'
    
    #Creating a database
    cursor.execute(sql)
    print("Database created successfully........")
    
    #Closing the connection
    conn.close()
    

def test_connection_to_db(collection_name):
    DB_NAME = db_name()
    
    try:
        create_db(DB_NAME)
        db = PGVector.from_documents(
            embedding=embeddings,
            documents=[],
            collection_name=collection_name,
            connection_string=CONNECTION_STRING + "postgres",
            pre_delete_collection=True
        )
        
        return True
    except Exception as e:
        logging.exception(e)
    return False

def query(question, collection_name):
    embeddings = HuggingFaceEmbeddings()

    store = PGVector(
        collection_name='Adamtown68555',
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    llm = initialize_model()

    docs_with_score = store.similarity_search_with_score(question)
    for i, doc in enumerate(docs_with_score):
        print(f"{i + 1}.", doc[0].page_content, "\n")


    context = []

    for doc, score in docs_with_score:
        context.append(doc.page_content)

    # If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.

    prompt_msgs = [
        SystemMessage(
            content=""
        ),
        HumanMessage(
            content=f"Use the given format to extract information from the following input:"
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        HumanMessage(content="Tips: Make sure to answer in complete sentences"),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    questionGeneratorChain = LLMChain(llm = llm, prompt=prompt)

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever()).combine_documents_chain

    mychain = ConversationalRetrievalChain(combine_docs_chain=qa_chain, question_generator=questionGeneratorChain, retriever=store.as_retriever() )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, store.as_retriever(), memory=memory)

    chat_history = []

    while question != "Q":
        question = input("What do you want to ask? ")
        response = mychain.run({'question': question, 'chat_history': chat_history})
        print(response)

