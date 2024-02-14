import glob
import os
import getpass

from langchain import LLMChain, PromptTemplate, OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from pydantic import BaseModel, Field

from langchain_mistralai import MistralAIEmbeddings


from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain.chains import SQLDatabaseChain
from faker import Faker

CONNECTION_STRING = "postgresql+psycopg2://postgres@localhost:5432/langchain2"
password = 'ornsoft1'
faker = Faker()
        
def collection_name():
    return faker.city() + f"{faker.zipcode()}"

def initialize(textlines: list, collection_name = None):
    if not collection_name:
        collection_name = collection_name()
    


CONNECTION_STRING = "postgresql+psycopg2://postgres@localhost:5432/langchain2"
COLLECTION_NAME = "test_8324"


def mysqldb():
    db = SQLDatabase.from_uri("mysql://root:@127.0.0.1:3306/")
    llm = OpenAI(temperature=0, verbose=True)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    return db_chain


def initialize():
    textfile = glob.glob("/Users/jonathan/work/llm/corpus/1738.txt")

    textlines = textfile.split("-----");

    for t in textlines:
        loader = TextLoader(t)
        documents = loader.load()
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embeddings = MistralAIEmbeddings()
        db = PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=collection_name,
            connection_string=CONNECTION_STRING,
            pre_delete_collection=True
        )

def query(question, collection_name):
    embeddings = MistralAIEmbeddings()

    store = PGVector(
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )


    docs_with_score = store.similarity_search_with_score(question)
    for i, doc in enumerate(docs_with_score):
        print(f"{i + 1}.", doc.page_content, "\n")


    context = []

    for doc, score in docs_with_score:
        context.append(doc.page_content)

    # If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt_msgs = [
        SystemMessage(
            content="""
<<<<<<< HEAD
=======
              Your name is ChatCEE. You are an AI assistant providing helpful advice regarding Certificat d'economie d'energie. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided. You should answer question in French language only.
You should only provide your answer based on the most up to date information in the context provided. You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.
If you can't find the answer in the context below, just say "Hmm, je ne suis pas sûr." Don't try to make up an answer. Lorsque l'on te pose une question, répond toujours en considérant que la personne qui te pose la question est un bénéficiaire de travaux de rénovation énergétique.
Si tu reçois une question demandant comment obtenir une prime cee, considère qu il s agit d’une question d’un bénéficiaire de travaux qui cherche à obtenir une aide  financiere en  euros , et non pas à connaître les règles entre le PNCEE et les obligés et délégataires.
Si la question n'est pas liée au contexte, répondez poliment que vous êtes réglé pour ne répondre qu'aux questions liées au contexte des CEE.
Si on te demande qui t'a crée ou qui a créé "ChatCEE", tu dois répondre spécifiquement que ton créateur est OrNsoft, l'entreprise de développement de renommer mondial spécialisé dans l'intelligence artificielle et de se rendre sur https://www.OrNsoft.com pour en savoir plus.
>>>>>>> 3e2aa0570e627906caba7a652ce492b30b83e3a3
Question: {question}
=========
{context}
=========
Answer in Markdown: ,
            """
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

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever()).combine_documents_chain

    mychain = ConversationalRetrievalChain(combine_docs_chain=qa_chain, question_generator=questionGeneratorChain, retriever=store.as_retriever() )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), store.as_retriever(), memory=memory)

    chat_history = []

    while question != "Q":
        question = input("What do you want to ask? ")
        response = mychain.run({'question': question, 'chat_history': chat_history})
        print(response)

