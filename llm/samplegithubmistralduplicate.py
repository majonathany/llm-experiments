from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

import nest_asyncio
nest_asyncio.apply()

# Articles to index
articles = ["https://www.fantasypros.com/2023/11/rival-fantasy-nfl-week-10/",
            "https://www.fantasypros.com/2023/11/5-stats-to-know-before-setting-your-fantasy-lineup-week-10/",
            "https://www.fantasypros.com/2023/11/nfl-week-10-sleeper-picks-player-predictions-2023/",
            "https://www.fantasypros.com/2023/11/nfl-dfs-week-10-stacking-advice-picks-2023-fantasy-football/",
            "https://www.fantasypros.com/2023/11/players-to-buy-low-sell-high-trade-advice-2023-fantasy-football/"]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()

# Converts HTML to plain text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=100,
                                      chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

retriever = db.as_retriever()

# Create prompt template
prompt_template = """
### [INST] Instruction: Answer the question based on your fantasy football knowledge. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

rag_chain.invoke("Should I start Gibbs next week for fantasy?")
