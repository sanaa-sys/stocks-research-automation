import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import groq
import os

# Initialize Groq client
groq_client = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode([text])[0].tolist()

# Custom Groq LLM class
class GroqLLM(LLM):
    model_name: str = "llama-3.1-70b-versatile"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = groq_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq"

# Initialize Groq language model
llm = GroqLLM()

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    You are an expert financial analyst. Given the following user query about stocks and the provided context, 
    create an improved, detailed response that answers the user's question.
    
    Context: {context}
    
    User Query: {query}
    
    Detailed Response:
    """
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit app
st.title("Advanced Stock Search App (Powered by Groq)")

# User input
user_query = st.text_input("Enter your stock search query:")

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "stocks"  # Make sure this matches your Pinecone index name
namespace = "stock-descriptions"

# Initialize the vector store
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = LangchainPinecone.from_existing_index(index_name=index_name, namespace=namespace,embedding=hf_embeddings)

if user_query:
    # Retrieve relevant context
    relevant_docs = vectorstore.similarity_search(user_query, k=5)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Generate response using RAG
    with st.spinner("Generating response..."):
        response = chain.run({"query": user_query, "context": context})
    
    # Display results
    st.subheader("Query:")
    st.write(user_query)
    
    st.subheader("Response:")
    st.write(response)

    # Display relevant stocks
    st.subheader("Relevant Stocks:")
    for doc in relevant_docs:
        stock_info = doc.metadata
        with st.expander(f"{stock_info['Name']} ({stock_info['Ticker']}) - Relevance: {1 - score:.2f}"):
            st.write(f"Sector: {stock_info.get('Sector', 'N/A')}")
            st.write(f"Industry: {stock_info.get('Industry', 'N/A')}")
            st.write(f"Market Cap: ${stock_info.get('Market Cap', 'N/A')}")
            st.write(f"Volume: {stock_info.get('Volume', 'N/A')}")
            st.write(f"Description: {doc.page_content}")

# Add instructions
st.sidebar.title("How to use")
st.sidebar.write("""
1. Enter a query about stocks or companies you're interested in.
2. The app will use Pinecone to find relevant stock information.
3. Groq's LLM will generate a detailed response based on the query and relevant information.
4. Results are displayed with expandable details for each relevant stock.
""")

# Add a note about API usage and limitations
st.sidebar.title("Notes")
st.sidebar.write("""
- This app uses the Groq API and Pinecone. Make sure you have set your API keys in the Streamlit secrets.
- Be mindful of API usage costs.
- The search is based on company descriptions and may not capture all relevant factors.
""")

