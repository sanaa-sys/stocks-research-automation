import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize OpenAI language model
llm = ChatOpenAI(model="gpt-3.5-turbo",
                 openai_api_key=st.secrets["OPENAI_API_KEY"],
                 temperature=0)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    You are an expert financial analyst. Given the following user query about stocks and the provided context, 
    create an improved, detailed prompt that would help in finding the most relevant stocks. 
    
    User Query: {query}
    
    Context: {context}
    
    Improved Prompt:
    """
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)


# Streamlit app
st.title("Advanced Stock Search App")

# User input
user_query = st.text_input("Enter your stock search query:")

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "stocks"  # Make sure this matches your Pinecone index name

# Initialize the vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

if user_query:
    # Retrieve relevant context
    relevant_docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate improved prompt using RAG
    with st.spinner("Generating improved prompt..."):
        improved_prompt = chain.run(query=user_query, context=context)
    
    # Display results
    st.subheader("Original Query:")
    st.write(user_query)
    
    st.subheader("Improved Prompt:")
    st.write(improved_prompt)

    # Search stocks based on the improved prompt
    with st.spinner("Searching for relevant stocks..."):
        results = vectorstore.similarity_search_with_score(improved_prompt, k=5)
    
    if results:
        st.subheader(f"Found {len(results)} relevant stocks:")
        for doc, score in results:
            stock_info = doc.metadata
            with st.expander(f"{stock_info['Name']} ({stock_info['Ticker']}) - Relevance: {1 - score:.2f}"):
                st.write(f"Sector: {stock_info.get('Sector', 'N/A')}")
                st.write(f"Industry: {stock_info.get('Industry', 'N/A')}")
                st.write(f"Market Cap: ${stock_info.get('Market Cap', 'N/A'):,}")
                st.write(f"Volume: {stock_info.get('Volume', 'N/A'):,}")
                st.write(f"Description: {doc.page_content}")
    else:
        st.write("No relevant stocks found.")

# Add instructions
st.sidebar.title("How to use")
st.sidebar.write("""
1. Enter a simple query about the types of stocks you're interested in.
2. The app will use LangChain and OpenAI to generate an improved, more detailed prompt.
3. The app will then use this improved prompt to search for relevant stocks in the Pinecone database.
4. Results are displayed with expandable details for each stock.
""")

# Add a note about API usage and limitations
st.sidebar.title("Notes")
st.sidebar.write("""
- This app uses the OpenAI API and Pinecone. Make sure you have set your API keys in the Streamlit secrets.
- Be mindful of API usage costs.
- The search is based on company descriptions and may not capture all relevant factors.
- The relevance score is based on vector similarity and may not always perfectly align with human judgment.
""")

