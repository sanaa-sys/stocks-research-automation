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
import yfinance as yf
import plotly.graph_objects as go
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key=st.secrets["NEWS_API_KEY"])

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
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("shortName", "N/A"),
        "summary": info.get("longBusinessSummary", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "price": info.get("currentPrice", "N/A"),
        "revenue_growth": info.get("revenueGrowth", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
    }

def format_large_number(num):
    if num == "N/A":
        return "N/A"
    try:
        num = float(num)
        if num >= 1e12:
            return f"${num/1e12:.1f}T"
        elif num >= 1e9:
            return f"${num/1e9:.1f}B"
        elif num >= 1e6:
            return f"${num/1e6:.1f}M"
        else:
            return f"${num:,.0f}"
    except:
        return "N/A"

def format_percentage(value):
    if value == "N/A":
        return "N/A"
    try:
        return f"{float(value)*100:.1f}%"
    except:
        return "N/A"

def display_stock_card(data, ticker):
    with st.container():
        st.markdown("""
            
        """, unsafe_allow_html=True)

        st.markdown(f"""
            {data['name']} ({ticker})

            {data['sector']} | {data['industry']}

            {data['summary'][:150]}...
        """, unsafe_allow_html=True)

        metrics = [
            {"label": "Market Cap", "value": format_large_number(data['market_cap'])},
            {"label": "Price", "value": format_large_number(data['price'])},
            {"label": "Growth", "value": format_percentage(data['revenue_growth'])},
            {"label": "Rating", "value": data['recommendation'].upper()}
        ]

        cols = st.columns(4)
        for col, metric in zip(cols, metrics):
            with col:
                st.metric(
                    label=metric['label'],
                    value=metric['value'],
                    delta=None,
                )

        st.markdown("", unsafe_allow_html=True)


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

    ticker_list = [doc.metadata.get('Ticker', '') for doc in relevant_docs if 'Ticker' in doc.metadata]

    # Generate response using RAG
    with st.spinner("Generating response..."):
        response = chain.run({"query": user_query, "context": context})
    

    
    st.subheader("Response:")
    st.write(response)

    # Display relevant stocks
    st.subheader("Relevant Stocks:")

    stock_data = []
    for ticker in ticker_list:
        data = fetch_stock_data(ticker)
        if data:
            display_stock_card(data, ticker)

    # Create comparison chart
    if len(ticker_list) > 0:
        st.subheader("Stock Price Comparison")
        fig = go.Figure()

        for i, ticker in enumerate(ticker_list):
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="1y")

                # Normalize the prices to percentage change
            hist_data['Normalized'] = (hist_data['Close'] / hist_data['Close'].iloc[0] - 1) * 100

            fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Normalized'],
                    name=f"{ticker}",
                    mode='lines'
                ))

        fig.update_layout(
                title="1-Year Price Performance Comparison (%)",
                yaxis_title="Price Change (%)",
                template="plotly_dark",
                height=500,
                showlegend=True
            )

        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Latest Market News")
    for ticker in ticker_list:
        with st.spinner(f"Loading news for {ticker}..."):
            news_items = newsapi.get_everything(
                        q=f"{ticker} stock",
                        language='en',
                        sort_by='relevancy',
                        page_size=3
                    )

        st.write(f"**Latest news for {ticker}**")
        for article in news_items['articles']:
             with st.expander(f"{article['title']}"):
                st.write(article['description'])
                st.write(f"**Source:** {article['source']['name']} | **Published:** {article['publishedAt']}")
                st.link_button("Read full article", article['url'])

