import streamlit as st
import pandas as pd
import openai
from pinecone import Pinecone, ServerlessSpec
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
import asyncio
import time
import logging
from threading import Thread
import traceback
from slack_sdk.errors import SlackApiError
from datetime import datetime, timedelta
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app title
st.title("Leave Buddy - Slack Bot")

# Load environment variables (replace with your actual keys)
openai.api_key = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
SLACK_BOT_TOKEN = "your-slack-bot-token"
SLACK_APP_TOKEN = "your-slack-app-token"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "leave-buddy-index"

# Initialize Slack app
app = AsyncApp(token=SLACK_BOT_TOKEN)

# Create a placeholder for logs
if 'logs' not in st.session_state:
    st.session_state.logs = ""
log_placeholder = st.empty()

# Function to update Streamlit log
def update_log(message):
    st.session_state.logs += message + "\n"
    log_placeholder.text_area("Logs", st.session_state.logs, height=300)

# Cached function to generate embeddings
@lru_cache(maxsize=1000)
def get_embedding(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return tuple(response['data'][0]['embedding'])
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

# Function to create embeddings
def create_embeddings(df):
    records = []
    for i, row in df.iterrows():
        text = f"{row['NAME']} is on leave on {row['DATE']} ({row['DAY']}) for {row['FESTIVALS']}. This is in {row['MONTH']} {row['YEAR']}."
        embedding = get_embedding(text)
        if embedding:
            records.append((str(i), embedding, {"text": text}))
    return records

# Function to upload data to Pinecone
def upload_to_pinecone(records):
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        index = pc.Index(index_name)
        index.upsert(vectors=records)
        
        return True, "Data uploaded to Pinecone successfully!"
    except Exception as e:
        logger.error(f"Error uploading to Pinecone: {e}")
        return False, f"Error uploading data to Pinecone: {str(e)}"

# Function to query Pinecone and format response
async def query_pinecone(query):
    try:
        index = pc.Index(index_name)
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return None
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        if results['matches']:
            context = " ".join([match['metadata']['text'] for match in results['matches']])
            return context
        else:
            return None
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
    return None

# Function to query GPT and generate response
async def query_gpt(query, context):
    try:
        today = datetime.now().strftime("%d-%m-%Y")
        
        messages = [
            {"role": "system", "content": f"""You are LeaveBuddy, an AI assistant for employee leave information. Today is {today}. Provide concise, direct answers about employee leaves based on the given context. Mention specific dates in your responses."""},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",  # Using a faster model
            messages=messages,
            max_tokens=100,
            n=1,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error querying GPT: {e}")
        return f"Error: Unable to process the query. Please try again."

# Function to process query and generate response
async def process_query(query):
    try:
        context = await query_pinecone(query)
        if context:
            response = await query_gpt(query, context)
            return response
        else:
            return "I'm sorry, I couldn't find relevant information for your query."
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return "I encountered an error while processing your query. Please try again later."

# Enhanced Slack event handler
@app.event("message")
async def handle_message(event, say):
    try:
        text = event.get("text", "")
        response = await process_query(text)
        await say(response)
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        await say("I'm sorry, I encountered an error. Please try again.")

# Function to run the Slack bot
def run_slack_bot():
    async def start_bot():
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()

    asyncio.run(start_bot())

# Sidebar for optional data upload
st.sidebar.header("Update Leave Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%d-%m-%Y')
    
    st.sidebar.write("Uploaded Data Preview:")
    st.sidebar.dataframe(df.head())
    
    if st.sidebar.button("Process and Upload Data"):
        with st.spinner("Processing and uploading data..."):
            embeddings = create_embeddings(df)
            success, message = upload_to_pinecone(embeddings)
        st.sidebar.write(message)
        if success:
            st.session_state['data_uploaded'] = True
            st.sidebar.success("Data processed and uploaded successfully!")

# Main interface for starting the Slack bot
st.header("Slack Bot Controls")
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

if st.button("Start Slack Bot", disabled=st.session_state.bot_running):
    st.session_state.bot_running = True
    st.write("Starting Slack bot...")
    thread = Thread(target=run_slack_bot)
    thread.start()
    st.success("Slack bot is running! You can now ask questions in your Slack channel.")

if st.session_state.bot_running:
    st.write("Slack bot is active and ready to answer queries.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Leave Buddy is ready to use!")
