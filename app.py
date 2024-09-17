from flask import Flask, request, jsonify
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
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "leave-buddy-index"

# Initialize Slack app
slack_app = AsyncApp(token=SLACK_BOT_TOKEN)

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
            {"role": "system", "content": f"""You are LeaveBuddy, an efficient AI assistant for employee leave information. Today is {today}. Follow these rules strictly:

1. Provide concise, direct answers about employee leaves.
2. provide processing message for every request of the user.
3. Always mention specific dates in your responses.
4. For queries about total leave days, use this format:
   [Employee Name] has [X] total leave days in [Year]:
   - [Date]: [Reason]
   - [Date]: [Reason]
   ...
   Total: [X] days
5. For presence queries:
   - If leave information is found for the date, respond with:
     "[Employee Name] is  present on [Date]. Reason: [Leave Reason]"
   - If no leave information is found for the date, respond with:
     "[Employee Name] is  not present on [Date]."
6. IMPORTANT: Absence of leave information in the database means the employee is present.
7. Only mention leave information if it's explicitly stated in the context.
8. Limit responses to essential information only.
9. Do not add any explanations or pleasantries."""},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            n=1,
            temperature=0.7,
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
        else:
            # If no context is found, assume the employee is present
            employee_name = query.split()[1]  # Extracts the name from "is [name] present today?"
            today = datetime.now().strftime("%d-%m-%Y")
            response = f"{employee_name} is present on {today}."
        return response
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return "I encountered an error while processing your query. Please try again later."

# Slack event handler
@slack_app.event("message")
async def handle_message(event, say):
    try:
        text = event.get("text", "")
        channel = event.get("channel", "")
        
        # Send initial processing message
        processing_message = await slack_app.client.chat_postMessage(
            channel=channel,
            text="Processing your request... :hourglass_flowing_sand:"
        )
        
        # Get the timestamp of the processing message
        processing_ts = processing_message['ts']
        
        # Process the query
        response = await process_query(text)
        
        # Update the processing message with the final response
        try:
            await slack_app.client.chat_update(
                channel=channel,
                ts=processing_ts,
                text=response
            )
        except SlackApiError as e:
            logger.error(f"Error updating message: {e}")
            # If update fails, send a new message
            await say(response)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        await say("I'm sorry, I encountered an error. Please try again.")

# Function to run the Slack bot
def run_slack_bot():
    async def start_bot():
        handler = AsyncSocketModeHandler(slack_app, SLACK_APP_TOKEN)
        await handler.start_async()

    asyncio.run(start_bot())

# Flask route for health check
@app.route('/')
def health_check():
    return jsonify({"status": "healthy"}), 200

# Flask route to start the Slack bot
@app.route('/start-bot', methods=['POST'])
def start_bot():
    if not os.environ.get('BOT_STARTED'):
        thread = Thread(target=run_slack_bot)
        thread.daemon = True
        thread.start()
        os.environ['BOT_STARTED'] = 'true'
        return jsonify({"message": "Slack bot started successfully"}), 200
    else:
        return jsonify({"message": "Slack bot is already running"}), 200

if __name__ == "__main__":
    app.run(debug=True)
