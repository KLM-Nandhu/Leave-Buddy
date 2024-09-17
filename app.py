import os
import openai
from pinecone import Pinecone, ServerlessSpec
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
import logging
from datetime import datetime
from functools import lru_cache
import pandas as pd
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

# Check if all required environment variables are set
required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "leave-buddy-index"

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
handler = SlackRequestHandler(app)

# Initialize Flask app
flask_app = Flask(__name__)

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
def query_pinecone(query):
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
def query_gpt(query, context):
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
     "[Employee Name] is not present on [Date]. Reason: [Leave Reason]"
   - If no leave information is found for the date, respond with:
     "[Employee Name] is present on [Date]."
6. IMPORTANT: Absence of leave information in the database means the employee is present.
7. Only mention leave information if it's explicitly stated in the context.
8. Limit responses to essential information only.
9. Do not add any explanations or pleasantries."""},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ]
        
        response = openai.ChatCompletion.create(
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
def process_query(query):
    try:
        context = query_pinecone(query)
        if context:
            response = query_gpt(query, context)
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
@app.event("message")
def handle_message(body, say):
    try:
        text = body.get("event", {}).get("text", "")
        user = body.get("event", {}).get("user", "")
        channel = body.get("event", {}).get("channel", "")
        
        # Send initial processing message
        processing_message = say("Processing your request... :hourglass_flowing_sand:")
        
        # Process the query
        response = process_query(text)
        
        # Update the processing message with the final response
        app.client.chat_update(
            channel=channel,
            ts=processing_message['ts'],
            text=f"<@{user}> {response}"
        )
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        say("I'm sorry, I encountered an error. Please try again.")

# Flask route for Slack events
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

# Function to update leave data
def update_leave_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%d-%m-%Y')
        
        print("Creating embeddings...")
        embeddings = create_embeddings(df)
        print("Uploading to Pinecone...")
        success, message = upload_to_pinecone(embeddings)
        print(message)
        if success:
            print("Data processed and uploaded successfully!")
        else:
            print("Failed to upload data.")
    except Exception as e:
        print(f"Error updating leave data: {str(e)}")

# Health check route
@flask_app.route("/", methods=["GET"])
def health_check():
    return "Leave Buddy is running!"

# Run the Flask app
if __name__ == "__main__":
    flask_app.run(debug=True)
