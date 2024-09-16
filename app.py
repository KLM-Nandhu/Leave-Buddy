import pandas as pd
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
import openai
from datetime import datetime
import asyncio
import os

# Slack and OpenAI setup
BOT_TOKEN = os.getenv("BOT_TOKEN")
APP_TOKEN = os.getenv("APP_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

app = AsyncApp(token=BOT_TOKEN)

# Load Excel data from a file
def load_data():
    df = pd.read_excel("./holidays.xlsx")  # Ensure the file is in the same directory
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
    return df

df = load_data()

async def query_gpt(prompt):
    response = await openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

async def get_holiday_info(query):
    today = datetime.now().date()
    gpt_interpretation = await query_gpt(f"Interpret this query about Nandhakumar's holiday schedule: '{query}'. Extract any dates or festivals mentioned.")
    
    if "date" in gpt_interpretation.lower():
        date_str = gpt_interpretation.split("date:")[-1].strip().split()[0]
        try:
            query_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return f"Unable to parse the date: {date_str}. Please use YYYY-MM-DD format."
        
        date_record = df[df['DATE'].dt.date == query_date]
        
        if not date_record.empty:
            festivals = ", ".join(date_record['FESTIVALS'].unique())
            return f"On {query_date}, Nandhakumar has the following holiday(s): {festivals}"
        else:
            return f"Nandhakumar has no holidays scheduled for {query_date}"
    
    elif "festival" in gpt_interpretation.lower():
        festival = gpt_interpretation.split("festival:")[-1].strip()
        festival_records = df[df['FESTIVALS'].str.contains(festival, case=False, na=False)]
        
        if not festival_records.empty:
            response = f"Nandhakumar's {festival} holiday(s) are on the following dates:\n"
            for _, row in festival_records.iterrows():
                response += f"{row['DATE'].strftime('%Y-%m-%d')} ({row['DAY']})\n"
            return response
        else:
            return f"No records found for the festival: {festival} in Nandhakumar's schedule"
    
    elif "check holidays" in query.lower():
        today_holidays = df[df['DATE'].dt.date == today]
        if today_holidays.empty:
            return "Nandhakumar has no holidays today."
        else:
            festivals = ", ".join(today_holidays['FESTIVALS'])
            return f"Nandhakumar's holiday(s) today: {festivals}"
    
    else:
        return "I'm not sure how to interpret that query. You can ask about specific dates, festivals, or check Nandhakumar's holidays for today."

@app.event("message")
async def handle_message(event, say):
    text = event.get("text", "")
    response = await get_holiday_info(text)
    await say(response)

@app.event("app_home_opened")
async def handle_app_home_opened_events(body, logger):
    logger.info(body)

async def start_bot():
    handler = AsyncSocketModeHandler(app, APP_TOKEN)
    await handler.start_async()

if __name__ == "__main__":
    asyncio.run(start_bot())
