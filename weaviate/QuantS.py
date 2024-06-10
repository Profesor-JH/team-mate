import uvicorn
import socketio
from fastapi import FastAPI
from typing import Dict, List
import yaml
import aiomysql

import asyncio
import  openai 
import tiktoken
import os
from dotenv import load_dotenv
from weaviate.weaviate_interface import setup_weaviate_interface

# Load environment variables from .env file
load_dotenv()

# Set the TOKENIZERS_PARALLELISM environment variable to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the API key is being retrieved correctly
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

openai.api_key = api_key


# Actually i have 4096 tokens available for gpt 3.5 turbo but i want the max to be little bit less than for precaution
MAX_TOKENS = 3896
CONTEXT_TOKENS_LIMIT = 2000  # 2000 for the context as we need to add our retrive context from weaviate!
OUTPUT_TOKENS_LIMIT = MAX_TOKENS - CONTEXT_TOKENS_LIMIT  # and the remaining tokens for output
MODEL_NAME = "gpt-3.5-turbo-0125"

# Initialize the tokenizer for the specific model
tokenizer = tiktoken.get_encoding("cl100k_base")

# Here is a function to count tokens so that we know we are still within a given limit
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Because i have some limit with gpt 3.5 turbo in interm of number of token to use this function will truncate context to fit within token limits
def truncate_context(query: str, context: str) -> str:
    context_tokens = tokenizer.encode(context)
    if len(context_tokens) > CONTEXT_TOKENS_LIMIT:
        truncated_context = tokenizer.decode(context_tokens[:CONTEXT_TOKENS_LIMIT])
    else:
        truncated_context = context
    prompt = f"""
    You are a job assistant helping users find relevant job opportunities. Provide detailed, relevant, and professionally formatted responses based on the user's query and the given context. Ensure the response is clear, visually appealing, and easy to read by using bullet points, and line breaks to separate sections.

    To enhance the training and educational experience, you are a personalized LLM-enhanced agent. You will serve as a supportive digital assistant to help students manage their tasks effectively, facilitate collaboration, and access necessary resources effortlessly. The solution should focus on specific students tackling specific challenges, such as taking university courses or participating in boot camps. The main objective is to guide trainees toward focusing on specific skills based on the job they want to pursue in the future.

    User query: {query}
    
    Context:
    {truncated_context}
    
    Response:
    """
    return prompt

# This is the main function to generate a response using GPT with streaming and it will be called later once we have the query and the context ready!
async def generate_response_with_gpt_stream(query: str, context: str, sid: str) -> str:
    prompt = truncate_context(query, context)
    response_chunks: List[str] = []

    print(f"Prompt sent to GPT: {prompt}")  # Debugging print

    # OpenAI's streaming response handling so that we can enable generation capabilities in the rssponses
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=OUTPUT_TOKENS_LIMIT,  # Limit the tokens for the output
        stream=True,
    )

    print("Starting to stream response...")  # Debugging print

    for chunk in response:
        print(f"Received chunk: {chunk}")  # Debugging print
        if "delta" in chunk.choices[0] and "content" in chunk.choices[0].delta:
            content = chunk.choices[0].delta["content"]
            response_chunks.append(content)
            print(f"Streaming chunk: {content}")  # Debugging print
            # Stream the partial response back to the client
            await sio.emit("partialResponse", {"textResponse": ''.join(response_chunks)}, room=sid)

    final_response = ''.join(response_chunks)
    print(f"Final response: {final_response}")  # Debugging print
    return final_response

# FastAPI application setting up!
app = FastAPI()
# Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

# Dictionary to store session data
sessions: Dict[str, List[Dict[str, str]]] = {}

# Weaviate Interface setting up!
weaviate_interface = setup_weaviate_interface()

# Print {"Hello":"World"} on localhost:7777
@app.get("/")
def read_root():
    return {"Hello": "World"}

@sio.on("connect")
async def connect(sid, env):
    print("New Client Connected to This id :" + " " + str(sid))

@sio.on("disconnect")
async def disconnect(sid):
    print("Client Disconnected: " + " " + str(sid))

@sio.on("connectionInit")
async def handle_connection_init(sid):
    await sio.emit("connectionAck", room=sid)

@sio.on("sessionInit")
async def handle_session_init(sid, data):
    print(f"===> Session {sid} initialized")
    session_id = data.get("sessionId")
    if session_id not in sessions:
        sessions[session_id] = []
    print(f"**** Session {session_id} initialized for {sid} session data: {sessions[session_id]}")
    await sio.emit("sessionInit", {"sessionId": session_id, "chatHistory": sessions[session_id]}, room=sid)

# Handle incoming chat messages
@sio.on("textMessage")
async def handle_chat_message(sid, data):
    print(f"Message from {sid}: {data}")
    session_id = data.get("sessionId")
    if session_id:
        if session_id not in sessions:
            raise Exception(f"Session {session_id} not found")
        received_message = {
            "id": data.get("id"),
            "message": data.get("message"),
            "isUserMessage": True,
            "timestamp": data.get("timestamp"),
        }
        sessions[session_id].append(received_message)

        # Now, i able to interact with Weaviate to get a response based on the message
        weaviate_query = data.get("message")
        try:
            # Vectorize the input query
            q_vector = weaviate_interface.vectorizer.embed_text(weaviate_query)
            flattened_q_vector = [item for sublist in q_vector.tolist() for item in sublist]

            query = f"""
            {{
                Get {{
                    StockNews(nearVector: {{
                        vector: {flattened_q_vector},
                        certainty: 0.7
                    }}) {{
                        ticker
                        company
                        sector
                        industry
                        country
                        news1
                        news2
                        news3
                        news4
                        news5
                        news6
                        news7
                        news8
                        news9
                        news10
                        news11
                        news12
                        news13
                        news14
                        news15
                        news16
                        news17
                        news18
                        news19
                        news20
                        news21
                        news22
                        news23
                        news24
                        news25
                        news26
                        news27
                        news28
                        news29
                        news30
                    }}
                }}
            }}
            """
            # Debug: Print the query just for checking, was getting some errors
            print(f"GraphQL Query: {query}")

            response = await weaviate_interface.client.run_query(query)
            # Debugging: print the raw response
            print(f"GraphQL Response: {response}")
            stock_news = response.get('data', {}).get('Get', {}).get('StockNews', [])
            if stock_news:
                context = ""
                for news in stock_news:
                    context += (
                        f"Ticker: {news['ticker']}\n"
                        f"Company: {news['company']}\n"
                        f"Sector: {news['sector']}\n"
                        f"Industry: {news['industry']}\n"
                        f"Country: {news['country']}\n"
                        f"News1: {news['news1']}\n"
                        f"News2: {news['news2']}\n"
                        f"News3: {news['news3']}\n"
                        f"News4: {news['news4']}\n"
                        f"News5: {news['news5']}\n"
                        f"News6: {news['news6']}\n"
                        f"News7: {news['news7']}\n"
                        f"News8: {news['news8']}\n"
                        f"News9: {news['news9']}\n"
                        f"News10: {news['news10']}\n"
                        f"News11: {news['news11']}\n"
                        f"News12: {news['news12']}\n"
                        f"News13: {news['news13']}\n"
                        f"News14: {news['news14']}\n"
                        f"News15: {news['news15']}\n"
                        f"News16: {news['news16']}\n"
                        f"News17: {news['news17']}\n"
                        f"News18: {news['news18']}\n"
                        f"News19: {news['news19']}\n"
                        f"News20: {news['news20']}\n"
                        f"News21: {news['news21']}\n"
                        f"News22: {news['news22']}\n"
                        f"News23: {news['news23']}\n"
                        f"News24: {news['news24']}\n"
                        f"News25: {news['news25']}\n"
                        f"News26: {news['news26']}\n"
                        f"News27: {news['news27']}\n"
                        f"News28: {news['news28']}\n"
                        f"News29: {news['news29']}\n"
                        f"News30: {news['news30']}\n\n"
                    )
                
                response_text = await generate_response_with_gpt_stream(weaviate_query, context, sid)
            else:
                response_text = "No relevant stock news found."
        except Exception as e:
            response_text = f"Error fetching stock news: {e}"

        response_message = {
            "id": data.get("id") + "_response",
            "textResponse": response_text,
            "isUserMessage": False,
            "timestamp": data.get("timestamp"),
            "isComplete": True,
        }
        await sio.emit("textResponse", response_message, room=sid)
        sessions[session_id].append(response_message)

        print(f"Message from {sid} in session {session_id}: {data.get('message')}")

    else:
        print(f"No session ID provided by {sid}")

async def main():
    # Load MySQL configuration
    with open("config.yaml", 'r') as stream:
        mysql_config = yaml.safe_load(stream)['database']


    await weaviate_interface.async_init()
    await weaviate_interface.upload_data_from_mysql(mysql_config)
    print("Uploaded Stock News with vectors")

if __name__ == "__main__":
    asyncio.run(main())
    uvicorn.run("main:app", host="0.0.0.0", port=6789, lifespan="on", reload=True)