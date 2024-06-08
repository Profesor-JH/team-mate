import uvicorn
import socketio
from fastapi import FastAPI
from typing import Dict, List

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
            # this will ectorize the user query in order to proceed with a semantic search
            query_vector = weaviate_interface.vectorizer.embed_text(weaviate_query).tolist()
            flattened_q_vector = [item for sublist in query_vector for item in sublist] # yeah this is because the former query_vector was in double [] and did not work properly to retrive the infos i want from the base
            query = f"""
            {{
                Get {{
                    Job(nearVector: {{
                        vector: {flattened_q_vector},
                        certainty: 0.7
                    }}) {{
                        title
                        company
                        description
                        apply_link
                    }}
                }}
            }}
            """
            # Debug: Print the query just for checking, was getting some errors
            print(f"GraphQL Query: {query}")

            response = await weaviate_interface.client.run_query(query)
            # Debuging: i m printing the raw response - 
            print(f"GraphQL Response: {response}")
            jobs = response.get('data', {}).get('Get', {}).get('Job', [])
            if jobs:
                context = ""
                for job in jobs:
                    context += f"Title: {job['title']}\nCompany: {job['company']}\nDescription: {job['description']}\nApply Link: {job['apply_link']}\n\n"
                
                response_text = await generate_response_with_gpt_stream(weaviate_query, context, sid)
            else:
                response_text = "No relevant jobs found."
        except Exception as e:
            response_text = f"Error fetching jobs: {e}"

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
    await weaviate_interface.async_init()
    #await weaviate_interface.upload_data_from_csv("all_nov_jobs.csv")  # for the first time we can upload data but after comment out to avoid uploading multiple time: waste of
    #print("Uploaded Jobs with vectors")

if __name__ == "__main__":
    asyncio.run(main())
    uvicorn.run("main:app", host="0.0.0.0", port=6789, lifespan="on", reload=True)