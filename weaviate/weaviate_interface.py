import csv
import pandas as pd
from datetime import datetime
from weaviate.schema_manager import SchemaManager
from weaviate.weaviate_client import WeaviateClient
from .http_client import HttpClient, HttpHandler
from weaviate.vectorizer import JobVectorizer
import os
from dotenv import load_dotenv



class WeaviateInterface:
    def __init__(self, url: str, openai_key: str, schema_file: str):
        self.http_handler = HttpHandler(HttpClient(url, {"X-OpenAI-Api-Key": openai_key}))
        self.client = WeaviateClient(self.http_handler)
        self.schema = SchemaManager(self.client, schema_file)
        self.vectorizer = JobVectorizer()  # Adding this line for vectorization

    async def async_init(self):
        """
        Asynchronous initialization tasks for WeaviateInterface.
        """
        if not await self.schema.is_valid():
            await self.schema.reset()

#### My function to add the given csv data , i observed some errors due to date format which i have refine now

    async def upload_data_from_csv(self, file_path: str) -> None:
        def format_date(date_str: str) -> str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.isoformat() + "Z"  # Convert the date to RFC3339 format to avoid insertion error later
            except ValueError:
                print(f"Invalid date format: {date_str}")
                return None

        uploaded_jobs = []

        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                job_data = {
                    "title": row.get("title"),
                    "company": row.get("company"),
                    "company_link": row.get("company_link"),
                    "place": row.get("place"),
                    "date": format_date(row.get("date")),
                    "apply_link": row.get("apply_link"),
                    "post_link": row.get("post_link"),
                    "seniority_level": row.get("seniority_level"),
                    "employmnet_type": row.get("employmnet_type"),
                    "description": row.get("description"),
                    "job_title_id": row.get("job_title_id"),
                    "job_desc_id": row.get("job_desc_id")
                }
                # here above, their is no need to add the vector as a field , weavite handle that automatically
                # Validate required fields and their types
                required_fields = [
                    "title", "company", "company_link", "place", "date", 
                    "apply_link", "post_link", "seniority_level", 
                    "employmnet_type", "description", "job_title_id", "job_desc_id"
                ]
                if not all(isinstance(job_data.get(field), str) for field in required_fields):
                    print("Error: One or more required fields are missing or have invalid data types.")
                    continue

                # let's vectorize job data
                job_vector = self.vectorizer.vectorize_job(job_data)
                flattened_vector = [item for sublist in job_vector.tolist() for item in sublist]

                # and remove keys with None values
                job_data = {k: v for k, v in job_data.items() if v is not None}

                try:
                    response = await self.client.create_object(job_data, "Job", flattened_vector)
                    print(f"Response from Weaviate: {response}")
                    uploaded_jobs.append(job_data)
                except Exception as e:
                    print(f"Error uploading job data: {e}")

        print(f"Uploaded Jobs: {uploaded_jobs}")
        return uploaded_jobs
    
    async def get_uploaded_jobs(self):
        query = """
        {
            Get {
                Job {
                    title
                    company
                    company_link
                    place
                    date
                    apply_link
                    post_link
                    seniority_level
                    employmnet_type
                    description
                    job_title_id
                    job_desc_id
                }
            }
        }
        """
        try:
            response = await self.client.run_query(query)
            jobs = response.get('data', {}).get('Get', {}).get('Job', [])
            return jobs
        except Exception as e:
            print(f"Error fetching jobs: {e}")
            return []
        
    async def search_jobs(self, user_query: str):
        query_vector = self.vectorizer.embed_text(user_query)
        query = f"""
        {{
            Get {{
                Job(nearVector: {{vector: {query_vector.tolist()}}}) {{
                    title
                    company
                    description
                    vector
                }}
            }}
        }}
        """
        try:
            response = await self.client.run_query(query)
            jobs = response.get('data', {}).get('Get', {}).get('Job', [])
            return jobs
        except Exception as e:
            print(f"Error searching jobs: {e}")
            return []
        

def setup_weaviate_interface() -> WeaviateInterface:
    load_dotenv()

    url = os.getenv("WEAVIATE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables")

    schema_file = "weaviate/schema.json"  # Replace with the path to your schema file
    weaviate_interface = WeaviateInterface(url, openai_key, schema_file)
    return weaviate_interface