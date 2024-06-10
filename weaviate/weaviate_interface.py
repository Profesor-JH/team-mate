import csv
import pandas as pd
import aiomysql
from datetime import datetime, date,timezone
from weaviate.schema_manager import SchemaManager
from weaviate.weaviate_client import WeaviateClient
from .http_client import HttpClient, HttpHandler
from weaviate.vectorizer import JobVectorizer
import os
import yaml
from dotenv import load_dotenv
import tqdm


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
        

    
    async def upload_data_from_mysql(self, mysql_config: dict) -> None:
        def format_date(date_input) -> str:
            try:
                if isinstance(date_input, (datetime, date)):
                    # Convert to RFC3339 format with time and timezone
                    return datetime.combine(date_input, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat()
                else:
                    raise ValueError("Invalid date format or type")
            except ValueError as e:
                print(f"Invalid date format: {date_input}, Error: {e}")
                return None
            
        uploaded_news = []
        connection = None

        try:

            # Connect to MySQL database
            connection = await aiomysql.connect(
                host=mysql_config['host'],
                user=mysql_config['user'],
                password=mysql_config['password'],
                db=mysql_config['database']
            )

            async with connection.cursor() as cursor:
                await cursor.execute("""
                    SELECT nh.InsertionDate, nh.Ticker, nh.News1, nh.News2, nh.News3, nh.News4, nh.News5, nh.News6, nh.News7, nh.News8, nh.News9, nh.News10,
                        nh.News11, nh.News12, nh.News13, nh.News14, nh.News15, nh.News16, nh.News17, nh.News18, nh.News19, nh.News20,
                        nh.News21, nh.News22, nh.News23, nh.News24, nh.News25, nh.News26, nh.News27, nh.News28, nh.News29, nh.News30,
                        dc.Company, ds.Sector, di.Industry, dco.Country
                    FROM Ratios_Tech.Stocks_News_headlines nh
                    INNER JOIN Dimension_Company dc ON nh.Ticker = dc.Ticker
                    INNER JOIN Dimension_Sector ds ON nh.Ticker = ds.Ticker
                    INNER JOIN Dimension_Industry di ON dc.Ticker = di.Ticker
                    INNER JOIN Dimension_Country dco ON dc.Ticker = dco.Ticker
                    where nh.InsertionDate = '2024-06-08'
                """)
                rows = await cursor.fetchall()

                # Define the required fields for validation
                required_fields = [
                    "insertion_date", "ticker", "company", "sector", "industry", "country",
                    "news1", "news2", "news3", "news4", "news5", "news6", "news7", "news8",
                    "news9", "news10", "news11", "news12", "news13", "news14", "news15",
                    "news16", "news17", "news18", "news19", "news20", "news21", "news22",
                    "news23", "news24", "news25", "news26", "news27", "news28", "news29", "news30"
                ]
                news_fields = [f"news{i}" for i in range(1, 31)]  

                for row in tqdm.asyncio.tqdm(rows):
                    # Print the row to inspect the data
                    print("Row data:", row)

                    # Ensure insertion_date is formatted correctly
                    insertion_date = format_date(row[0])
                    if insertion_date is None:
                        print("Error: Invalid insertion date")
                        continue

                    news_data = {
                        "insertion_date": insertion_date,
                        "ticker": row[1],
                        "company": row[32],
                        "sector": row[33],
                        "industry": row[34],
                        "country": row[35]
                        }
                    # Add news fields
                    for i, news_field in enumerate(news_fields, start=2):
                        news_data[news_field] = str(row[i]) if row[i] is not None else ""

                    # Print news_data to inspect the transformation
                    print("Transformed news_data:", news_data)

                    # Validate required fields and their types
                    missing_fields = [field for field in required_fields if news_data.get(field) is None]
                    if missing_fields:
                        print(f"Error: Missing fields: {missing_fields}")
                        continue

                    invalid_type_fields = [field for field in required_fields if not isinstance(news_data.get(field), str)]
                    if invalid_type_fields:
                        print(f"Error: Fields with invalid types: {invalid_type_fields}")
                        continue



                    # Vectorize news data (you may need to implement vectorize_news method similar to vectorize_job)
                    news_vector = self.vectorizer.vectorize_news(news_data)
                    flattened_vector = [item for sublist in news_vector.tolist() for item in sublist]

                    # Remove keys with None values
                    news_data = {k: v for k, v in news_data.items() if v is not None}

                    try:
                        response = await self.client.create_object(news_data, "StockNews", flattened_vector)
                        print(f"Response from Weaviate: {response}")
                        uploaded_news.append(news_data)
                    except Exception as e:
                        print(f"Error uploading news data: {e}")

        except Exception as e:
            print(f"Error connecting to MySQL database: {e}")

        finally:
            if connection is not None:
                await connection.ensure_closed()

        print(f"Uploaded News: {uploaded_news}")
        return uploaded_news

        

def setup_weaviate_interface() -> WeaviateInterface:
    load_dotenv()

    url = os.getenv("WEAVIATE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables")

    schema_file = "weaviate/schema.json"  # Replace with the path to your schema file
    weaviate_interface = WeaviateInterface(url, openai_key, schema_file)
    return weaviate_interface