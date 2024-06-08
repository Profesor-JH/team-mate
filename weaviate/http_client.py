import httpx
from typing import Any, Dict, Optional

class HttpClient:
    def __init__(self, base_url: str, headers: Dict[str, str]) -> None:
        self.base_url = base_url
        self.headers = headers
        self.client = httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> httpx.Response:
        url = f"{self.base_url}{endpoint}"
        response = await self.client.request(method, url, headers=self.headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP error responses
        return response

class HttpHandler:
    def __init__(self, http_client: HttpClient) -> None:
        self.http_client = http_client

    async def get_json_response(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Any:
        try:
            response = await self.http_client.make_request(method, endpoint, data)
            if response.text:
                json_response = response.json()
            else:
                json_response = {}
            return json_response
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise e
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            raise e
        except ValueError as e:
            print(f"Value error occurred while parsing JSON response: {e}")
            raise e

