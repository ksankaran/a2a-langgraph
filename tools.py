import httpx
from typing import Any
from langchain_core.tools import tool

# constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "ksankaran-weather/1.0"

def make_nws_request(url: str) -> dict[str, Any] | None:
	"""
	Make a request to the NWS API and return the response as a dictionary.
	"""
	headers = {
		"User-Agent": USER_AGENT,
		"Accept": "application/geo+json"
	}
	with httpx.Client() as client:
		try:
			response = client.get(url, headers=headers, timeout=20.0, follow_redirects=True)
			response.raise_for_status()
			return response.json()
		except httpx.RequestError as e:
			print(f"Request error: {e}")
		except httpx.HTTPStatusError as e:
			print(f"HTTP error: {e}")
		return None

@tool(description="Get the weather forecast for a given latitude and longitude.")
def get_forecast(lat: float, lon: float) -> str:
	"""
	Get the weather forecast for a given latitude and longitude.
	"""
	url = f"{NWS_API_BASE}/points/{lat},{lon}"
	points_data = make_nws_request(url)
	if not points_data:
		return "No forecast found or an error occurred."
	
	forecast_url = points_data["properties"]["forecast"]
	data = make_nws_request(forecast_url)

	if not data:
		return "Unable to fetch detailed forecast."

	forecast = data["properties"].get("periods", [])
	if not forecast:
		return "No forecast available."

	forecast_messages = [f"{period['name']}: {period['detailedForecast']}" for period in forecast]
	return "\n---\n".join(forecast_messages)