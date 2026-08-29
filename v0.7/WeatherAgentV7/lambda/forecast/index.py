import json
import urllib.request


def handler(event, context):
    """Gateway Lambda target: args arrive as the event dict per the tool schema."""
    lat, lon = event["latitude"], event["longitude"]
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m"
    )
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())["current"]
