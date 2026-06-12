import requests

client_id = "184138"
client_secret = "bc7aa21501aa1e9c54e6a87b6afaa9a681dd2787"
refresh_token = "8e1602d0baafaa67123c3f9137bfb7d40f71117f"

response = requests.post(
    "https://www.strava.com/oauth/token",
    data={
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
)

print(response.json())