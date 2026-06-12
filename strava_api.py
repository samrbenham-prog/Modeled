import requests
import pandas as pd

def get_strava_activities(access_token, max_pages=5):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    activities = []

    for page in range(1, max_pages + 1):
        print(f"Pulling page {page}...")

        response = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers=headers,
            params={
                "page": page,
                "per_page": 200
            },
            timeout=20
        )

        print("Status code:", response.status_code)
        print("Rate Limit Headers:")
        print("Limit:", response.headers.get("X-RateLimit-Limit"))
        print("Usage:", response.headers.get("X-RateLimit-Usage"))

        if response.status_code == 429:
            print("Rate limit exceeded. Wait and try again.")
            break

        if response.status_code != 200:
            print("Error:")
            print(response.text)
            break

        batch = response.json()

        print("Activities on this page:", len(batch))

        if len(batch) == 0:
            break

        activities.extend(batch)

        if len(batch) < 200:
            break

    return pd.DataFrame(activities)