import requests
from urllib.parse import urlencode, urlparse, parse_qs


CLIENT_ID = "184138"
CLIENT_SECRET = "bc7aa21501aa1e9c54e6a87b6afaa9a681dd2787"
REDIRECT_URI = "https://anger-lung-deceiver.ngrok-free.dev"
SCOPE = "read,activity:read_all"


def build_authorization_url():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "approval_prompt": "force",
        "scope": SCOPE,
    }

    return "https://www.strava.com/oauth/authorize?" + urlencode(params)


def extract_code_from_redirect_url(redirect_url):
    parsed_url = urlparse(redirect_url)
    query_params = parse_qs(parsed_url.query)

    code = query_params.get("code", [None])[0]
    scope = query_params.get("scope", [None])[0]

    return code, scope


def exchange_code_for_token(code):
    response = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
        },
        timeout=20,
    )

    return response.json()