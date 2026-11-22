
import os
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2 import id_token

PROJECT_NUMBER = "449073621753"
REGION = "asia-south1"
SERVICE = "pharma-hitl-dashboard"

IAP_AUDIENCE = (
    f"/projects/{PROJECT_NUMBER}"
    f"/locations/{REGION}"
    f"/services/{SERVICE}"
)

def get_verified_reviewer():
    # Only trust the signed IAP assertion.
    jwt = st.context.headers.get(
        "X-Goog-IAP-JWT-Assertion"
    )

    if not jwt:
        return None

    try:
        claims = id_token.verify_token(
            jwt,
            Request(),
            audience=IAP_AUDIENCE,
            certs_url=(
                "https://www.gstatic.com/"
                "iap/verify/public_key"
            ),
        )

        if claims.get("iss") != (
            "https://cloud.google.com/iap"
        ):
            return None

        email = claims.get("email")
        subject = claims.get("sub")

        if not email or not subject:
            return None

        return {
            "email": email,
            "subject": subject,
        }

    except Exception:
        return None
