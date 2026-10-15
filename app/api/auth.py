import hmac
from typing import Annotated

from fastapi import Depends, Request

from app.config.settings import TokenIdentity
from app.errors import RuntimeErrorBase


def authenticate(request: Request) -> TokenIdentity:
    parts = request.headers.get("Authorization", "").split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise RuntimeErrorBase("unauthenticated", "A bearer token is required.", 401)
    match = None
    for token, identity in request.app.state.settings.api_tokens.items():
        if hmac.compare_digest(parts[1].encode("utf-8"), token.encode("utf-8")):
            match = identity
    if match is None:
        raise RuntimeErrorBase("unauthenticated", "The bearer token is invalid.", 401)
    return match


Identity = Annotated[TokenIdentity, Depends(authenticate)]
