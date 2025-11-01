# backend/auth.py
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import os

SECRET_KEY = os.environ.get("JWT_SECRET", "supersecretkey")
ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
MAX_BCRYPT_BYTES = 72
def hash_password(password: str):
    """
    Hash password using bcrypt.
    Truncate password to 72 bytes to avoid bcrypt limitation.
    Handles Unicode characters safely.
    """
    # Encode to UTF-8 bytes
    password_bytes = password.encode("utf-8")
    # print(password_bytes,"vpassword_bytes")
    # Truncate to max 72 bytes
    if len(password_bytes) > MAX_BCRYPT_BYTES:
        password_bytes = password_bytes[:MAX_BCRYPT_BYTES]
        # Decode back to string safely (ignore incomplete bytes at the end)
        password = password_bytes.decode("utf-8", errors="ignore")
        print(password,"password1")
    print(pwd_context.hash(password),"passwordhash")
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > MAX_BCRYPT_BYTES:
        password_bytes = password_bytes[:MAX_BCRYPT_BYTES]
        password_bytes = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.verify(password_bytes, hashed_password)


def create_access_token(data: dict, expires_minutes: int = 60*24*7):
    to_encode = data.copy()
    # print(to_encode,"to_encode")
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    # print(encoded_jwt,"encoded_jwt")
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(payload,"payload")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
