import os
import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# to get a string like this run:
# openssl rand -hex 32

try:
  secfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'pvt', 'conf', 'sec_config.json')
  with open(secfg_file) as sin:
    secfg = json.load(sin)

  users_db = secfg['users']
  SECRET_KEY = secfg['JWT']['secret_key']
  ALGORITHM = secfg['JWT']['algorithm']
  ACCESS_TOKEN_EXPIRE_MINUTES = secfg['JWT']['access_token_expire_minutes']
except Exception as e:
  raise Exception('security loading error : {}'.format(e)) from e

class Token(BaseModel):
  access_token: str
  token_type: str

class TokenData(BaseModel):
  username: Optional[str] = None

class User(BaseModel):
  username: str
  email: Optional[str] = None
  full_name: Optional[str] = None
  disabled: Optional[bool] = None
  role: Optional[str] = None

class UserInDB(User):
  hashed_password: str

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='login')

def verify_password(plain_password, hashed_password):
  return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
  return pwd_context.hash(password)

def get_user(db, username: str):
  if username in db:
    user_dict = db[username]
    return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
  user = get_user(fake_db, username)
  if not user:
    return False
  if not verify_password(password, user.hashed_password):
    return False
  return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
  to_encode = data.copy()
  if expires_delta:
    expire = datetime.utcnow() + expires_delta
  else:
    expire = datetime.utcnow() + timedelta(minutes=15)
  to_encode.update({'exp': expire})
  encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
  return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
  credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail='Could not validate credentials',
    headers={'WWW-Authenticate': 'Bearer'},
  )
  try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(payload)
    username: str = payload.get('sub')
    if username is None:
      raise credentials_exception
    token_data = TokenData(username=username)
  except JWTError as e:
    credentials_exception.detail += f': {e}'
    raise credentials_exception
  user = get_user(users_db, username=token_data.username)
  if user is None:
    raise credentials_exception
  return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
  if current_user.disabled:
    raise HTTPException(status_code=400, detail='Inactive user')
  return current_user
