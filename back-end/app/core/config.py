import os
from tkinter import NO
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
load_dotenv(os.path.join(BASE_DIR, '.env'))


class Settings():
    PROJECT_NAME = os.getenv('PROJECT_NAME', 'FASTAPI BASE')
    API_PREFIX = ''
    BACKEND_CORS_ORIGINS = ['*']
    SECURITY_ALGORITHM = 'HS256'
    PORT = int(os.getenv('PORT', 8000))
    NODE_ENV = os.getenv('NODE_ENV', 'development')


settings = Settings()