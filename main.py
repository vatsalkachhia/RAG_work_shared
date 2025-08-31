from fastapi import FastAPI
from routes import router

app = FastAPI()

# Include the router with all routes
app.include_router(router)

