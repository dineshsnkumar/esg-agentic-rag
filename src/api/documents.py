from fastapi import FastAPI

app = FastAPI()

@app.get("/documents")
async def documents():
    return "Returing the list of documents"