from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from app.recommend import get_recommendations

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend")
def recommend(request: Request, query: str = Form(...)):
    if not query.strip():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Please enter a valid query."})
    
    recommendations = get_recommendations(query)
    return templates.TemplateResponse("index.html", {"request": request, "query": query, "recommendations": recommendations})