from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root_index():
    return "Hello World"

@app.get("/user")
def root_index2():
    return "저는 유저입니다."

@app.get("/book")
def get_books():
    return {"책 목록" : ["인생 언리얼 교과서", "Hey, 파이썬! 생성형 AI 활용 맵 만들어줘"]}