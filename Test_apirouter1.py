from fastapi import FastAPI, APIRouter

app = FastAPI()

@app.get("/")
def root_index():
    return "# 세상에서 가장 안 지켜지는 줄은? 스케줄 ㅋㅋ \n"+ "Book? 북? 김정은 ㅋㅋ \n"+"세상에서 가장 동그란 밭은? 라우터 ㅋㅋ"

# 사용자 관련 기능
router_users = APIRouter(prefix="/users", tags=['users'])

@router_users.get("/")
def read_user():
    return {"유저" : "유저목록"}

@router_users.post("/sign")
def sign_user():
    return {"유저" : "유저목록"}

#이벤트 
router_event = APIRouter(prefix="/events",  tags=['events'])


@router_event.get("/")
def read_user():
    return {"유저" : "유저목록"}

@router_event.post("/sign")
def sign_user():
    return {"유저" : "유저목록"}


#라우터 등록 -> 라우터를 등록해야 메인서버에 등록될 수 있음 
app.include_router(router_users)
app.include_router(router_event)