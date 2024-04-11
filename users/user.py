from fastapi import FastAPI, APIRouter

#이벤트 
router_users = APIRouter(prefix="/users",  tags=['users'])


@router_users.get("/")
def read_user():
    return {"유저" : "유저목록"}

@router_users.post("/sign")
def sign_user():
    return {"유저" : "유저목록"}