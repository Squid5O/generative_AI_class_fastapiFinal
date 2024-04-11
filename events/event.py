from fastapi import FastAPI, APIRouter

#이벤트 
router_event = APIRouter(prefix="/events",  tags=['events'])

@router_event.get("/")
def read_user():
    return {"유저" : "유저목록"}

@router_event.post("/sign")
def sign_user():
    return {"유저" : "유저목록"}
