from fastapi import APIRouter

router_PlayerData = APIRouter(prefix="/playdata", tags=["playdata"])

@router_PlayerData.get("/")
async def get_playdata():
    # 플레이 데이터 조회 로직
    return {"status": "Play data retrieved successfully"}

@router_PlayerData.post("/")
async def collect_playdata():
    # 플레이 데이터 수집 로직
    return {"status": "Play data collected successfully"}
