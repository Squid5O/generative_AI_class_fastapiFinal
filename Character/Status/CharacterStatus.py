from fastapi import APIRouter

router_Status = APIRouter(prefix="/Status", tags=["Status"])

@router_Status.get("/")
async def get_character_status():
    # 캐릭터 상태 조회 로직
    return {"status": "Character status retrieved successfully"}

@router_Status.post("/")
async def update_character_status():
    # 캐릭터 상태 업데이트 로직
    return {"status": "Character status updated successfully"}
