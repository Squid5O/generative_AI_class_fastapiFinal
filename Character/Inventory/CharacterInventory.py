from fastapi import APIRouter

router_Inventory = APIRouter(prefix="/inventory", tags=["inventory"])

@router_Inventory.get("/")
async def get_inventory():
    # 인벤토리 조회 로직
    return {"status": "Inventory retrieved successfully"}

@router_Inventory.post("/")
async def update_inventory():
    # 인벤토리 업데이트 로직
    return {"status": "Inventory updated successfully"}
