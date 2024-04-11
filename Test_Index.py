from fastapi import FastAPI
import uvicorn
from Character.Chat.CharacterChat import router_Chat
from Character.Inventory.CharacterInventory import router_Inventory
from Character.PlayerData.CharacterPlayerData import router_PlayerData
from Character.Status.CharacterStatus import router_Status

app = FastAPI()
@app.get("/")
def index_read():
    return "서버 가동중입니다."

app.include_router(router_Chat)
app.include_router(router_Inventory)
app.include_router(router_PlayerData)
app.include_router(router_Status)


if __name__ == "__main__":
    uvicorn.run(app, port=8001)