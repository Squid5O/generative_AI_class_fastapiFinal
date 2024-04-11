from fastapi import FastAPI, HTTPException # 예외처리를 위한 라이브러리
import pandas as pd

app = FastAPI()

chat_dict = {}

#이미 저장된 파일이 있다면, 유저정보를 가져올 수 있다.
df = pd.read_csv("chat.csv")
print(df)

# 사용자 정보 조회
# @app.post("/text")
# def text_maker(chat_num:int, chat: str):
#     chat_dict[chat_num] = [chat] # 딕셔너리에 데이터 저장
#     df = pd.DataFrame(chat_dict) # 최종 딕셔너리를 데이터프레임으로
#     df.to_csv("chat.csv") # 채팅내역 딕셔너리를 csv파일로 저장
#     return chat_dict

@app.post("/text")
def text_maker(chat_num: int, chat: str):
    chat_dict[chat_num] = chat
    
    #파일 만들어서 채팅내역 저장하고 싶은 경우
    with open('chat.text', 'w', encoding="utf-8") as file:
        file.write(str(chat_num) + "_" +chat+"\n")
    return chat_dict

@app.get("/log")
def chat_log():
    return pd.read_csv("chat.text")

# 사용자 정보를 저장할 딕셔너리
#user_dict = {}

#    user_dict[user_id] = {"username": username, "email": email}
