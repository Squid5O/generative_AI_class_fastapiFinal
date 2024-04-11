from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# 1. 데이터 베이스 모델 스키마 정의
class User(Base):
    __tablename__ = 'users' # 테이블 이름
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    

# 2. 데이터베이스 엔진 및 세션 생성
DATABASE_URL = "sqlite:///./Test_fastapi_database.db"
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)


# 3. 데이터 베이스 테이블 생성
Base.metadata.create_all(bind=engine)


###### 이하 코드에서 연결해보기 ######
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

app = FastAPI()
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = sessionLocal()
    try:
        yield db # 요청 처리 동안 데이터베이스 세션
    finally:
        db.close() # 요청 처리가 끝나면 데이터베이스 세션을 닫기

# 1. 회원가입 구현
@app.post("/users")
def create_user(name: str, email: str, db: Session= Depends(get_db)):
    db_user = User(name=name, email=email)
    
    if db.query(User).filter(User.name == name).first():
        raise HTTPException(status_code=404, detail="User not found")
    
    db.add(db_user)
    db.commit()
    return {"name": db_user.name, "email" : email}

# 1. 회원가입 구현
# @app.post("/users/")
# def create_user(name: str, email: str, db: Session = Depends(get_db)):
#     db_user = User(name=name, email=email)
#     db.add(db_user)
#     db.commit()
#     return {"name": name, "email": email}

# 2. 전체 회원 조회 구현
@app.get("/users/")
def read_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

# 3. 회원 데이터 수정
@app.put("/users/{user_id}")
def update_user(user_id: int, name: str, email: str, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db_user.name = name
    db_user.email = email
    db.commit()
    db.refresh(db_user)
    return {"id": user_id, "name": name, "email": email}

# 4. 회원 데이터 삭제
@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    return {"message": "User deleted successfully"}
