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

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. 데이터 베이스 테이블 생성
Base.metadata.create_all(bind=engine)
