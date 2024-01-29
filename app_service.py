 
 """
Модуль валидации типов
"""



from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime

from pydantic import BaseModel
from typing import List
import os
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sqlalchemy import create_engine

import pandas as pd
from sqlalchemy import create_engine
from pydantic import BaseModel

class FeedRequest(BaseModel):
    """
    Класс FeedRequest для валидации типов при подаче
    сервису входных признаков запроса ленты.
    """
    user_id: int = 200
    request_time: datetime = datetime(2021, 10, 1)
    posts_limit: int = 5


class PostGet(BaseModel):
    """
    Класс PostGet для валидации типов постов при
    выдаче рекомендаций
    """
    post_id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class FeedResponse(BaseModel):
    """
    Класс FeedResponse для валидации типов отклика при
    выдаче ленты постов
    """
    exp_group: str = ""
    feed: List[PostGet] = []
	
	
	

engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml", pool_size=30)


   ###загрузка модели

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path(os.getcwd())
    from_file = CatBoostClassifier()
    return from_file

### 3 считываем таблицу
#df = pd.read_sql('SELECT * FROM treshcheva_nina_features_lesson_6', con=engine) 
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)
    
    
    
def load_features() -> pd.DataFrame: #которая бы загружала признаки с помощью функции batch_load_sql, которую мы предоставили вам выше. 
    df_features = batch_load_sql('SELECT * FROM treshcheva_nina_features_lesson_21') #загрузка признаков с помощью батчей
    return pd.DataFrame(df_features)





model = load_model()
features = load_features()

app = FastAPI()

def get_recommended_feed(id: int, time: datetime, limit: int):
    user_features = features[1].loc[features[1].user_id == id]
    user_features = user_features.drop('user_id', axis = 1)
    
    posts_features = features[0].drop(['index', 'text'], axis = 1)
    content = features[0][['post_id', 'text', 'topic']]
    

    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    predict = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts
    
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]
 
    recommended_posts = filtered_.sort_values('predicts')[-limit:].index
   
    return [
       PostGet(**{
             'id': i,
             'text': content[content.post_id == i].text.values[0],
             'topic': content[content.post_id == i].topic.values[0]
              }) for i in recommended_posts
    ]



@app.get("/post/recommendations/", response_model=List[PostGet])
def get_post_recommendations(id: int = None, limit: int = 10, db: Session = Depends(get_db)):
    result = db.query(Post).select_from(Feed).filter(Feed.action == 'like').join(Post).group_by(Post.id).order_by(func.count(Post.id).desc()).limit(
        limit).all()
    return result

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int, 
        time: datetime, 
        limit: int = 5) -> List[PostGet]:
     return get_recommended_feed(id, time, limit) 
"""
Обработайте запрос в endpoint, используя модель. Для этого отберите признаки для конкретного user_id и сделайте предсказания. 
Необходимо всегда возвращать 5 постов для каждого юзера.
"""
