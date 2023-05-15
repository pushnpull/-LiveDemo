import csv
import io
import os
import re
import math
import pickle
import asyncio
import concurrent.futures
import mysql.connector
import pandas as pd
import numpy as np
from fastapi import FastAPI,BackgroundTasks, Response
import mysql.connector
from database import execute_query
from sqlalchemy import create_engine
from pydantic import BaseModel
from bs4 import BeautifulSoup

from scipy.sparse import csr_matrix

from lightfm.data import Dataset
from lightfm import LightFM
from lightfm import cross_validation
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

dataset = Dataset()
model = LightFM(no_components=50, loss='warp')

knn_model = None
item_representation=None
item_representation_masked=None
mask=None
# to feed model
item_features = None
user_features = None


#a map for user_id to index in the model
usermap={}
#id list of unique users 
unique_users=[]
#id list of unique items
live_item_list=[]
#dataframe of unique items
live_item_df = None

highest_user_id = None
highest_item_id = None

# Establish a connection to MySQL
def cnxs():
    return mysql.connector.connect(user='root', password='root',
                              host='localhost', port=3306,
                              database='chilling')


# create an SQLAlchemy engine
engine = create_engine("mysql+pymysql://root:root@localhost:3306/chilling")



likes,likes_created_at,likes_updated_at = None,'1000-01-01 00:00:00', '1000-01-01 00:00:00'
favourites, favourites_created_at, favourites_updated_at= None,'1000-01-01 00:00:00', '1000-01-01 00:00:00'
audio,audio_created_at, audio_updated_at = None,'1000-01-01 00:00:00', '1000-01-01 00:00:00'
users,users_created_at, users_updated_at = None,'1000-01-01 00:00:00', '1000-01-01 00:00:00'
likes_fav,likes_fav_created_at, likes_fav_updated_at = None,'1000-01-01 00:00:00', '1000-01-01 00:00:00'


app = FastAPI()




global_variables = {
    'dataset': 'dataset.pkl',
    'model': 'model.pkl',
    'knn_model': 'knn_model.pkl',
    'item_representation': 'item_representation.pkl',
    'item_representation_masked': 'item_representation_masked.pkl',
    'mask': 'mask.pkl',
    'item_features': 'item_features.pkl',
    'user_features': 'user_features.pkl',
    'usermap': 'usermap.pkl',
    'unique_users': 'unique_users.pkl',
    'live_item_list': 'live_item_list.pkl',
    'live_item_df': 'live_item_df.pkl',
    'highest_user_id': 'highest_user_id.pkl',
    'highest_item_id': 'highest_item_id.pkl',
    'likes': 'likes.pkl',
    'likes_created_at': 'likes_created_at.pkl',
    'likes_updated_at': 'likes_updated_at.pkl',
    'favourites': 'favourites.pkl',
    'favourites_created_at': 'favourites_created_at.pkl',
    'favourites_updated_at': 'favourites_updated_at.pkl',
    'audio': 'audio.pkl',
    'audio_created_at': 'audio_created_at.pkl',
    'audio_updated_at': 'audio_updated_at.pkl',
    'users': 'users.pkl',
    'users_created_at': 'users_created_at.pkl',
    'users_updated_at': 'users_updated_at.pkl',
    'likes_fav': 'likes_fav.pkl',
    'likes_fav_created_at': 'likes_fav_created_at.pkl',
    'likes_fav_updated_at': 'likes_fav_updated_at.pkl'
}

for var_name, file_name in global_variables.items():
    file_path = os.path.join(os.getcwd(), "pickles", file_name)
    if os.path.isfile(file_path):
        try:
            with open(file_path, "rb") as f:
                globals()[var_name] = pickle.load(f)
                print(f"Loaded pickled file: {file_path}")
        except ModuleNotFoundError:
            with open(file_path, "rb") as f:
                globals()[var_name] = pd.read_pickle(f)
                print(f"Loaded pickled file: {file_path}")
        except pickle.UnpicklingError:
            print(f"Error: File is corrupted: {file_path}")
    else:
        print(f"Error: File not found: {file_path}")




@app.get("/load-eveything")
async def load_everything():
    global global_variables

    for var_name, file_name in global_variables.items():
        file_path = os.path.join(os.getcwd(), "pickles", file_name)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    globals()[var_name] = pickle.load(f)
                    print(f"Loaded pickled file: {file_path}")
            except ModuleNotFoundError:
                with open(file_path, "rb") as f:
                    globals()[var_name] = pd.read_pickle(f)
                    print(f"Loaded pickled file: {file_path}")
            except pickle.UnpicklingError:
                print(f"Error: File is corrupted: {file_path}")
        else:
            print(f"Error: File not found: {file_path}")


@app.get("/likes-data")
async def likes_data():
    global likes,likes_created_at, likes_updated_at

    with engine.connect() as con:
        likes = pd.read_sql('SELECT * FROM likes', con)
    
    # Update variables to keep track of last updated row
    likes_created_at = likes['created_at'].max()
    likes_updated_at = likes['updated_at'].max()

    _ = ['likes','likes_created_at', 'likes_updated_at']
    for var in _: 
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")
    # Return success message
    return {"message": "Data saved to file successfully!"}


@app.get("/update-likes")
async def update_likes():
    global likes, likes_created_at, likes_updated_at

    # Connect to MySQL database
    with engine.connect() as con:
        query = f"SELECT * FROM likes WHERE created_at > '{likes_created_at}' OR updated_at > '{likes_updated_at}'"
        df= pd.read_sql(query, con)

    if df.empty:
        return {"message": "No new data to update!"}
    
    likes = pd.concat([likes, df], ignore_index=True).drop_duplicates(subset=['user_id', 'audio_id'], keep='last')

    # Update variables to keep track of last updated row
    likes_created_at = likes['created_at'].max()
    likes_updated_at = likes['updated_at'].max()

    _ = ['likes','likes_created_at', 'likes_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")
    # Return success message
    return {"message": "Data saved to file successfully!"}


@app.get("/favorites-data")
async def favorites_data():
    global favourites, favourites_created_at, favourites_updated_at

    with engine.connect() as con:
        favorites = pd.read_sql('SELECT * FROM favourites', con)
    
    # Update variables to keep track of last updated row
    favourites_created_at = favourites['created_at'].max()
    favourites_updated_at = favourites['updated_at'].max()
    print(favourites.shape[0])
    # Return success message

    _ = ['favourites','favourites_created_at', 'favourites_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")

    return {"message": "Data saved to file successfully!"}


@app.get("/update-favorites")
async def update_favorites():
    global favourites, favourites_created_at, favourites_updated_at

    # Connect to MySQL database
    with engine.connect() as con:
        query = f"SELECT * FROM favourites WHERE created_at > '{favourites_created_at}' OR updated_at > '{favourites_updated_at}'"
        df= pd.read_sql(query, con)

    if df.empty:
        return {"message": "No new data to update!"}
    
    favourites = pd.concat([favourites, df], ignore_index=True).drop_duplicates(subset=['user_id', 'audio_id'], keep='last')


    print(favourites.shape[0])
    # Update variables to keep track of last updated row
    favourites_created_at = favourites['created_at'].max()
    favourites_updated_at = favourites['updated_at'].max()

    _ = ['favourites','favourites_created_at', 'favourites_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")

    # Return success message
    return {"message": "Data saved to file successfully!"}


@app.get("/audio-data")
async def audio_data():
    global audio, audio_created_at, audio_updated_at

    with engine.connect() as con:
        audio = pd.read_sql('SELECT * FROM audio', con)

    audio['tags'] = audio['tags'].apply(lambda x: re.findall(r'\b\w+(?:\.\w+)*\b', x))
    audio['author'] = audio['author'].apply(lambda x: [BeautifulSoup(x, 'html.parser').get_text()])
    audio['narrator'] = audio['narrator'].apply(lambda x: [BeautifulSoup(x, 'html.parser').get_text()])
    audio['title'] = audio['title'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    audio['description'] = audio['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().split())
    # Update variables to keep track of last updated row
    audio['all'] = audio['tags'] + audio['author'] + audio['narrator']
    audio_created_at = audio['created_at'].max()
    audio_updated_at = audio['updated_at'].max()
    
    print(audio.shape[0])
    print(audio.head())
    # Return success message
    _ = ['audio','audio_created_at', 'audio_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")
    return {"message": "Data saved to file successfully!"}


@app.get("/update-audio")
async def update_audio():
    global audio, audio_created_at, audio_updated_at

    # Connect to MySQL database
    with engine.connect() as con:
        query = f"SELECT * FROM audio WHERE created_at > '{audio_created_at}' OR updated_at > '{audio_updated_at}'"
        df= pd.read_sql(query, con)

    if df.empty:
        return {"message": "No new data to update!"}
    
    audio = pd.concat([audio, df], ignore_index=True).drop_duplicates(subset=['id'], keep='last')
    audio['tags'] = audio['tags'].apply(lambda x: re.findall(r'\b\w+(?:\.\w+)*\b', x))
    audio['author'] = audio['author'].apply(lambda x: [BeautifulSoup(x, 'html.parser').get_text()])
    audio['narrator'] = audio['narrator'].apply(lambda x: [BeautifulSoup(x, 'html.parser').get_text()])
    audio['title'] = audio['title'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    audio['description'] = audio['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text().split())
    audio['all'] = audio['tags'] + audio['author'] + audio['narrator']

    print(audio.shape[0])
    # Update variables to keep track of last updated row
    audio_created_at = audio['created_at'].max()
    audio_updated_at = audio['updated_at'].max()

    _ = ['audio','audio_created_at', 'audio_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")
    # Return success message
    return {"message": "Data saved to file successfully!"}



@app.get("/users-data")
async def users_data():
    global users, users_created_at, users_updated_at

    with engine.connect() as con:
        query='SELECT id, fullname, phone_number, email, status, created_at, updated_at FROM users'
        users = pd.read_sql(query, con)
    
    # Update variables to keep track of last updated row
    users_created_at = users['created_at'].max()
    users_updated_at = users['updated_at'].max()
    print(users.shape[0])
    print(users.head())
    # Return success message
    _ = ['users','users_created_at', 'users_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")

    return {"message": "Data saved to file successfully!"}


@app.get("/update-users")
async def update_users():
    global users, users_created_at, users_updated_at

    # Connect to MySQL database
    with engine.connect() as con:
        query = f"SELECT id, fullname, phone_number, email, status, created_at, updated_at FROM users WHERE created_at > '{users_created_at}' OR updated_at > '{users_updated_at}'"
        df= pd.read_sql(query, con)

    if df.empty:
        return {"message": "No new data to update!"}
    
    users = pd.concat([users, df], ignore_index=True).drop_duplicates(subset=['id'], keep='last')
    print(users.shape[0])
    # Update variables to keep track of last updated row
    users_created_at = users['created_at'].max()
    users_updated_at = users['updated_at'].max()

    _ = ['users','users_created_at', 'users_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")
    # Return success message
    return {"message": "Data saved to file successfully!"}


@app.get("/likes-favourites")
async def likes_favourites():
    global likes_fav, likes_fav_created_at, likes_fav_updated_at

    with engine.connect() as con:
        query = """
        SELECT likes.user_id, likes.audio_id, 
        CASE WHEN likes.status = 2 THEN -1 ELSE likes.status END AS status, 
        likes.created_at,likes.updated_at 
        FROM likes LEFT OUTER JOIN favourites 
        ON likes.user_id = favourites.user_id AND likes.audio_id = favourites.audio_id AND likes.status = favourites.status 
        WHERE (likes.is_deleted = 0) AND likes.status != 0 

        UNION 

        SELECT favourites.user_id, favourites.audio_id, favourites.status, favourites.created_at, favourites.updated_at 
        FROM likes RIGHT OUTER JOIN favourites 
        ON likes.user_id = favourites.user_id AND likes.audio_id = favourites.audio_id AND likes.status = favourites.status 
        WHERE likes.user_id IS NULL AND (favourites.is_deleted = 0) AND favourites.status != 0 ;
       
        """
        likes_fav = pd.read_sql(query, con)

    # Update variables to keep track of last updated row
    likes_fav_created_at = likes_fav['created_at'].max()
    likes_fav_updated_at = likes_fav['updated_at'].max()
    print(likes_fav.shape[0])
    print(likes_fav.head())
    # Return success message
    _ = ['likes_fav','likes_fav_created_at', 'likes_fav_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")

    return {"message": "Data saved to file successfully!"}


@app.get("/update-likes-fav")
async def update_likes_fav():
    global likes_fav, likes_fav_created_at, likes_fav_updated_at

    # Connect to MySQL database
    with engine.connect() as con:
        query = f"""
        SELECT likes.user_id, likes.audio_id, 
        CASE WHEN likes.status = 2 THEN -1 ELSE likes.status END AS status, 
        likes.created_at,likes.updated_at 
        FROM likes LEFT OUTER JOIN favourites 
        ON likes.user_id = favourites.user_id AND likes.audio_id = favourites.audio_id AND likes.status = favourites.status 
        WHERE (likes.is_deleted = 0) AND likes.status != 0 AND (likes.created_at > '{likes_fav_created_at}' OR likes.updated_at > '{likes_fav_updated_at}') 

        UNION 

        SELECT favourites.user_id, favourites.audio_id, favourites.status, favourites.created_at, favourites.updated_at 
        FROM likes RIGHT OUTER JOIN favourites 
        ON likes.user_id = favourites.user_id AND likes.audio_id = favourites.audio_id AND likes.status = favourites.status 
        WHERE likes.user_id IS NULL AND (favourites.is_deleted = 0) AND favourites.status != 0 
        AND (favourites.created_at > '{likes_fav_created_at}' OR favourites.updated_at > '{likes_fav_updated_at}') ;
       
        """
        df= pd.read_sql(query, con)

    if df.empty:
        return {"message": "No new data to update!"}
    
    likes_fav = pd.concat([likes_fav, df], ignore_index=True).drop_duplicates(subset=['user_id', 'audio_id'], keep='last')
    print(likes_fav.shape[0])
    # Update variables to keep track of last updated row
    likes_fav_created_at = likes_fav['created_at'].max()
    likes_fav_updated_at = likes_fav['updated_at'].max()

    _ = ['likes_fav','likes_fav_created_at', 'likes_fav_updated_at']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")
    # Return success message
    return {"message": "Data saved to file successfully!"} 


async def alluser():
    #since we have likes and fav data of deleted users we nedd to merge deleted users id to main users table
    
    global users,likes_fav,usermap ,unique_users ,highest_user_id
    alls=pd.concat([users['id'], likes_fav['user_id']], ignore_index=True)
    print(len(alls))
    alls = alls.drop_duplicates().reset_index(drop=True)
    print(len(alls))
    unique_users = alls.unique()
    #we will use one more userid as a buffer for new users
    highest_user_id = (unique_users.max() +1)
    alls[alls.index.max()+1] = highest_user_id
    print("----------------------------------", highest_user_id ,"-----------------------------------")
    # alls = alls.append(pd.Series(highest_user_id),ignore_index=True)
    #this them does not have highest_user_id data
    temp = alls.to_dict()
    #updated unique_users with highest_user_id to feed dataset.fit
    unique_users = alls.unique()
    usermap = {value: key for key, value in temp.items()}

    _ = ['unique_users','highest_user_id','usermap']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")
 

async def item_user_feature_list_and_datasetfit():
    global audio,dataset,unique_users,live_item_list,live_item_df,highest_item_id

    live_item_list=audio.loc[(audio['status'] == 1) & (audio['is_deleted'] == 0), 'id'].unique()
    live_item_df=audio.loc[(audio['status'] == 1) & (audio['is_deleted'] == 0), ['id','title','view_count','title','author','narrator']]
# list of all item features ,apperently it is user features also
    highest_item_id=audio['id'].max()
    # highest_item_id = (live_item_df[id].max())
    print(len(live_item_list))
    for i in range(0,len(live_item_list),10):
        print(*live_item_list[i:i+10])
    
    item_features_list = audio['all'].apply(pd.Series).stack().reset_index(drop=True)

    #
    
    dataset.fit(users=unique_users 
            ,items=[x for x in range(audio['id'].max()+1)]
            ,item_features=item_features_list 
            ,user_features=item_features_list)
    print(dataset)
    _ = ['live_item_list','live_item_df','highest_item_id']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")

async def item_feature_generation():
    global audio,dataset,item_features

    item_features_raw = list(zip(audio['id'], audio['all']))

    item_features = dataset.build_item_features(item_features_raw)
    print(item_features)

    _ = ['item_features']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")


# async def user_feature_generation():
#     global users,audio,likes_fav,dataset,user_features

#     likes_fav_grouped = likes_fav.groupby("user_id")["audio_id"].apply(list).reset_index()
#     print(likes_fav_grouped.head())
#     tags_list = []
#     for audio_id in likes_fav_grouped['audio_id']:
#         tags = audio.loc[audio['id'].isin(audio_id)]['all'].to_list()
#         placeholder=[]
#         for tag in tags:
#             placeholder.extend(tag)
#         placeholder=set(placeholder)
#         tags_list.append([*placeholder])
    
#     likes_fav_grouped['tags'] = tags_list

#     user_features_raw = list(zip(likes_fav_grouped['user_id'], likes_fav_grouped['tags']))
#     print(len(user_features_raw))
#     user_features = dataset.build_user_features(user_features_raw)
#     print(user_features)


async def user_feature_generation():
    global users, audio, likes_fav, dataset, user_features

    likes_fav_grouped = likes_fav.groupby("user_id")["audio_id"].apply(list).reset_index()
    print(likes_fav_grouped.head())

    # get tags for all audio
    audio_tags = audio.set_index('id')['all'].to_dict()

    tags_list = []
    for audio_id_list in likes_fav_grouped['audio_id']:
        tags = set()
        for audio_id in audio_id_list:
            tags.update(audio_tags[audio_id])
        tags_list.append(list(tags))

    likes_fav_grouped['tags'] = tags_list

    user_features_raw = list(zip(likes_fav_grouped['user_id'], likes_fav_grouped['tags']))
    print(len(user_features_raw))
    user_features = dataset.build_user_features(user_features_raw)
    print(user_features)

    _ = ['user_features']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")


async def interaction_matrix_generation():
    global likes_fav,dataset,interactions,item_features,user_features,model

    values = likes_fav['status'].values
    rows = likes_fav['user_id'].values
    cols = likes_fav['audio_id'].values
    ratings_matrix = csr_matrix((values, (rows, cols)))

    interactions, weights = dataset.build_interactions(
    (rows[i], cols[i], values[i]) for i in range(ratings_matrix.nnz)
)
    model.fit(interactions, item_features=item_features, user_features=user_features, sample_weight=weights,epochs=5)
    print(model)
    _ = ['dataset','model']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")

#todo
async def knn():
    global live_item_list, item_features, dataset , model , knn_model ,item_representation ,mask ,item_representation_masked
    from sklearn.neighbors import NearestNeighbors
    knn_model = NearestNeighbors(n_neighbors=math.floor(math.sqrt(len(live_item_list))), metric='cosine')
    
    #since we are using likes and fav data of deleted users we have item representation of deleted items also, we have to trim it down to live items
    item_representation = item_features.dot(model.item_embeddings)


    # i think i have to provide whole item representation to knn model
    # because then index of item representation MIGHT(i can't be sure) be different
    # todo look into it
    
    # print(len(item_representation))
    # #trimming down item representation to live items
    items={x for x in range(audio['id'].max()+1)}
    # print(len(items))
    live_item_list_set=set(live_item_list)
    # print(len(live_item_list_set))
    deleted_items=items-live_item_list_set
    # print(len(deleted_items))
    mask = np.ones(audio['id'].max()+1)
    mask[list(deleted_items)] = 0
    item_representation_masked = item_representation[mask.astype(bool)]

    # item_representation_for_knn=np.delete(item_representation, list(deleted_items), axis=0)
    # print(len(item_representation_for_knn))
    knn_model.fit(item_representation_masked)

    _ = ['knn_model','item_representation','item_representation_masked','mask']
    for var in _:
        file_path = os.path.join(os.getcwd(), "pickles", f"{var}.pkl")
        with open(file_path, "w+b") as f:
            pickle.dump(globals()[var], f)
            print(f"Saved pickled file: {file_path}")

#todo
def convert_itemlist_to_vector_list(item_list):
    global item_representation_masked
    vector_list = [item_representation_masked[x] for x in item_list]
    return vector_list

def calculate_vector(vector_list):
    n = len(vector_list)
    weights = {
        1: [1],
        2: [0.4, 0.6],
        3: [0.2222222222222, 0.33333333333333, 0.4444444444444],
        4: [0.14285714285714285, 0.2142857142857143, 0.2857142857142857, 0.3571428571428571],
        5: [0.1, 0.15, 0.2, 0.25, 0.3],
        6: [0.07407407407407406,0.11111111111111112,0.1481481481481481,0.18518518518518517,0.2222222222222222,0.25925925925925924],
        7: [0.057142857142857155,0.08571428571428574,0.11428571428571431,0.14285714285714285,0.17142857142857146,0.20000000000000004,0.22857142857142856],
        8: [0.04545454545454546,0.06818181818181819,0.09090909090909091,0.11363636363636363,0.13636363636363638,0.1590909090909091,0.18181818181818182,0.2045454545454546],
        9: [0.03703703703703704,0.05555555555555556,0.07407407407407406,0.0925925925925926,0.11111111111111112,0.12962962962962965,0.1481481481481481,0.16666666666666669,0.18518518518518517],
        10: [0.030769230769230774,0.046153846153846156,0.06153846153846154,0.07692307692307691,0.09230769230769231,0.10769230769230768,0.12307692307692306,0.1384615384615385,0.15384615384615388,0.16923076923076924],
        11: [0.02597402597402598,0.03896103896103897,0.05194805194805195,0.06493506493506493,0.07792207792207792,0.09090909090909091,0.10389610389610389,0.11688311688311691,0.1298701298701299,0.14285714285714288,0.15584415584415587],
        12: [0.022222222222222223,0.03333333333333333,0.044444444444444446,0.05555555555555555,0.06666666666666667,0.07777777777777778,0.08888888888888889,0.1,0.11111111111111112,0.12222222222222223,0.13333333333333333,0.14444444444444443],
        13: [0.019230769230769232,0.02884615384615385,0.038461538461538464,0.04807692307692308,0.057692307692307696,0.0673076923076923,0.07692307692307693,0.08653846153846154,0.09615384615384616,0.10576923076923077,0.11538461538461539,0.125,0.1346153846153846],
        14: [0.01680672268907563,0.025210084033613446,0.03361344537815126,0.04201680672268907,0.050420168067226885,0.0588235294117647,0.0672268907563025,0.07563025210084034,0.08403361344537817,0.09243697478991597,0.10084033613445378,0.1092436974789916,0.11764705882352938,0.12605042016806722],
        15: [0.014814814814814819,0.022222222222222227,0.02962962962962963,0.037037037037037035,0.044444444444444446,0.05185185185185185,0.059259259259259255,0.06666666666666668,0.0740740740740741,0.08148148148148149,0.0888888888888889,0.09629629629629631,0.10370370370370369,0.11111111111111112,0.11851851851851852],
        16: [0.013157894736842105,0.019736842105263158,0.02631578947368421,0.03289473684210526,0.039473684210526314,0.04605263157894737,0.05263157894736842,0.05921052631578947,0.06578947368421052,0.07236842105263158,0.07894736842105263,0.08552631578947368,0.09210526315789473,0.09868421052631578,0.10526315789473684,0.11184210526315789],
        17: [0.011764705882352941,0.01764705882352941,0.023529411764705882,0.029411764705882353,0.03529411764705882,0.04117647058823529,0.047058823529411764,0.05294117647058824,0.0588235294117647,0.06470588235294118,0.07058823529411765,0.07647058823529412,0.08235294117647059,0.08823529411764706,0.09411764705882353,0.1,0.10588235294117647],
        18: [0.010582010582010583,0.015873015873015876,0.021164021164021163,0.026455026455026454,0.031746031746031744,0.037037037037037035,0.042328042328042326,0.04761904761904763,0.05291005291005292,0.058201058201058205,0.0634920634920635,0.06878306878306878,0.07407407407407406,0.07936507936507936,0.08465608465608465,0.08994708994708996,0.09523809523809522,0.10052910052910051],
        19: [0.009569377990430623,0.014354066985645933,0.019138755980861243,0.02392344497607655,0.028708133971291867,0.03349282296650718,0.03827751196172248,0.04306220095693781,0.047846889952153124,0.05263157894736843,0.05741626794258373,0.06220095693779905,0.06698564593301434,0.07177033492822966,0.07655502392344497,0.0813397129186603,0.08612440191387559,0.0909090909090909,0.0956937799043062],
        20: [0.008695652173913045,0.01304347826086957,0.017391304347826087,0.02173913043478261,0.026086956521739132,0.030434782608695653,0.034782608695652174,0.03913043478260871,0.04347826086956523,0.04782608695652175,0.052173913043478265,0.05652173913043479,0.0608695652173913,0.06521739130434784,0.06956521739130435,0.07391304347826089,0.07826086956521738,0.08260869565217391,0.08695652173913043,0.09130434782608697],
    }
    weighted_avg = np.average(vector_list, axis=0, weights=weights[n])
    return weighted_avg    

def redis_guard_for_new_items(item_id):
    global highest_item_id
    return True if item_id <= highest_item_id else False


def fetch_knn(vector):
    global knn_model , mask
    distances, indices = knn_model.kneighbors(vector)
    indices_masked = indices[:, mask[indices[0]].astype(bool)]
    return indices_masked

def predict_knn(indices,user_id):
    global usermap,model,item_features,user_features,highest_user_id
    predictions = model.predict(user_ids=usermap.get(user_id,usermap[int(highest_user_id)]), item_ids=indices,user_features=user_features,item_features=item_features)
    return predictions

def predict_all_items_sorted(user_id):
    global usermap,model,item_features,user_features,highest_user_id,live_item_list
    predictions = model.predict(user_ids=usermap.get(user_id,usermap[int(highest_user_id)]), item_ids=live_item_list,user_features=user_features,item_features=item_features)
    sorted_predictions = [int(y) for x,y in sorted(zip(predictions, live_item_list),reverse=True)]
    return sorted_predictions

def print_audio(indices,predictions):
    global audio
    selected_rows = audio.loc[audio["id"].isin(indices),['id','view_count','title','author','narrator']]
    # print(selected_rows)
    selected_rows['predictions'] = predictions
    selected_rows = selected_rows.sort_values(by=['predictions'],ascending=False).head(10)
    print(selected_rows[['id','view_count','title','author','narrator','predictions']])

async def initialize_model():
    await users_data()
    await likes_favourites()
    await audio_data()
    await alluser()
    await item_user_feature_list_and_datasetfit()
    await item_feature_generation()
    await user_feature_generation()
    await interaction_matrix_generation()
    await knn()

async def update_model():
    await update_users()
    await update_likes_fav()
    await update_audio()
    await alluser()
    await item_user_feature_list_and_datasetfit()
    await item_feature_generation()
    await user_feature_generation()
    await interaction_matrix_generation()
    await knn()


# @app.get("/init")
# async def finalize(initialize_models: BackgroundTasks):

#     asyncio.create_task(initialize_model())
    
#     return {"message": "tasks started in the background"}


@app.get("/init")
async def finalize(initialize_models: BackgroundTasks):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, asyncio.run, update_model())
    
    return {"message": "tasks started in the background"}


@app.get("/knn")
async def knns():
    await knn()
    return {"message": "ruk bhai"}


class Item(BaseModel):
    user_id: int
@app.post("/predict")
async def predict(item:Item):
    user_id= item.user_id

    global model,user_features,item_features,usermap,live_item_list,live_item_df,highest_user_id
    # try:
    #     predictions = model.predict(user_ids=usermap[user_id], item_ids=live_item_list,user_features=user_features,item_features=item_features)
    # except Exception as e:
    #     print(str(e))
    predictions = model.predict(user_ids=usermap.get(user_id,highest_user_id), item_ids=live_item_list,user_features=user_features,item_features=item_features)
    temp = live_item_df.copy()
    temp['score'] = predictions
    temp = temp.sort_values(by='score', ascending=False, inplace=False)

    return {"predictions":temp[['id','title','score']].head(30).values.tolist()}    





from socket_server import sio_app
app.mount("/", sio_app)
