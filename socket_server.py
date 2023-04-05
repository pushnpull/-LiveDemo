import socketio
import redis
from timeit import default_timer as timer
import numpy as np
from finals import convert_itemlist_to_vector_list,calculate_vector,fetch_knn , predict_knn,predict_all_items_sorted,print_audio,redis_guard_for_new_items
# from redistimeseries.client import Client
# rts = Client()
day_in_ms = 86400000
sio_server = socketio.AsyncServer(async_mode='asgi',cross_allowed_origins=[])
sio_app = socketio.ASGIApp(socketio_server=sio_server,socketio_path='/sockets')

r = redis.Redis(host='192.168.216.207', port=6379, db=0,decode_responses=True)
ts=r.ts()


def exp_weighted_avg(last_n):
    n = len(last_n)
    weights = {
        1: [1],
        2: [0.4, 0.6],
        3: [0.2222222222222, 0.33333333333333, 0.4444444444444],
        4: [0.14285714285714285, 0.2142857142857143, 0.2857142857142857, 0.3571428571428571],
        5: [0.1, 0.15, 0.2, 0.25, 0.3]
    }
    weighted_avg = np.average(last_n, axis=0, weights=weights[n])
    return weighted_avg



@sio_server.event
async def connect(sid, environ):
    print("connect ",sid)
    # rts.add(sid, 1,1.1)
    # x= rts.get(sid)
    # print(x)
    await sio_server.emit('my_response', {'data': 'Connected', 'count': 0}, room=sid)

@sio_server.event
async def server_side_fetch(sid, data):
    print('message server side', data)

    await sio_server.emit('my_response', {'data': data['data'], 'count': 0}, room=sid)


@sio_server.event
async def submit_interections(sid, data):
    userid = data['userid']
    itemid = data['itemid']

    # i will be not adding new items which is not present in item embeddings
    if redis_guard_for_new_items(itemid):
        ts.add(userid,"*",itemid,retention_msecs=day_in_ms)
        records=[int(y) for x,y in ts.range(userid, "-", "+")[-5:]]
        print(records)
    else:
        pass

    print('message server side', data)
    await sio_server.emit('my_response', {'data': data, 'count': 0}, room=sid)

@sio_server.event
async def demand_predictions(sid, data):
    #fixme
    start = timer()

    userid = data['userid']
    
    #fetch last 5 items
    records=[]
    try:
        records=[int(y) for x,y in ts.range(userid, "-", "+")[-5:]]

        indices = fetch_knn([calculate_vector(convert_itemlist_to_vector_list(records))])
    
        indices= list(set(list(indices[0])) - set(records[-5:]))
        # print(len(indices),indices)
        predictions = predict_knn(indices, userid)
        # print(len(predictions),predictions)
        sorted_predictions = [int(y) for x,y in sorted(zip(predictions, indices),reverse=True)]
        # print(sorted(zip(predictions, indices),reverse=True))
        print(sorted_predictions)
    except Exception as e:
        print(e)
    else:
        sorted_predictions = predict_all_items_sorted(userid)
        await sio_server.emit('my_response', {'data': sorted_predictions, 'count': 0}, room=sid)

    """#todo else lagana hai
    if records == []:
        sorted_predictions = predict_all_items_sorted(userid)
        await sio_server.emit('my_response', {'data': sorted_predictions, 'count': 0}, room=sid)
    print(records)

    indices = fetch_knn([calculate_vector(convert_itemlist_to_vector_list(records))])
    
    indices= list(set(list(indices[0])) - set(records[-5:]))
    # print(len(indices),indices)
    predictions = predict_knn(indices, userid)
    # print(len(predictions),predictions)
    sorted_predictions = [int(y) for x,y in sorted(zip(predictions, indices),reverse=True)]
    # print(sorted(zip(predictions, indices),reverse=True))
    print(sorted_predictions)
"""

    """#todo deleteme
    temp_indices=indices.copy()
    temp_indices.sort()
    temp_predictions=predict_knn(temp_indices, userid)
    print_audio(indices=temp_indices,predictions=temp_predictions)
    #fixme
    end = timer()
    print("-----------------------", end - start ,"-----------------------")"""
    await sio_server.emit('my_response', {'data': sorted_predictions}, room=sid)

@sio_server.event
async def disconnect(sid):
    print('disconnect ', sid)





