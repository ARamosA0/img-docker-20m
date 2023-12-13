from flask import Flask, render_template, request, make_response, g, jsonify
from redis import Redis
import os
import socket
import random
import json
import logging
import dask.dataframe as dd
import pandas as pd
import math
import time

hostname = socket.gethostname()

app = Flask(__name__)
app.config['mi_dataframe'] = None

def get_redis_broker():
    if not hasattr(g, 'redis'):
        #cambiar el puerto por 6380 para comectarce al otro redis
        g.redis = Redis(host="redis-collect", port=6379, db=0, socket_timeout=5)
    return g.redis

###

# Init K-vector with correct value based on distance type
def initVectDist(funName, N):
    if funName == 'euclidiana' or funName == 'manhattan' or funName == 'euclidianaL' or funName == 'manhattanL':
        ls = [99999] * N
    else:
        ls = [-1] * N

    lu = [None] * N
    return ls, lu


# Keep the closest values, avoiding sort
def keepClosest(funname, lstdist, lstuser, newdist, newuser, N):
    if funname == 'euclidiana' or funname == 'manhattan' or funname == 'euclidianaL' or funname == 'manhattanL':
        count = -1
        for i in lstdist:
            count += 1
            if newdist > i:
                continue
            lstdist.insert(count, newdist)
            lstuser.insert(count, newuser)
            break
    else:
        count = -1
        for i in lstdist:
            count += 1
            if newdist < i:
                continue
            lstdist.insert(count, newdist)
            lstuser.insert(count, newuser)
            break

    if len(lstdist) > N:
        lstdist.pop()
        lstuser.pop()
    return lstdist, lstuser


def readLargeFileDask(filename, delim=','):
    # Utiliza assume_missing=True para manejar valores no especificados en la conversión a int64
    ddf = dd.read_csv(filename, delimiter=delim, header=None, assume_missing=True, low_memory=False)
    print(ddf.dtypes)
    print(ddf)
    #df = ddf.to_numeric(ddf)
    #print(df.dtypes)
    print("inicia grouped")
    #ddf[2] = ddf[2].astype("int16")
    ddf_grouped = ddf.groupby(0).apply(lambda group: dict(zip(group[1], group[2])), meta=('x', 'f8'))
    result = ddf_grouped.compute().to_dict()

    return result

# Euclidian distance
def euclidianaL(user1, user2):
    dist = 0.0
    count = 0
    print(user1)
    print(user2)
    for i in user2:
        if not (user1.get(i) is None):
            x = user1.get(i)
            y = user2.get(i)
            dist += math.pow(x - y, 2)
            count += 1

    if count == 0:
        return 9999.99
    return math.sqrt(dist)

# K-Nearest neighbour
def knn_L(N, distancia, usuario, data):  # N numero de vecinos
    funName = distancia.__name__
    print('k-nn', funName)

    listDist, listName = initVectDist(funName, N)
    nsize = len(data)
    otherusers = range(0, nsize)
    vectoruser = data.get(usuario)

    for i in range(0, nsize):
        tmpuser = i
        if tmpuser != usuario:
            tmpvector = data.get(tmpuser)
            if not (tmpvector is None):
              tmpdist = distancia(vectoruser, tmpvector)
              if tmpdist is not math.nan:
                listDist, listName = keepClosest(funName, listDist, listName, tmpdist, tmpuser, N)

    return listDist, listName

def recommendationL(usuario, distancia, N, items, minr, data):
    ldistK, luserK = knn_L(N, distancia, usuario, data)

    user = data.get(usuario)
    recom = [None] * N
    for i in range(0, N):
        recom[i] = data.get(luserK[i])
    # print('user preference:', user)

    lstRecomm = [-1] * items
    lstUser = [None] * items
    lstObj = [None] * items
    k = 0

    fullObjs = {}
    count = 0
    for i in recom:
        for j in i:
          tmp = fullObjs.get(j)
          if tmp is None:
            fullObjs[j] = [i.get(j), luserK[count]]
          else:
            nval = i.get(j)
            if nval > tmp[0]:
              fullObjs[j] = [nval, luserK[count]]
        count += 1

    finallst = topSuggestions(fullObjs, count, items)
    return finallst

def topSuggestions(fullObj, k, items):
  rp = [-1]*items

  for i in fullObj:
    rating = fullObj.get(i)

    for j in range(0, items):
      if rp[j] == -1 :
        tmp = [i, rating[0], rating[1]]
        rp.insert(j, tmp)
        rp.pop()
        break
      else:
        tval = rp[j]
        if tval[1] < rating[0]:
          tmp = [i, rating[0], rating[1]]
          rp.insert(j, tmp)
          rp.pop()
          break

  return rp



@app.route("/", methods=['POST','GET'])
def hello():
    return make_response(jsonify(
    {
    'message':'Datos Cargados',
    }
    ))

@app.route("/data", methods=['POST','GET'])
def getdata():
    inicio = time.time()
    data = readLargeFileDask('ratings.csv')
    app.config['mi_dataframe'] = data
    #g.dfg = data
    fin = time.time()
    return make_response(jsonify(
    {
    'message':'Datos Cargados',
    'tiempo':fin-inicio,
    'len':len(data)
    }
    ))

@app.route("/<int:user_id>", methods=['GET'])
def getruta(user_id):
    #redis = get_redis_broker()
    #valor = redis.get('getUsuarioCursos')
    
    print("///////////////////////////////////////////////////////////////////////")
    #lstdb20 = readLargeFileDask('ratings.csv')
    #ldist, luser = knn_L(15, euclidianaL, id, lstdb20)
    #data = g.lstdb
    df = app.config.get('mi_dataframe')

    if df is not None:
        id = int(user_id)
        rfunc = euclidianaL
        inicio = time.time()
        
        lista = recommendationL(id, euclidianaL, 10, 20, 3.0, df)
        fin = time.time()
        return make_response(jsonify(
        {
        'message':'API V2',
        'tiempo':fin-inicio,
        'resultado':lista
        }
        ))

    else:
        return "La variable global 'mi_dataframe' no está definida."
     




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)
