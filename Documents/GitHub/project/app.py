#!/usr/bin/python
#-*-coding: utf-8 -*-
##from __future__ import absolute_import
###
from flask import Flask, jsonify, render_template, request
import json
import numpy as np

import numpy as np
import pandas as pd
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,TemplateSendMessage,ImageSendMessage, StickerSendMessage, AudioSendMessage
)
from linebot.models.template import *
from linebot import (
    LineBotApi, WebhookHandler
)

app = Flask(__name__)

lineaccesstoken ='lscPq43zcNBt9Sk8mOkJAGSGbBZBW63Kex9GlvR1Fz226i/1MUZ8QXJnZV/WE5VxPr2w+t7Dx1+INMKonuYP672ORhswC2GFlANQWCF0MPgHc1h8zXELl3DrsJ8si2lJjH4c07/S0PHLijoTBuYzXQdB04t89/1O/w1cDnyilFU='
line_bot_api = LineBotApi(lineaccesstoken)
dat = pd.read_csv('key.csv')
ansdat = pd.read_csv('ans.csv')
####################### new ########################
@app.route('/')
def splittraintest(dat,trainratio=0.7):
    sdat = dat.sample(frac=1,random_state=0)
    ntrain = int(len(dat)*trainratio)
    traindat = sdat.iloc[0:ntrain]
    testdat = sdat.iloc[ntrain:]
    return traindat,testdat
trdat,tedat = splittraintest(dat)
def get_wd_tokens(text):
    tokens = word_tokenize(text)
    return tokens
def get_cr_tokens(text):
    tokens = list(text)
    return tokens
def get_wo_cr_tokens(text):
    tokens = get_wd_tokens(text) + get_cr_tokens(text)
    return tokens
trkeyword = trdat['Keyword'].values
vectorizer = TfidfVectorizer(tokenizer=get_wo_cr_tokens, ngram_range=(1,3))
vectorizer.fit(trkeyword)
tekeyword = tedat['Keyword'].values
trfeat = vectorizer.transform(trkeyword)
tefeat = vectorizer.transform(tekeyword)
trlabel = trdat['Intent'].values
telabel = tedat['Intent'].values
trfeat_norm = normalize(trfeat)
tefeat_norm = normalize(tefeat)
model = LinearSVC(random_state=0)
model.fit(trfeat_norm, trlabel)
pred = model.predict(tefeat_norm)

def getintent(keyword):
    keyword = [keyword]
    feat = vectorizer.transform(keyword)
    feat_norm = normalize(feat)
    pred = model.predict(feat_norm)[0]
    return pred
def getresult(keyword):
    pred = getintent(keyword)
    result = ansdat[ansdat['Intent'] == pred]
    textresult = result
    textresult = textresult['Answer'].values+'\n'+textresult['Source'].values
    textresult = textresult[0]
    return textresult
def index():
    return "Hello World!"


@app.route('/webhook', methods=['POST'])
def callback():
    json_line = request.get_json(force=False,cache=False)
    json_line = json.dumps(json_line)
    decoded = json.loads(json_line)
    no_event = len(decoded['events'])
    for i in range(no_event):
        event = decoded['events'][i]
        event_handle(event)
    return '',200


def event_handle(event):
    print(event)
    try:
        userId = event['source']['userId']
    except:
        print('error cannot get userId')
        return ''

    try:
        rtoken = event['replyToken']
    except:
        print('error cannot get rtoken')
        return ''
    try:
        msgId = event["message"]["id"]
        msgType = event["message"]["type"]
    except:
        print('error cannot get msgID, and msgType')
        sk_id = np.random.randint(1,17)
        replyObj = StickerSendMessage(package_id=str(1),sticker_id=str(sk_id))
        line_bot_api.reply_message(rtoken, replyObj)
        return ''

    if msgType == "text":
        msg = str(event["message"]["text"])
        result = getresult(msg)
        replyObj = TextSendMessage(text= result)
        line_bot_api.reply_message(rtoken, replyObj)

    else:
        sk_id = np.random.randint(1,17)
        replyObj = StickerSendMessage(package_id=str(1),sticker_id=str(sk_id))
        line_bot_api.reply_message(rtoken, replyObj)
    return ''

if __name__ == '__main__':
    app.run(debug=True)
