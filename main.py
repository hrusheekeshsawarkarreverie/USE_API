from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from model import IntentClassifierAPI
from typing import List


obj=IntentClassifierAPI()


app=FastAPI()

class Sentences(BaseModel):
    sentence_list : List[str]

@app.post('/')
async def function(item: Sentences):
    sentence_example=item.sentence_list
    # obj=IntentClassifierAPI()
    encoded_list=obj.run_classifier(sentence_example)
    print(type(encoded_list))
    print(encoded_list.shape)
    print(type(encoded_list.tolist()))
    return {"Embeddings":encoded_list.tolist()}