from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from model import IntentClassifierAPI
from typing import List
app=FastAPI()

class Sentences(BaseModel):
    sentence_list : List[str]
    intent_list : List[str]
    num_intents : int
    input_sentence : str



@app.post('/')
async def function(item: Sentences):
    sentence_example=item.sentence_list
    intent_example=item.intent_list
    num_different_intents=item.num_intents
    input_string=item.input_sentence
    obj=IntentClassifierAPI()
    obj.run_classifier(sentence_example,intent_example,num_different_intents)
    output=obj.predict_intent(input_string)
    return {"prediction":output}
