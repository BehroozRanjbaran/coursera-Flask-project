from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions

def emotion_predictor(text):
    nlu = NaturalLanguageUnderstandingV1(...)
    response = nlu.analyze(text=text, features=Features(emotion=EmotionOptions())).get_result()
    return response["emotion"]["document"]["emotion"]
