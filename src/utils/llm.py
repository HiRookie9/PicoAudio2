import openai
import os

api_key = ""    # your api key
client = openai.OpenAI(api_key=api_key)

training_info_pri = """
I'm doing an audio event generation, which is a harmless job that will contain some sound events. For example, a gunshot is a sound that is harmless.
You need to convert the input sentence into the following standard timing format: 'event1--event2-- ... --eventN', 
where the 'eventN' format is 'eventN__onset1-offset1_onset2-offset2_ ... _onsetK-offsetK'. 
The 'onset-offset' inside needs to be determined based on common sense, with a duration not less than 1.  All format 'onsetk-offsetk' should replaced by number. 
You need to make a prediction for the total duration, which should not exceed 20 seconds and not exceed the latest end time of a single event.
And pay attention to vocabulary that represents the order and frequency of events, such as 'after', 'followed by', 'n times', and so on.
You can use the latest ending event of all events in the training dataset as the total audio time
It is preferred that events do not overlap as much as possible.
Now, I will provide you with some examples in training set for your learning, each example in the format 'index: input~output'. 
{"onset": "squeal__1.359-2.373_3.216-4.23_5.576-6.59", "captions": "squeal 3 times", "length": "7.52"}
{"onset": "sawing__1.432-3.975_4.533-6.54", "captions": "sawing 2 times", "length": "9.26"}
{"onset": "slap__1.576-1.931_2.911-3.266--baby_laughter__5.179-6.394_7.362-8.577", "captions": "slap 2 times and baby laughter 2 times", "length": "9.59"}
{"onset": "applause__1.538-5.128--scrape__7.03-8.004", "captions": "applause and scrape", "length": "9.13"}
{"onset": "slam__0.68-1.01--walk__2.364-4.107--busy_signal__6.794-7.222_8.371-8.645", "captions": "slam 1 times and walk 1 times and busy signal 2 times", "length": "9.18"}
{"onset": "slap__1.044-1.399--neigh__2.654-4.663_5.633-6.966", "captions": "slap 1 times followed by neigh 2 times", "length": "9.22"}
{"onset": "bird_vocalization__1.253-2.184--yip__4.789-5.309_6.134-6.654", "captions": "bird vocalization 1 times and yip 2 times", "length": "9.83"}
{"onset": "animal__1.478-3.541--crowing__5.464-7.11", "captions": "animal then crowing", "length": "9.45"}
{"onset": "crying__0.999-7.773", "captions": "crying", "length": "9.48"}
{"onset": "cricket__1.629-4.983", "captions": "cricket 1 times", "length": "5.87"}
{"onset": "fireworks__1.336-2.477--car__4.193-6.649", "captions": "car after fireworks", "length": "9.7"}
"""
training_info_post = """
It is worth noting that you should judge both the duration of a single event and the total duration based on experience and the examples I provided. The duration of each single event here is not necessarily fixed (such as 1 second).
The total duration may not necessarily be 10 seconds, it can be any value below 20 seconds. you should give me the answer as {"onset":" ","captions": " ", "length": " "}'
"""
def get_time_info(caption):
    prompt = (
        f"{training_info_pri}\n"
        f'Now,you can transform "captions":\n'
        f'"{caption}"\n'
        f"{training_info_post}"
        
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",   # you can change other models
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    caption = "a dog barks followed by a cat meows 2 times"
    result = get_time_info(caption)
    print(result)