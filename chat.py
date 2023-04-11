import openai
import pandas as pd

openai.api_key = "API_KEY"
PROMPT = '''Here are question answer pairs 

Q: where do you get a cashier's check from?
A:  A customer asks a bank for a cashier's check, and the bank debits the amount from the customer's account immediately, and assumes the responsibility for covering the cashier's check. That is in contrast with a personal check, in which the bank does not debit the amount from the customer's account until the check is deposited or cashed by the recipient. 

Q: when is a pilot on an ifr flight plan responsible for avoiding other aircraft
A:  When operation of an aircraft under VFR is not safe, because the visual cues outside the aircraft are obscured by weather or darkness, instrument flight rules must be used instead. IFR permits an aircraft to operate in instrument meteorological conditions (IMC), which is essentially any weather condition less than VMC but in which aircraft can still operate safely. Use of instrument flight rules is also required when flying in `` Class A '' airspace regardless of weather conditions. Class A airspace extends from 18,000 feet above mean sea level to flight level 600 (60,000 feet pressure altitude) above the contiguous 48 United States and overlying the waters within 12 miles thereof. Flight in Class A airspace requires pilots and aircraft to be instrument equipped and rated and to be operating under Instrument Flight Rules (IFR). In many countries commercial airliners and their pilots must operate under IFR as the majority of flights enter Class A airspace; however, aircraft operating as commercial airliners must operate under IFR even if the flight plan does not take the craft into Class A airspace, such as with smaller regional flights. Procedures and training are significantly more complex compared to VFR instruction, as a pilot must demonstrate competency in conducting an entire cross-country flight solely by reference to instruments. 

Q: who was charlie writing to in perks of being a wallflower movie?
A: The story begins with a quiet, sensitive 15-year-old boy named Charlie writing letters about his life to an unknown recipient. Charlie chooses that person because he said that he heard the person was nice and thought that this person would not be judgmental. He discusses his first year at high school, grappling with two traumatic experiences from his past: the suicide of his only middle-school friend, Michael Dobson, a year before, and the death of his favorite aunt, Helen, during his early childhood. 
'''



def get_answer(question):

    question = PROMPT + f"\nQ: {question}?\nA: "
    messages = {"role": "user", "content": question},
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message.content
    return reply

data_pd = pd.read_csv("annotated_pairs.csv")
data_pd["GPT Answer"] = data_pd["Questions"].apply(get_answer)

data_pd.to_csv("answer_pairs2.csv", index=False)


#go through each value in the question column pass it into get answer