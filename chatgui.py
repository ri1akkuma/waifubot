import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from PIL import ImageTk, Image as PILI

# from keras.models import load_model (db)
from tensorflow.python.keras.models import load_model 
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents_generic.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#Creating GUI with tkinter
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "xXNicoNii<3Xx: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 
#create main window
base = Tk()
base.title("waifubot!")
base.geometry("1000x500")
base.resizable(width=TRUE, height=TRUE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", wrap=WORD, font="Arial",
padx=2, pady=2)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create PhotoImage object
nico_is_here = r"images\whiteday-nico.png"
og_nico = PILI.open(nico_is_here)
smol_nico = og_nico.resize((410,590), PILI.ANTIALIAS)
nico_image = ImageTk.PhotoImage(smol_nico)

# nico_image = PhotoImage(file=r"C:\Users\Generic\Documents\chatbot\images\whiteday-nico.png")
send_image = PhotoImage(file=r"images\airplane-send-3.png")
send_button_image = send_image.subsample(7)

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), width=60, height=30,
                    bd=0, bg="#ffd1dc", activebackground="#e18aaa",fg='#ffffff',
                    command= send, image=send_button_image, cursor="heart", 
                    justify=CENTER, compound=RIGHT, relief=RAISED)

# SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width=12, height=5,
#                     bd=0, bg="#ffd1dc", activebackground="#e18aaa",fg='#ffffff',
#                     command= send, cursor="heart", justify=CENTER, padx=-5)
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="100", height="5", font="Arial", padx=4, pady=5)
#EntryBox.bind("<Return>", send)

# Create a temporary waifu frame
waifu = Label(base, text="dis waifu<3", image=nico_image, width=20, height=100)

#Place all components on the screen
scrollbar.place(relx=0.8,rely=0.06, relheight=0.6)
ChatLog.place(relx=0.01, rely=0.06, relheight=0.6, relwidth=0.75)
EntryBox.place(relx=0.01, rely=0.7, relheight=0.23, relwidth=0.55)
SendButton.place(relx=0.57, rely=0.725, relheight=0.18, relwidth=.15)
waifu.place(relx=0.8, rely=0.03, relheight=0.95, relwidth=0.25)

#most recent
# scrollbar.place(x=800,y=6, height=386)
# ChatLog.place(x=6,y=6, height=350, width=800)
# EntryBox.place(x=128, y=401, height=90, width=400)
# SendButton.place(x=18, y=401, height=90)

# scrollbar.place(x=376,y=6, height=386)
# ChatLog.place(x=6,y=6, height=400, width=650)
# EntryBox.place(x=128, y=401, height=90, width=265)
# SendButton.place(x=6, y=401, height=90)

base.mainloop()
