'''
character to vec preprocessor
'''

import numpy as np
import sys

#open text
with open('hhgttg.txt','r') as f:
    text = f.read()

#remove linebreaks
text = text.replace("\n"," ")

#char to int mappings    
dic = {
    'a':0,
    'b':1,
    'c':2,
    'd':3,
    'e':4,
    'f':5,
    'g':6,
    'h':7,
    'i':8,
    'j':9,
    'k':10,
    'l':11,
    'm':12,
    'n':13,
    'o':14,
    'p':15,
    'q':16,
    'r':17,
    's':18,
    't':19,
    'u':20,
    'v':21,
    'w':22,
    'x':23,
    'y':24,
    'z':25,
    'A':26,
    'B':27,
    'C':28,
    'D':29,
    'E':30,
    'F':31,
    'G':32,
    'H':33,
    'I':34,
    'J':35,
    'K':36,
    'L':37,
    'M':38,
    'N':39,
    'O':40,
    'P':41,
    'Q':42,
    'R':43,
    'S':44,
    'T':45,
    'U':46,
    'V':47,
    'W':48,
    'X':49,
    'Y':50,
    'Z':51,
    '1':52,
    '2':53,
    '3':54,
    '4':55,
    '5':56,
    '6':57,
    '7':58,
    '8':59,
    '9':60,
    '0':61,
    '-':62,
    '.':63,
    ',':64,
    '!':65,
    '?':66,
    '(':67,
    ')':68,
    '\'':69,
    '"':70,
    ' ':71,
    }

#empty containers
vec = np.empty((0,72))
prevchar = None
vecs = []

#convert text to numpy array of character indices
for i, char in enumerate(text, start=1):
    sys.stdout.write("Progress: %i of %i  \r" % (i,len(text)))
    sys.stdout.flush()
    
    #save then reset numpy array every 10k characters for speed
    if i % 10000 == 0:
        vecs.append(np.copy(vec))
        vec = np.empty((0,72))
    
    #convert current char to int, append to numpy array
    try:
        #ignore duplicate spaces
        if prevchar == " " and char == " ":
            prevchar = char
            continue
        new = np.zeros((1,72))
        new[0,dic[char]] = 1
        vec = np.vstack((vec,new))
        prevchar = char
    except:
        prevchar = char

#concat all saved numpy arrays
vec = np.empty((0,72))
for i, v in enumerate(vecs, start=1):
    i += 1
    sys.stdout.write("Concatenating text: %i of %i  \r" % (i,len(vecs)))
    sys.stdout.flush()
    vec = np.vstack((vec,v))

#save to disk
np.save('hhgttg.npy',vec)