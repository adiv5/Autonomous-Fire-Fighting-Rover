from __future__ import unicode_literals
import spacy
import textacy
from nltk import Tree
import os

def tag_pos(s):
    return(s.tag_)

sent=[]
def split_s(sentence,s):
    global sent
    if len(sent)==0:
        #print(str(s))
        sent=sentence.split(str(s))
        #print(sent)
        sent=[i.strip() for i in sent]
        ''' for i in sent:
            i=i.strip()
            print(i)'''
        print(sent)
    else:
        print(str(s))
        for i in sent:
            i=i.split(str(s))
            i=[s.strip() for s in i]
            print(i) 



nlp = spacy.load('en_core_web_sm',parser=True, tagger=True,entity=False)

tags=[]
sentence="Go towards the table which is to the left of the chair and behind the cone then towards the stool"

doc=nlp(u"Go towards the table which is to the left of the chair and behind the cone, then towards the stool")
def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)

[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

for np in doc.noun_chunks:
    print(np.text)



for s in doc:

    if s.tag_ == "RB":
        split_s(sentence,s)
    tags.append((s,tag_pos(s)))
print (tags)

for i in sent:
    doc1=nlp(i)
    [to_nltk_tree(sent.root).pretty_print() for sent in doc1.sents]

