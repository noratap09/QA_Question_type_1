from pythainlp.tag import pos_tag, pos_tag_sents
from pythainlp.tag.named_entity import ThaiNameTagger

ner = ThaiNameTagger()
#Load POS
pos_file = open("POS_ALL.txt",'r',encoding = 'utf-8-sig')
txt = pos_file.read()
pos_all = txt.split(",")

pos_len = len(pos_all)
print(pos_len)