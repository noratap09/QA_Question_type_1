from pythainlp.tag import pos_tag, pos_tag_sents
from pythainlp.tag.named_entity import ThaiNameTagger

ner = ThaiNameTagger()
#Load POS
"""
pos_file = open("POS_ALL.txt",'r',encoding = 'utf-8-sig')
txt = pos_file.read()
pos_all = txt.split(",")

pos_len = len(pos_all)
"""

#Load Unit
pos_file = open("all_num_unit.txt",'r',encoding = 'utf-8-sig')
txt = pos_file.read()
num_unit_all = txt.split("|")

print("," in num_unit_all)

#import json

#my_file = open("result_to_post process.json",'r',encoding = 'utf-8-sig')
#txt = my_file.read()

#json_obj = json.loads(txt)

#json_obj[*question_id*]
#print(json_obj["14002"])

#json_obj[*question_id*]["frame_data"][*index ของ frame นั้นๆ*]
#print(json_obj["14005"]["frame_data"][0])

# [1, '<doc id="', '0.00021539831', '0.0', ['PUNCT', 'O'], 'PUNC'] อันนี้คือ 1 คำ
# format คือ แบบนี้
# [ตำแหน่งเริ่มของคำ , คำ , ค่าที่ได้จากmodel , ground truth , Name entity , Part of speech]

def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2) + 1)

def union(min1, max1, min2, max2):
    return (max1-min1 + 1)+(max2-min2 + 1 )-overlap(min1,max1,min2,max2)

def iou(x1,y1,x2,y2):
    i = overlap(x1,y1,x2,y2)
    u = union(x1,y1,x2,y2)
    return(i/u)

#print(overlap(1,5,6,10))
#print(union(1,5,6,10))
#print(iou(1,5,6,10))
