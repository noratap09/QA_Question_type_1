"""
import glob
from shutil import copyfile

from pythainlp.tag import pos_tag, pos_tag_sents
from pythainlp.tag.named_entity import ThaiNameTagger

ner = ThaiNameTagger()

path = glob.glob("../elasticsearch_api_python/document/pp/*.txt")
count = 0

result_person_name = "|"

for myfile in path:
    print(count)
    #if(count>5):
    #    break
    data = open(myfile,'r',encoding = 'utf-8-sig')
    txt = data.read()
    txt_ne = ner.get_ner(txt)
    #print(txt_ne)

    person_name = list()
    for i in txt_ne:
        if((i[2] == 'B-PERSON') or (i[2] == 'I-PERSON')):
            person_name.append(i)
    
    old_value = ""
    for i in person_name:
        if(i[2] == 'B-PERSON' and old_value == 'I-PERSON'):
            result_person_name = result_person_name+"|"+i[0]
        elif((i[2] == 'B-PERSON' and old_value == 'B-PERSON')):
            result_person_name = result_person_name+"|"+i[0]
        else:
            result_person_name = result_person_name+i[0]
        old_value = i[2]

    result_person_name = result_person_name+"|"
    count = count+1

person_name_list = result_person_name.split("|")
person_name_list = list(set(person_name_list))
result_person_name = "|".join(person_name_list)

person_name_file = open("all_person_name.txt","w",encoding = 'utf-8') 
person_name_file.write(result_person_name)
person_name_file.close()


#Load person name
person_name_file = open("all_person_name.txt",'r',encoding = 'utf-8-sig')
person_name_txt = person_name_file.read()
person_name_list = person_name_txt.split("|")
person_name_list = list(set(person_name_list))

result_person_name = "|".join(person_name_list)
person_name_file = open("all_person_name.txt","w",encoding = 'utf-8') 
person_name_file.write(result_person_name)
person_name_file.close()
"""

import deepcut

#Load person name
person_name_file = open("all_person_name.txt",'r',encoding = 'utf-8-sig')
person_name_txt = person_name_file.read()
person_name_list = person_name_txt.split("|")
person_name_list = list(set(person_name_list))
#print(person_name_list[0:10])

all_person_name_troken = list()
for i_index,i in enumerate(person_name_list,start=0):
    print(i_index)
    person_name_troken = deepcut.tokenize(i)
    all_person_name_troken = all_person_name_troken + person_name_troken
#print(all_person_name_troken)
all_person_name_troken = list(set(all_person_name_troken))

#Write troken file
result_person_name = "|".join(all_person_name_troken)
person_name_file = open("all_person_name_troken.txt","w",encoding = 'utf-8') 
person_name_file.write(result_person_name)
person_name_file.close()