import deepcut
import json

my_file = open("data_set_fix.json",'r',encoding = 'utf-8-sig')
txt = my_file.read()

json_obj = json.loads(txt)

from pythainlp.tag import pos_tag, pos_tag_sents
from pythainlp.tag.named_entity import ThaiNameTagger

ner = ThaiNameTagger()

def ck_have_num(ans):
    ans_ne = ner.get_ner(ans)
    for i in ans_ne:
        if(i[1]=="NUM"): return True
    return False

def predict_have_num(question):
    question_troken = deepcut.tokenize(question)
    if(question.find("เท่าไร")!=-1 or question.find("เท่าไหร่")!=-1 or question.find("เท่าใด")!=-1):
        return True
    elif("กี่" in question_troken):
        return True
    else:
        return False

def predict_have_person_name(question):
    if(question.find("ใคร")!=-1):
        return True
    else:
        return False


count = 0
score = 0

all_num_unit = list()

for count_data , data in enumerate(json_obj['data'][0:15000],start=1):
    if(data['question_type']==1):
        ans = data['answer']
        question = data['question']
        question_id = data['question_id']
        ans_troken = deepcut.tokenize(ans)
        question_troken = deepcut.tokenize(question)
        #print("QUS : ",question_troken)
        #print("ANS : ",ans_troken)
        if(ck_have_num(ans_troken)): 
            count = count + 1
            #print("Question_ID : ",question_id)
            #print("QUS : ",question_troken)
            #print("ANS : ",ans_troken)
        """
        if(predict_have_person_name(question)):
            ans_ne = ner.get_ner(ans)
            print("Question_ID : ",question_id)
            print("QUS : ",question_troken)
            print("ANS : ",ans_ne)
        """
        if(predict_have_num(question)):
            score = score + 1
            ans_ne = ner.get_ner(ans_troken)
            print("Question_ID : ",question_id)
            print("QUS : ",question_troken)
            print("ANS : ",ans_ne)

            for i in ans_ne:
                if(i[1]!="NUM"): 
                    all_num_unit.append(i[0])
        #if(predict_have_num(question) == False and ck_have_num(ans_troken) == True):
            #print("Question_ID : ",question_id)
            #print("QUS : ",question_troken)
            #print("ANS : ",ans_troken)     


print("Total : ",score,"/",count)
all_num_unit = list(set(all_num_unit))
print(all_num_unit)
all_num_unit = "|".join(all_num_unit)

file_all_num_unit = open("all_num_unit.txt","w",encoding = 'utf-8') 
file_all_num_unit.write(all_num_unit)
file_all_num_unit.close()

        
