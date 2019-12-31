from pythainlp.tag import pos_tag, pos_tag_sents
from pythainlp.tag.named_entity import ThaiNameTagger 
import deepcut

text = "วิเชษฐ์ เกษมทองศรี นายวิเชษฐ์ เกษมทองศรี อดีตรัฐมนตรีว่าการกระทรวงทรัพยากรธรรมชาติและสิ่งแวดล้อม ในรัฐบาลนางสาวยิ่งลักษณ์ ชินวัตร อดีตรัฐมนตรีช่วยว่าการกระทรวงคมนาคม ในรัฐบาลของพันตำรวจโท ทักษิณ ชินวัตร และสมาชิกสภาผู้แทนราษฎรจังหวัดราชบุรี สังกัดพรรคไทยรักไทยประวัติ ประวัติ. นายวิเชษฐ์ เกษมทองศรี เกิดเมื่อวันที่ 7 กรกฎาคม พ.ศ. 2505 ที่อำเภอเมืองราชบุรี จังหวัดราชบุรี เป็นบุตรของนายชัยวัฒน์ กับนางสุภาพร เกษมทองศรี สำเร็จการศึกษาระดับประกาศนยีบัตรวิชาชีพ (ปวช.)"

#print("sent_tokenize:", sent_tokenize(text))
#list_w  = deepcut.tokenize(text)
#sents = ("word_tokenize 1 :",list_w)
#print("word_tokenize 2 :", word_tokenize(text))
#print("word_tokenize 2, without whitespace:", word_tokenize(text, keep_whitespace=False))
print(pos_tag(deepcut.tokenize(text)))

#เขียนคำสั่ง clean text ลงไฟล์ใหม่โดยไฟล์นี้จะเอาไ้ใช้เป็น training set
#if(pos_tag_type == "PUNC" ---> delete it
# else write on new file (cln_troken_data)

#ner = ThaiNameTagger()
#ent = ner.get_ner(text)
#print(ent)