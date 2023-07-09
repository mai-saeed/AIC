import re
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import json
from tqdm import tqdm

def infere_model(model_path,input_data,device):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path, src_lang="ar_AR", tgt_lang="ar_AR")
    model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
    arabic_stop = ['تلك', 'وتلك', 'بتلك', 'وهذا','وهذه','هذا','بهذا','هذه'
         ,'هؤلاء', 'كان','كانت','قد','وقد','فقد','مما','بقد'
         ,'لقد','وكان','وكانت','كانوا','وبذلك','ذلك','وذلك','مثلا'
         ,'ومثلا','لكن','ولكن','كما'
        ,'وكما','لذلك','ولذلك','فبذلك','كذلك','وكذلك','وحينئذ','حينئذ'
        'مثلما','غيرها','الحبيبة','ولكننا','لكننا','التي','الذي'
        ,'والتي','والذي','الحبيب','حيث','وحيث','وبالفعل','بالفعل','فعلا','وفعلا'
        'إذ','وإذ','إلخ','وإلخ','حينما','وحينما','بل','وبل','والخ','الخ','ايضا','وايضا','إذا','وإذا',
              'رابعا','خامسا','اولا','ثانيا','ثالثا','سادسا','أولا','سابعا','ثامنا']
    def clean_arabic_text(text):
        # Arabic Letter Normalisation
        text = " ".join([word for word in text.split() if word not in arabic_stop])
        text = re.sub(r'عزيزي الطالب / عزيزتي الطالبة','', text)
        text = re.sub(r'عزيزي الطالب','', text)
        text = re.sub(r'/ عزيزتي الطالبة','', text)
        text = re.sub(r'عزيزتي الطالبة', '',text)
        text = re.sub(r' لاحظ الخريطة التالية لتتعرف أهم هذه الفتوحات ', '',text)
        text = re.sub(r'لاحظ الشكل التالي لتتعرف أهم النتائج التي ترتبت على فتح بلاد المغرب', '', text)
        text = re.sub(r'لاحظ.*?\.', '', text)
        text = re.sub(r'خريطة \([^)]+\)', '', text)
        text = re.sub(r'[\u064B-\u065F]', '', text)
        # Replace Tatweel with a single character
        text = re.sub(r'[\u0640]+', '', text)
        text = text.replace(r".....", "")
        text = text.replace(r"....", "")
        text = text.replace(r". . . .", "")
        text = text.replace(r"...", '')
        text = text.replace(r". . .", '')
        text = text.replace(r' . . . ', '')
        text = text.replace(r'..', '')
        text = text.replace(r'. .', '')
        text = text.replace(r'…', " ")
        text = re.sub(r'\b\s+\b', ' ', text) # space
        text = re.sub(r'\.(?=\.)', ' ', text) #dots
        text = re.sub(r',(?=,)', ' ', text)
        text = re.sub(r'\.\s*\.', '', text)
        text = re.sub(r'[^.،؛/\w\s]', " ", text)
        text = WHITESPACE_HANDLER(text)
        
        return text
    results = {}
    for id in tqdm(input_data.keys(),desc="inferring model"):
        paragraph_length = len(input_data[id].split())
        min_L = int(paragraph_length * (58/100))
        max_L = int(paragraph_length * (68/100))

        input_ids = tokenizer.encode(clean_arabic_text(input_data[id]), return_tensors='pt').to(device)


        output = model.generate(input_ids,no_repeat_ngram_size=8,min_length = min_L, max_length = max_L,  early_stopping=True)
        summary = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
        results[id]=summary
        
    return results