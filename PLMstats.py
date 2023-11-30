from transformers import AutoModel,AutoTokenizer,AutoConfig
from transformers import T5Tokenizer,T5Model
import pandas as pd

def getInfo(C,T,M,df_1,df_2):
    # info about model through config
    df_1 = df_1._append({
        'Model_name':C.model_type,
        'Layer_count':C.num_hidden_layers,
        'Hidden_size':C.hidden_size,
        'AttnHeads_count':C.num_attention_heads,
        'Intermediate_size':C.intermediate_size
    },ignore_index=True)
    
    # info about the tokenizer
    df_2 = df_2._append({
        'Tokeniser_name':C.model_type,
        'Vocab_size':T.vocab_size,
        'Token_capacity':T.model_max_length,
        'Special_tkns':T.special_tokens_map,
    },ignore_index=True)
    
    return df_1,df_2    

def loader(name):
    if ('t5' in name):
        cnfg    = AutoConfig.from_pretrained(name)
        tknizer = T5Tokenizer.from_pretrained(name)
        mdl     = T5Model.from_pretrained(name) 
    else:
        cnfg    = AutoConfig.from_pretrained(name)
        tknizer = AutoTokenizer.from_pretrained(name)
        mdl     = AutoModel.from_pretrained(name) 
    
    return cnfg,tknizer,mdl

models = ['google/electra-base-discriminator',
          'thenlper/gte-large',
          'google/t5-v1_1-xl',
          'microsoft/deberta-v3-large']

Models_df = pd.DataFrame(columns=['Model_name','Layer_count','Hidden_size',
                                'AttnHeads_count','Intermediate_size'])
Tokeniser_df = pd.DataFrame(columns=['Tokeniser_name','Vocab_size',
                                     'Token_capacity','Special_tkns',])

for name in models:
    C,T,M = loader(name)
    Models_df,Tokeniser_df = getInfo(C,T,M,Models_df,Tokeniser_df)
    
print("task completed")
print(Models_df)
print('\n')
print(Tokeniser_df)