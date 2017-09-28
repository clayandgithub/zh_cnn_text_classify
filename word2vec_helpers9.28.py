def padding_sentences(input_sentences, padding_token, padding_sentence_length = None):
#    sentences = [sentence.split(' ') for sentence in input_sentences]
    padding_sentences=[]
    #input_sentences为每一句话构成的列表而构成的列表
    if padding_sentence_length is not None:
       max_sentence_length = padding_sentence_length
    else:
       max_sentence_length = max([len(sentence) for sentence in input_sentences]) 
#    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in input_sentences])
    for sentence in input_sentences:
        #如果句子长度大于给定的长度最大值，则取前max_sentence_length个字符
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
        #如果句子长度小于给定的长度最大值，则用给定的字符将原句填充到最大长度
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
        padding_sentences.append(sentence)
    return padding_sentences, max_sentence_length
