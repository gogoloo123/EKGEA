# text -- 要转换的文本
# filters -- 过滤的字符集合，包括标点符号和空白字符串等非单词字符
# lower  -- 是否转换为小写
# split  -- 单词之间的分隔符


def text_to_word_sequence(text,
                          filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if text is None:
        return []
    if lower: text = text.lower()  # if lower为真，则执行冒号后的代码，text.lower() -- 转换为小写形式
    translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))} # ord -- 用来返回参数的ASCII数值  zip -- 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    text = text.translate(translate_table) # 按照指定的映射表进行替换，这里是将哪些非单词字符全部替换为空格
    seq = text.split(split)
    return [i for i in seq if i]
