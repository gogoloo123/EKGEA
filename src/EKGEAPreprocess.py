from preprocess.BertDataLoader import BertDataLoader
from preprocess.KBStore import KBStore
from config.KBConfig import * # 导入基本参数
# 预处理模块
if __name__ == '__main__':
    # 初始化对象，fs1用来处理中文属性/关系三元组，fs2用来处理英文属性/关系三元组
    fs1 = KBStore(dataset1)
    fs2 = KBStore(dataset2)
    # 保存属性三元组和关系三元组的值和id和对应关系到文件中
    fs1.load_kb()
    fs2.load_kb()

    KBStore.save_blocks(fs1, fs2)


    # 对产生的seq序列进行分词，并对分词向量和对应id及出现的次数进行保存
    dl1 = BertDataLoader(dataset1)
    dl2 = BertDataLoader(dataset2)
    dl1.run()
    dl2.run()
