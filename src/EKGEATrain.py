from preprocess.BertDataLoader import BertDataLoader
from config.KBConfig import *
from preprocess.KBStore import KBStore
from tools.Announce import Announce
from train.PairwiseTrainer import PairwiseTrainer
import torch as t
from train.RelationTrainer import RelationTrainer

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # device = t.device("cuda" if t.cuda.is_available() else "cpu")
    device = t.device('cuda')# 将张量或模型放到cuda上进行计算
    print(device)
    tids1 = BertDataLoader.load_saved_data(dataset1)  # 加载seq分词后的值，列表类型，列表元素为元组，元组有两个元素，第一个是id，表示哪个实体，根据id可以在entities_tab文件中可以确定，元组第二个元素是列表，表示该实体及其属性分词后每个词对应的id
    tids2 = BertDataLoader.load_saved_data(dataset2)  # dataset2 -- attr_triples_2  , rel_triple_2

    max_len1 = max([len(tokens) for eid, tokens in tids1]) # 获取token最大的长度
    max_len2 = max([len(tokens) for eid, tokens in tids2])
    print(Announce.printMessage(), 'Max len 1:', max_len1)
    print(Announce.printMessage(), 'Max len 2:', max_len2)
    eid2tids1 = {eid: tids for eid, tids in tids1} # 将每个token转换成字典类型
    eid2tids2 = {eid: tids for eid, tids in tids2}
    fs1 = KBStore(dataset1)  # 初始化两个对象，用来操作中文或英文的数据
    fs2 = KBStore(dataset2)
    # fs1.load_entities()
    # fs2.load_entities()
    fs1.load_kb_from_saved()  # 从保存的文件中取出处理好的数据
    fs2.load_kb_from_saved()
    if args.basic_bert_path is None:
        trainer = PairwiseTrainer() # 初始化批次大小，批次数量，最近邻样本数等参数
        trainer.data_prepare(eid2tids1, eid2tids2, fs1, fs2) # 获取训练集，验证集和测试集的数据加载器
        trainer.train(device=device)
# 先训练属性，在训练关系
    rel_trainer = RelationTrainer()
    rel_trainer.data_prepare(eid2tids1, eid2tids2, fs1, fs2)
    if args.basic_bert_path is None:
        bert_model_path = links.model_save
    else:
        bert_model_path = args.basic_bert_path
    rel_trainer.train(bert_model_path, device=device)
