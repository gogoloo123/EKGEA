import os
import random
import shutil
from preprocess import Parser
from tools import FileTools

old_version_path = os.path.abspath('../data/ECHI')
new_version_path = os.path.abspath('../data')

os.chdir(old_version_path)
datasets = os.listdir('.')
print(datasets)
# 列表中存放元组，元组中的数据是属性三元组
def load_attr(src, dst):
    # tups -- 经过处理后的数据集，是个列表
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf) # *tup  -- 拆包，将tup中的元素以\t为分隔的形式打印到wf指定的文件中，将zh_att_triples文件数据放进attr_triples_1中


for dataset in datasets:
    if dataset ==  '.ipynb_checkpoints':
        continue
    os.chdir('/'.join((old_version_path, dataset)))
    # ds1:zh  ds2:en
    ds1, ds2 = dataset.split('_')
    # print(os.getcwd())
    # 复制数据集文件
    dataset_new_path = '/'.join((new_version_path, dataset))
    if not os.path.exists(dataset_new_path):
        os.mkdir(dataset_new_path)
    load_attr('_'.join((ds1, 'att_triples')), '/'.join((dataset_new_path, 'attr_triples_1')))# 处理属性三元组
    load_attr('_'.join((ds2, 'att_triples')), '/'.join((dataset_new_path, 'attr_triples_2')))
    # shutil.copy('attr_triples_1', dataset_new_path)
    # shutil.copy('attr_triples_2', dataset_new_path)
    shutil.copy('ent_ILLs', '/'.join((dataset_new_path, 'ent_links')))# shutil.copy(src,dst) -- 将src目录的文件复制到dst目录中,不是复制快捷方式，会复制内容
    shutil.copy('_'.join((ds1, 'rel_triples')), dataset_new_path + '/rel_triples_1') # 处理关系三元组
    shutil.copy('_'.join((ds2, 'rel_triples')), dataset_new_path + '/rel_triples_2')
    # 结束
    # folds = os.listdir('mapping')
    # print(folds)
    ent_links = FileTools.load_list('/'.join((dataset_new_path, 'ent_links'))) # 加载种子实体对列表，列表里存放的是子列表，子列表存放的是种子实体对
    random.seed(11037)
    random.shuffle(ent_links) # 打乱列表中元素的顺序，直接修改原始列表
    ent_len = len(ent_links)
    train_len = ent_len * 2 // 10
    valid_len = ent_len * 1 // 10 # 整除
    train_links = ent_links[:train_len] # 2:1:7
    valid_links = ent_links[train_len: train_len + valid_len]
    test_links = ent_links[train_len + valid_len:] # 种子实体对分割训练集，验证集，测试集
    new_fold_path = '/'.join((new_version_path, dataset, '721_5fold', '0'))
    if not os.path.exists(new_fold_path):
        os.makedirs(new_fold_path)
    os.chdir(new_fold_path)

    FileTools.save_list(train_links, '/'.join((new_fold_path, 'train_links'))) # 保存训练集，验证集，测试集的数据到文件中
    FileTools.save_list(valid_links, '/'.join((new_fold_path, 'valid_links')))
    FileTools.save_list(test_links, '/'.join((new_fold_path, 'test_links')))
