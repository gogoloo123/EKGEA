import csv
import re
from collections import defaultdict
from typing import List, Iterator

from tqdm import tqdm

from config.KBConfig import *
from preprocess import Parser
from preprocess.Parser import OEAFileType
from tools import FileTools
from tools.Announce import Announce
from tools.MultiprocessingTool import MPTool
from tools.MyTimer import MyTimer
from tools.text_to_word_sequence import text_to_word_sequence


class KBStore:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.entities = []  # 列表里存放的是当前数据集文件中实体名称（无重复） entities_tab_.txt
        # self.classes = self.entities
        # self.literals = self.entities
        self.literals = []  # 存放的是当前数据集文件中的所有属性值（无重复） literals_tab_.txt
        self.entity_ids = {}# 字典  -- 键是实体名称（str），值是id（int）
        self.classes_ids = {}
        self.literal_ids = {}  # 字典 -- 键是属性值，值是int类型的数值

        self.relations = []   # 存放的是关系名称，相邻的两个是一对，分别表示正向和反向
        # self.properties = self.relations
        self.properties = []  # 存放数据集文件中的所有属性名称 properties_tab_.txt
        self.relation_ids = {}  # 字典  -- 关系名称对应id
        self.property_ids = {} # 字典 -- 键是属性名称，值是int类型的数值

        self.words = []
        self.word_ids = {}

        self.facts = {} # 关系三元组对应的字典 ， 键是头实体id，值是列表类型，列表中的每个元组由关系和尾实体构成，字典中相邻的两个元素是一对，即头实体:[(关系，尾实体)]，尾实体:[(新关系，头实体)]，
        self.literal_facts = {}# 属性三元组对应的字典，存放属性三元组的关系，键是实体id，值是列表，列表中的元组是由属性id和属性值id组成
        self.blocks = {}  # 字典类型， 键是属性值对应的id，值是有该属性值的实体id的集合
        self.word_level_blocks = {} # 字典类型，键是属性值，值是该属性值的实体id的集合

        self.properties_functionality = None # 每种属性的比例
        self.relations_functionality = None # 每种关系比例，或者说重要程度

    def load_kb(self) -> None:
        # 定义了一个时间类对象，用来计时
        timer = MyTimer()
        # load attr
        # 参数依次为：'../data/zh_en/attr_triples_1'，load函数，枚举类型，attr=0
        self.load_path(self.dataset.attr, self.load, OEAFileType.attr)
        if args.relation: # 处理关系三元组
            self.load_path(self.dataset.rel, self.load, OEAFileType.rel)  # 处理头实体，关系和尾实体的对应关系，以及对应的id
            self.relations_functionality = KBStore.calculate_func(self.relations, self.relation_ids, self.facts, self.entity_ids) # 参数为：关系名称，关系名称及对应的id，实体和关系中另两个值的对应关系，实体及对应id，计算关系占比
        # 增加对facts按relation排序
        for ent, facts in self.facts.items():
            facts: list
            facts.sort(key=lambda x: (x[0], x[1]), reverse=False)
        self.properties_functionality = KBStore.calculate_func(self.properties, self.property_ids, self.literal_facts,
                                                               self.entity_ids) # 计算属性的比例
        timer.stop()

        print(Announce.printMessage(), 'Finished loading in', timer.total_time())
        self.save_base_info()  # 保存实体和属性值
        self.save_datas()  # 保存属性名称和关系名称

    def save_base_info(self):
        print(Announce.doing(), 'Save Base Info')
        FileTools.save_dict_reverse(self.entity_ids, self.dataset.entities_out) # 保存实体到entities_out路径中

        FileTools.save_dict_reverse(self.literal_ids, self.dataset.literals_out) # 保存属性值到literals_out路径中
        print(Announce.done(), 'Finished Saving Base Info')

    def save_property_table(self):
        table_path = self.dataset.table_out
        print(Announce.doing(), 'Save', table_path)
        with open(table_path, 'w', encoding='utf-8') as fp:
            header = ['id', 'ent_name']
            if not functionality_control:
                header.extend(self.property_ids.keys())
            else:
                for p, pid in self.property_ids.items():
                    if self.properties_functionality[pid] > functionality_threshold:
                        header.append(p)
            writer = csv.DictWriter(fp, header)
            writer.writeheader()
            dicts = MPTool.packed_solver(self.get_property_table_line).send_packs(self.entity_ids.items()).receive_results()
            # dicts = [dic for dic in dicts if dicts is not None]
            dicts = filter(lambda dic: dic is not None, dicts)
            dicts = list(dicts)
            for dic in dicts:
                writer.writerow(dic) # 整合实体的属性，类别保存到文件中
        print(Announce.done())
        return dicts, header

    def save_seq_form(self, dicts: Iterator, header: List):
        def get_seq(dic: dict):
            eid = dic['id']
            values = [dic[key] for key in header if key in dic]
            seq = ' '.join(values)
            assert len(seq) > 0
            return eid, seq
        seq_path = self.dataset.seq_out
        print(Announce.doing(), 'Save', seq_path)
        header = header.copy()
        header.remove('id')
        seqs = MPTool.packed_solver(get_seq).send_packs(dicts).receive_results()
        # seqs = [get_seq(dic) for dic in dicts]
        FileTools.save_list(seqs, seq_path)
        print(Announce.done())

    def save_facts(self):
        print(Announce.doing(), 'Save facts')
        FileTools.save_dict_p(self.facts, self.dataset.facts_out)
        print(Announce.done(), 'Save facts')

    def save_datas(self):
        print(Announce.doing(), 'Save data2')
        print(Announce.printMessage(), 'Save', self.dataset.properties_out)
        with open(self.dataset.properties_out, 'w', encoding='utf-8') as writer:  # 保存属性名到properties_out路径中
            for r, id in self.property_ids.items():
                print(id, r, self.properties_functionality[id], sep='\t', file=writer)

        if args.relation:
            print(Announce.printMessage(), 'Save', self.dataset.relations_out)
            with open(self.dataset.relations_out, 'w', encoding='utf-8') as wfile: # 保存关系名到relation_out路径中 relations_python_.txt
                for r, id in self.relation_ids.items():
                    print((id, r, self.relations_functionality[id]), file=wfile)

        # 保存property csv
        dicts, header = self.save_property_table() # 保存每个实体的所有属性名和属性值到表格中 properties_csv_.csv
        self.save_seq_form(dicts, header) # 值与保存到表格中的一样，形式是保存到文件中 sequence_form_tab_.txt
        if args.relation:
            self.save_facts() # 保存关系三元组及其对应关系的数据 facts_python_.txt
        print(Announce.done(), 'Finished')

    def load_kb_from_saved(self):
        self.load_entities() # 加载实体列表，以及实体和id的对应关系的字典
        self.load_literals() # 加载属性值和id对应关系的字典
        self.load_relations() # 加载关系的列表，以及关系和对应id的字典
        self.load_properties() # 加载属性名和对应id的字典
        self.load_facts() # 加载关系三元组的关系，正反两个方向上的
        pass

    def load_entities(self):
        print(Announce.doing(), 'Load entities', self.dataset.entities_out)
        # self.entity_ids = FileTools.load_dict_reverse(self.dataset.entities_out)
        entity_list = FileTools.load_list(self.dataset.entities_out)
        self.entity_ids = {ent: int(s_eid) for s_eid, ent in entity_list}
        self.entities = [ent for s_eid, ent in entity_list]
        print(Announce.done())
        pass

    def load_relations(self):
        relation_list = FileTools.load_list_p(self.dataset.relations_out)
        self.relation_ids = {rel: rid for rid, rel, func in relation_list}
        self.relations = [rel for rid, rel, func in relation_list]

    def load_literals(self):
        entity_list = FileTools.load_list(self.dataset.literals_out)
        self.literal_ids = {ent: int(s_eid) for s_eid, ent in entity_list}

    def load_properties(self):
        property_list = FileTools.load_list(self.dataset.properties_out)
        self.property_ids = {prop: s_pid for s_pid, prop, s_func in property_list}

    def load_facts(self):
        fact_list = FileTools.load_list_p(self.dataset.facts_out)
        self.facts = {eid: elist for eid, elist in fact_list}

    @staticmethod
    def save_blocks(fs1, fs2):
        if args.blocking:

            pass
        pass

    # @staticmethod
    # def not_blocking(fs1, fs2):
    #     print(Announce.printMessage(), 'not blocking')
    #     fs1: KBStore
    #     fs2: KBStore
    #     print(Announce.printMessage(), 'get all entities')
    #     e1s = set(fs1.entity_ids.values())
    #     e2s = set(fs2.entity_ids.values())
    #     print(Announce.doing(), 'save all entities as a whole block')
    #     with open(links.block, 'w', encoding='utf-8') as wfile:
    #         print((e1s, e2s), file=wfile)
    #     print(Announce.done())

    def get_property_table_line(self, line):
        e, ei = line
        e: str
        ename = e.split('/')[-1]
        dic = {'id': ei, 'ent_name': ename}
        facts = self.literal_facts.get(ei)
        # if facts is None:
        #     return None
        if facts is not None:
            fact_aggregation = defaultdict(list)
            for fact in facts:
                # 过滤函数性低的
                if functionality_control and self.properties_functionality[fact[0]] <= functionality_threshold:
                    continue
                fact_aggregation[fact[0]].append(self.literals[fact[1]])

            for pid, objs in fact_aggregation.items():
                pred = self.properties[pid]
                obj = ' '.join(objs)
                dic[pred] = obj

        return dic
        # writer.writerow(dic)
        pass

    @staticmethod
    def load_path(path, load_func, file_type: OEAFileType) -> None:
        print(Announce.doing(), 'Start loading', path)
        if os.path.isdir(path):  # 判断给定的路径是否是一个目录
            for file in sorted(os.listdir(path)): # os.listdir(path)返回了path下的所有文件和子目录的列表，然后按照字母顺序进行排序
                if os.path.isdir(file):
                    continue
                file = os.path.join(path, file)
                # load_func(file, type)
                KBStore.load_path(file, load_func, file_type)
        else:
            load_func(path, file_type)  # load_func 就是后面的load函数
        print(Announce.done(), 'Finish loading', path)

    def load(self, file: str, file_type: OEAFileType) -> None:
        tuples = Parser.for_file(file, file_type)
        with tqdm(desc='add tuples', file=sys.stdout) as tqdm_add:  # desc -- 进度条标题  total -- 预期的迭代次数 ，with -- 创建一个上下文管理器，并命名为tqdm_add file -- 指定输出进度消息的位置，
            tqdm_add.total = len(tuples)
            for args in tuples:
                self.add_tuple(*args, file_type) # 将实体，属性，属性值/头实体，关系，尾实体在数据集文件中出现的位置，他们的id值，他们的对应关系保存到对应的集合，列表中
                tqdm_add.update()
        pass

    def add_tuple(self, sbj: str, pred: str, obj: str, file_type: OEAFileType) -> None:
        assert sbj is not None and obj is not None and pred is not None, 'sbj, obj, pred None'
        if file_type == OEAFileType.attr:
            if obj.startswith('"'):
                obj = obj[1:-1]  # 以双引号开头则去掉双引号
            toks = text_to_word_sequence(obj)  # 将文本字符串转换为单词列表，这里是将属性三元组中的每个部分都转换为一个列表，如toks = ['mrna']
            for tok in toks:
                if len(tok) < 5:
                    continue
                if bool(re.search(r'\d', tok)):  # re.search(r'\d', tok) -- 正则表达式，用于判断tok是否包含数字，存在则返回一个匹配对象，并转换为True值，不存在返回一个None，转换为False
                    return
            sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids)  # 将实体，属性，属性值分别加入对应的字典和列表中
            obj_id = self.get_or_add_item(obj, self.literals, self.literal_ids)
            pred_id = self.get_or_add_item(pred, self.properties, self.property_ids)
            self.add_fact(sbj_id, pred_id, obj_id, self.literal_facts)
            self.add_to_blocks(sbj_id, obj_id)
            words = text_to_word_sequence(obj)
            self.add_word_level_blocks(sbj_id, words)
        elif file_type == OEAFileType.rel:  # 处理关系
            sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids) # 添加头实体
            obj_id = self.get_or_add_item(obj, self.entities, self.entity_ids) # 添加尾实体
            pred_id = self.get_or_add_item(pred, self.relations, self.relation_ids)  # 添加关系
            pred2_id = self.get_or_add_item(pred + '-', self.relations, self.relation_ids)
            self.add_fact(sbj_id, pred_id, obj_id, self.facts)  # 正向添加三元组
            self.add_fact(obj_id, pred2_id, sbj_id, self.facts) # 反向添加三元组

    def add_item(self, name: str, names: list, ids: dict) -> int:
        iid = len(names)
        names.append(name)
        ids[name] = iid
        return iid

    def get_or_add_item(self, name: str, names: list, ids: dict) -> int:
        if name in ids:
            return ids.get(name)
        else:
            return self.add_item(name, names, ids)

    def add_fact(self, sbj_id, pred_id, obj_id, facts_list: dict) -> None:
        if sbj_id in facts_list:  # 根据头实体判断关系三元组是否在fact——list表中
            facts: list = facts_list.get(sbj_id)
            facts.append((pred_id, obj_id))
        else:
            facts_list[sbj_id] = [(pred_id, obj_id)]

    def add_to_blocks(self, sbj_id, obj_id) -> None:
        if obj_id in self.blocks:
            block: set = self.blocks.get(obj_id)
            block.add(sbj_id)
        else:
            self.blocks[obj_id] = {sbj_id}  # 保存的是当前属性值在三元组中所出现的位置，也就是实体的位置

    def add_word_level_blocks(self, entity_id, words):
        for word in words:
            if word in self.word_level_blocks:
                block: set = self.word_level_blocks.get(word)
                block.add(entity_id)
            else:
                self.word_level_blocks[word] = {entity_id}  # 与block类似，只不过不是属性值的id，而是属性值
        pass
    # 参数为关系，关系id，关系三元组对应结构，实体id
    @staticmethod
    def calculate_func(r_names: list, r_ids: dict, facts_list: dict, sbj_ids: dict) -> list:
        num_occurrences = [0] * len(r_names) # num_occurrences用来记录每个关系出现的次数，关系的位置和relation列表是对应的
        func = [0.] * len(r_names) # 初始化一个列表，长度与关系名列表一致，列表元素类型为float
        num_subjects_per_relation = [0] * len(r_names)
        last_subject = [-1] * len(r_names) # 记录当前关系最后一次出现的实体的id

        for sbj_id in sbj_ids.values():  # 获取字典中所有的值
            facts = facts_list.get(sbj_id)
            if facts is None:
                continue
            for fact in facts:
                num_occurrences[fact[0]] += 1  # fact[0]表示关系 ，fact[1]表示尾实体
                if last_subject[fact[0]] != sbj_id:
                    last_subject[fact[0]] = sbj_id
                    num_subjects_per_relation[fact[0]] += 1 # 当前关系最后一次出现的实体id与本次实体id不一致时，统计关系出现的次数

        for r_name, rid in r_ids.items():
            func[rid] = num_subjects_per_relation[rid] / num_occurrences[rid] # 计算不同实体存在这种关系/关系总的出现次数
            print(Announce.printMessage(), rid, r_name, func[rid], sep='\t')
        return func
