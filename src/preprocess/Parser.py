import re
from enum import Enum
from typing import List, Tuple

from tools.MultiprocessingTool import MultiprocessingTool

pref = {
    "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd:": "http://www.w3.org/2001/XMLSchema#",
    "owl:": "http://www.w3.org/2002/07/owl#", "skos:": "http://www.w3.org/2004/02/skos/core#",
    "dc:": "http://purl.org/dc/terms/",
    "foaf:": "http://xmlns.com/foaf/0.1/",
    "vcard:": "http://www.w3.org/2006/vcard/ns#",
    "dbp:": "http://dbpedia.org/",
    "y1:": "http://www.mpii.de/yago/resource/",
    "y2:": "http://yago-knowledge.org/resource/",
    "geo:": "http://www.geonames.org/ontology#",
    'wiki:': 'http://www.wikidata.org/',
    'schema:': 'http://schema.org/',
    'freebase:': 'http://rdf.freebase.com/',
    'dbp_zh': 'http://zh.dbpedia.org/',
    'dbp_fr': 'http://fr.dbpedia.org/',
    'dbp_ja': 'http://ja.dbpedia.org/',
}


class OEAFileType(Enum):
    attr = 0
    rel = 1
    ttl_full = 2


def strip_square_brackets(s):
    # s = ""
    if s.startswith('"'):
        rindex = s.rfind('"') # 找到最右边的双引号的索引
        if rindex > 0:
            s = s[:rindex + 1]
    else:
        if s.startswith('<'):
            s = s[1:]
        if s.endswith('>'):
            s = s[:-1]
    return s


def compress_uri(uri):
    uri = strip_square_brackets(uri)
    if uri.startswith("http://"):
        for key, val in pref.items():
            if uri.startswith(val):
                uri = uri.replace(val, key)
    return uri


def oea_attr_line(line: str):
    fact: List[str] = line.strip('\n').split('\t')  # 将结果赋值给名为fact的字符串列表
    if not fact[2].startswith('"'):
        fact[2] = ''.join(('"', fact[2], '"'))  # 判断属性值是不是有双引号，没有则给他加上
    return compress_uri(fact[0]), compress_uri(fact[1]), compress_uri(fact[2])


def oea_rel_line(line: str) -> Tuple:
    fact: List[str] = line.strip('\n').split('\t')
    # fact: List[str] = line.strip('\n').split(' ')
    # fact = [rel for rel in fact if rel != '']
    return compress_uri(fact[0]), compress_uri(fact[1]), compress_uri(fact[2])


def oea_truth_line(line: str) -> Tuple:
    fact: List[str] = line.strip().split('\t')
    return compress_uri(fact[0]), compress_uri(fact[1])


# ([^\\s]+): 匹配一个或多个非空白字符。
# \\s+: 匹配一个或多个空白字符（包括空格、制表符、换行符等）。
# ([^\\s]+): 匹配一个或多个非空白字符。
# \\s+: 匹配一个或多个空白字符。
# (.+): 匹配一个或多个任意字符（除了换行符）。
# \\s*: 匹配零个或多个空白字符。

ttlPattern = "([^\\s]+)\\s+([^\\s]+)\\s+(.+)\\s*"
# ttlPattern = "([^\\s]+)\\s+([^\\s]+)\\s+([^\\s]+)"


def stripSquareBrackets(s):
    # s = ""
    if s.startswith('"'):
        rindex = s.rfind('"')
        if rindex > 0:
            s = s[:rindex + 1]
    else:
        if s.startswith('<'):
            s = s[1:]
        if s.endswith('>'):
            s = s[:-1]
    return s

# line -- pack中的元素，即数据集中的某个文件的元素
def ttl_no_compress_line(line):
    if line.startswith('#'):
        # 判断是否#号开头，代表这个元素是注释掉的，不用处理
        return None, None, None
    # 使用正则表达式对字符串进行匹配
    # line.rstrip去除字符串末尾的空白字符
    # 返回一个匹配对象，，匹配失败返回None
    fact = re.match(ttlPattern, line.rstrip())
    if fact is None:
        print(line)
    sbj = stripSquareBrackets(fact[1])
    pred = stripSquareBrackets(fact[2])
    obj = stripSquareBrackets(fact[3])
    return sbj, pred, obj

# file：zh_att_triples
def for_file(file, file_type: OEAFileType) -> list:
    line_solver = None
    if file_type == OEAFileType.attr:
        line_solver = oea_attr_line
    elif file_type == OEAFileType.rel:
        line_solver = oea_rel_line
    elif file_type == OEAFileType.ttl_full:
        line_solver = ttl_no_compress_line
        # 如果line_solver值为None，则断言错误，为了确保函数不为空，line_solver是一个函数
    assert line_solver is not None
    print(file)
    with open(file, 'r', encoding='utf-8') as rfile:
        # 初始化多进程的工具类的对象
        mt = MultiprocessingTool()
        # results - list列表，列表的元素是一个元组[('IL-2', 'rna类型', 'mRN'),('IL-3', 'rna类型', 'mRN')]
        results = mt.packed_solver(line_solver).send_packs(rfile).receive_results()
        # results = [line_solver(line) for line in rfile]
        results = [triple for triple in results if triple[0] is not None]
    return results
