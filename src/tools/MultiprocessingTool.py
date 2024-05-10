import multiprocessing as mp
import sys
from itertools import chain
from threading import Thread

from tqdm import tqdm

from tools.Announce import Announce

# 多进程工具类，完成单进程到并发执行的转换。
class MultiprocessingTool:
    def __init__(self, num_workers=10, use_semaphore=False):
        self._num_workers = num_workers
        self.__sender_queue = mp.Queue() # 创建多进程队列
        self.__result_queue = mp.Queue()
        self.__processes = [] # 空列表，用于存储创建的工作进程对象。
        self.__pack_count = 0
        self._use_semaphore = use_semaphore
        self.invoke_freq = 100
        self.__sem = mp.Semaphore(2000) # 创建一个具有初始计数值2000的信号量对象，并将其赋给实例变量__sem。这个信号量可能用于限制并发访问。
        self._min_process_count = 500000 # 最小进程数
        self._verbose = False

        # self.tqdm_pack_solver = tqdm(desc='solved packages')
        # self.tqdm_receiver = tqdm(desc='received packages')
        # 发送包
    def send_packs(self, iterator, pack_size=5000):
        # 创建一个tqdm对象并将其赋值给类的self.tqdm_receiver实例变量，用于显示进度条
        self.tqdm_receiver = tqdm(desc='received packages', file=sys.stdout)
        pack = [None] * pack_size
        i = 0
        j = 0
        for item in iterator:
            # 扩充pack_size，因为数据量太大，一个包的数量最大是5000，看最后接收多少个包
            if i == pack_size:
                self.__sender_queue.put((j, pack))
                if self._verbose and j % self.invoke_freq == 0:
                    print(' '.join((Announce.printMessage(), 'Pack', str(j), 'sent')))
                pack = [None] * pack_size
                i = 0
                j += 1
            pack[i] = item
            i += 1
        # 去除pack中值为None的元素
        pack = [item for item in pack if item is not None]
        self.__sender_queue.put((j, pack))
        self.__pack_count = j + 1
        # self.tqdm_pack_solver.total = self.__pack_count
        # self.__pack_count变量存储了需要接收的数据包总数，通过将其赋值给tqdm对象的total属性，可以在接收数据包时动态地显示接收进度
        self.tqdm_receiver.total = self.__pack_count
        for i in range(self._num_workers):
            self.__sender_queue.put((None, None))
        print(' '.join((Announce.printMessage(), 'Finished sending packs')))
        print(' '.join((Announce.printMessage(), 'Total Pack Count:', str(self.__pack_count))))
        return self

    def packed_solver(self, item_solver, **kwargs):
        for i in range(self._num_workers):
            # 创建一个进程对象
            # target -- 指定进程要执行的目标函数或方法
            # args -- 传递给目标函数的位置参数的元组
            # kwargs -- 传递给目标函数的关键字参数的字典
            p = mp.Process(target=self._packed_solver, args=(item_solver,), kwargs=kwargs)
            # 将进程对象p添加到列表中，可以用于后续对该进程的操作
            self.__processes.append(p)
            p.start()
        return self
# 判断元素是否符合正则表达式，符合则存进结果队列中
    def _packed_solver(self, item_solver, **kwargs):
        while True:
            if self._use_semaphore:
                self.__sem.acquire() # self.__sem是一个信号量对象，使用信号量锁定资源，有资源继续执行，没有则阻塞
                # id  -- 用于判断队列中是否还有数据，否则返回为None
                # pack是数据集中某一个文件中所有的数据，是一个列表，需要取出来进行预处理
                # get方法用于从队列中取数据，取完从队列中移出
            id, pack = self.__sender_queue.get()
            if id is None:
                break # 队列中元素全部取出，循环结束
            result_pack = [None] * len(pack)
            # 遍历pack中的元素，i是元素索引，item是值
            for i, item in enumerate(pack):
                # 对元素进行处理，大致就是判断元素是否符合正则表达式，符合则取出每个元素并按照规则分开 。返回一个元组
                result = item_solver(item, **kwargs) # item_solver ： ttl_no_compress_line
                result_pack[i] = result
            self.__result_queue.put((id, result_pack)) # 存进结果队列中，也会有值为None的元素
            #  self._verbose用于判断是否输出详细信息
            if self._verbose:
                print(' '.join((Announce.printMessage(), 'Pack', str(id), 'finished')))
            # self.tqdm_pack_solver.update()
            # sleep(1)
        self.__result_queue.put((None, None)) # 最后存入（None，None），最后也会取出（None，None）用于判断队列中元素是否取完
    # 接收包
    def receive_results(self, processor=None, **kwargs):
        finished_workers = 0
        result_packs = [None] * self.__pack_count
        while True:
            id, result_pack = self.__result_queue.get()
            if id is None:
                finished_workers += 1
                if finished_workers == self._num_workers:
                    break
                continue
            try:
                if processor is not None:
                    result_pack = processor(result_pack, **kwargs)
                result_packs[id] = result_pack
            except IndexError:
                print('err', id, len(result_packs))

            if self._verbose:
                print(' '.join((Announce.printMessage(), 'Result', str(id), 'received')))
            self.tqdm_receiver.update() # 更新进度条
        self.tqdm_receiver.close() # 关闭进度条
        return self._results_unpack(result_packs)

    def _results_unpack(self, result_packs):
        # results = list(_flatten(result_packs))
        results = list(chain.from_iterable(result_packs))
        print(Announce.printMessage(), 'concat finished')
        return results

    def reveive_and_process(self, processor, **kwargs):
        finished_workers = 0
        tp = None
        old_id = None
        old_pack = None
        first = True
        multi_pack = None
        count = 0
        pack_count = 0
        while True:
            id, result_pack = self.__result_queue.get()
            if self._use_semaphore:
                self.__sem.release()
            # print(result_pack)
            if id is None:
                finished_workers += 1
                if finished_workers == self._num_workers:
                    break
                continue
            pack_count += 1
            self.tqdm_receiver.update()
            if multi_pack is None:
                multi_pack = result_pack
            else:
                # multi_pack: list
                multi_pack.extend(result_pack)
            c = map(lambda x: len(x), result_pack)
            d = sum(c)
            count += d
            if self._verbose:
                print(' '.join((Announce.printMessage(), 'Result', str(id), str(pack_count), 'get', 'count:', str(count))))
            if count < self._min_process_count:
                continue
            # print(' '.join((Announce.printMessage(), str(id), str(first))))
            if not first:
                tp.join()
                if self._verbose:
                    print(' '.join((Announce.printMessage(), 'Result', str(old_id), 'finished')))
            if first:
                first = False
            # processor(result_pack, **kwargs)
            old_id = id
            old_pack = multi_pack
            print(Announce.printMessage(), 'processing', id)
            tp = Thread(target=processor, args=(old_pack,), kwargs=kwargs)
            tp.start()
            multi_pack = None
            count = 0
        if tp is not None:
            tp.join()
        if self._use_semaphore:
            self.__sem.release()
        print(len(multi_pack))
        if multi_pack is not None:
            print('run')
            old_pack = multi_pack
            tp = Thread(target=processor, args=(old_pack,), kwargs=kwargs)
            tp.start()
            tp.join()
        print(' '.join((Announce.printMessage(), 'Result', str(old_id), 'finished')))
        self.tqdm_receiver.close()
        print(Announce.printMessage(), 'all packs finished')
        pass


MPTool = MultiprocessingTool()

if __name__ == '__main__':
    tool = MultiprocessingTool(num_workers=2)
    l = [i for i in range(500)]
    tool.packed_solver(lambda x, y: x * y, y=2)
    tool.send_packs(l, 5)
    results = tool.receive_results()
    # tool.reveive_and_process(lambda x: print(x))
    # print(results)
