"""
    Batch-generate data
"""

import os
import numpy as np
import multiprocessing as mp
from subprocess import call
from utils import printout
import time


class GraspDataGen(object):

    def __init__(self, num_processes, flog=None):
        self.num_processes = num_processes
        self.flog = flog
        
        self.todos = []
        self.processes = []
        self.is_running = False
        self.Q = mp.Queue()

    def __len__(self):
        return len(self.todos)

    def add_one_collect_job(self, data_dir, category, trial_id):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('COLLECT', category, data_dir, trial_id, np.random.randint(10000000))
        self.todos.append(todo)
    
    def add_one_recollect_job(self, src_data_dir, dir1, dir2, recollect_record_name, tar_data_dir, x, y):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('RECOLLECT', src_data_dir, recollect_record_name, tar_data_dir, np.random.randint(10000000), x, y, dir1, dir2)
        self.todos.append(todo)
    
    def add_one_checkcollect_job(self, src_data_dir, dir1, dir2, recollect_record_name, tar_data_dir, x, y):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('CHECKCOLLECT', src_data_dir, recollect_record_name, tar_data_dir, np.random.randint(10000000), x, y, dir1, dir2)
        self.todos.append(todo)
    
    @staticmethod
    def job_func(pid, todos, Q):
        succ_todos = []
        print("start collect.py")
        for todo in todos:
            if todo[0] == 'COLLECT':
                cmd = 'python collect_data_threefinger.py %s --out_dir %s --trial_id %d --random_seed %d --no_gui' \
                        % (todo[1], todo[2], todo[3], todo[4])
                folder_name = todo[2]
                job_name = '%s_%s' % (todo[1], todo[3])
            elif todo[0] == 'RECOLLECT':
                cmd = 'python recollect_data.py %s %s %s --random_seed %d --no_gui --x %d --y %d --dir1 %s --dir2 %s' \
                        % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7], todo[8])
                print("recollect", (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7], todo[8]))
                folder_name = todo[3]
                job_name = todo[2]
            elif todo[0] == 'CHECKCOLLECT':
                cmd = 'python checkcollect_data.py %s %s %s --random_seed %d --no_gui --x %d --y %d --dir1 %s --dir2 %s > /dev/null 2>&1' \
                        % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7], todo[8])
                folder_name = todo[3]
                job_name = todo[2]
            ret = call(cmd, shell=True)
            if ret == 0:
                succ_todos.append(os.path.join(folder_name, job_name))
            if ret == 2:
                succ_todos.append(None)
            print("call ret", ret)
        Q.put(succ_todos)

    def start_all(self):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot start all while DataGen is running!')
            exit(1)

        total_todos = len(self)
        num_todos_per_process = int(np.ceil(total_todos / self.num_processes))
        np.random.shuffle(self.todos) #将 self.todos 列表或数组中的元素顺序随机打乱。
        for i in range(self.num_processes):
            todos = self.todos[i*num_todos_per_process: min(total_todos, (i+1)*num_todos_per_process)]
            p = mp.Process(target=self.job_func, args=(i, todos, self.Q)) # mp.Process 被用来创建一个新的进程，该进程将执行 self.job_func 函数，并传递给它一些参数。
            p.start()
            self.processes.append(p)
        
        self.is_running = True

    def join_all(self):
        if not self.is_running:
            printout(self.flog, 'ERROR: cannot join all while DataGen is idle!')
            exit(1)

        ret = []
        for p in self.processes:
            ret += self.Q.get()

        for p in self.processes:
            p.join()

        self.todos = []
        self.processes = []
        self.Q = mp.Queue()
        self.is_running=False
        return ret


