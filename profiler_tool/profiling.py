import os
import sys
import mxnet as mx
import argparse

import time

def parse_args():
    parser = argparse.ArgumentParser(description='Convert results from pkl to json')
    parser.add_argument('--symbol', help='symbol path', required=True, type=str)
    parser.add_argument('--params', help='params path', required=True, type=str)
    parser.add_argument('--out_file', help='output_json_path', default='network_profiler.json', type=str)
    parser = parser.parse_args() 
    return parser


def set_env():
    os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
    os.environ['MXNET_EXEC_BULK_EXEC_TRAIN'] = '0'
    os.environ['MXNET_EXEC_BULK_EXEC_INFERENCE'] = 'false'


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_env()

    # mx.profiler.set_config(profile_symbolic=True, filename=self.config.TEST.profiler_file)
    mx.profiler.set_config(profile_all=True, filename=self.config.TEST.profiler_file)
    print('profile file save to {0}'.format(self.config.TEST.profiler_file))            

    assert len(data_batch) == 1
    assert len(self.mod_list) == 1

    # dry run
    for i in range(self.config.TEST.profiler_dry_run):
        self.mod_list[0].forward(data_batch[0])

    # real run
    profiler.set_state('run')
    t0 = time.clock()
    for i in range(self.config.TEST.profiler_real_run):
        self.mod_list[0].forward(data_batch[0])
        for output in self.mod_list[0].get_outputs(merge_multi_context=True)[0]:
            output.wait_to_read()
    profiler.set_state('stop')
    t1 = time.clock()
    print('{} ms/it'.format((t1 - t0)*1000.0 / self.config.TEST.profiler_real_run))
    print('Finish profile! set config.TEST.use_profiler=False to run real predicting.')
