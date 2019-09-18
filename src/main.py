import os
import tensorflow as tf
from utils import load_train_ini
from oprations import CascadedModel
import os

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def main(_):
    # load training parameters #
    ini_file = 'parameters.ini'
    param_sets = load_train_ini(ini_file)
    param_set = param_sets[0]

    # print config parameters
    display_config(param_set)

    print('====== Phase >>> %s <<< ======' % param_set['phase'])

    if not os.path.exists(
            param_set['chkpoint_dir']):  # ../outcome/model/checkpoint
        os.makedirs(param_set['chkpoint_dir'])
    if not os.path.exists(param_set['labeling_dir']):
        os.makedirs(param_set['labeling_dir'])

    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # GPU configuration，per_process_gpu_memory_fraction means 95％GPU MEM
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.97,
        allow_growth=True)
    graph_options = tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0))
    with tf.Session(config=tf.ConfigProto(device_count={"cpu": 5}, intra_op_parallelism_threads=1,
                                          inter_op_parallelism_threads=5, gpu_options=gpu_options,
                                          graph_options=graph_options, allow_soft_placement=True)) as sess:
        model = CascadedModel(sess, param_set)
        if param_set['phase'] == 'train':
            try:
                if os.path.exists("train.log"):
                    os.remove("train.log")
                if os.path.exists("test.log"):
                    os.remove("test.log")
            except BaseException:
                pass
            model.train()  # training process
        elif param_set['phase'] == 'gen_map':
            print("gen map!")
            if os.path.exists("generate_map.log"):
                os.remove("generate_map.log")
            model.test_generate_map()
        elif param_set['phase'] == 'test':
            try:
                if os.path.exists("test_result.log"):
                    os.remove("test_result.log")
            except BaseException:
                pass
            model.test4crsv()


# display configurations
def display_config(src_list):
    """write Configuration values to a file."""
    print("\nConfigurations:")
    for (k, v) in src_list.items():
        print("{:30} {}".format(k, v))
    print("\n")


if __name__ == '__main__':
    tf.app.run()
