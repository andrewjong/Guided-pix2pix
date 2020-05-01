import os
import os.path as osp
from tqdm import tqdm
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import util 


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    model = create_model(opt)
    model.setup(opt)

    # results are saved in task_results
    results_path = opt.results_dir
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    output_list = []
    target_list = []

    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for i, data in pbar:
        if i >= opt.num_test:
            break

        if "guide_path" in data:
            input_path = data["guide_path"][0] # take out the batch
            out_path = osp.sep.join(input_path.split(osp.sep)[-2:])[:-15] # remove _keypoints.json
        else:
            out_path = str(i)
        pbar.set_description(out_path)
        # process
        input = data['A']
        target = data['B']
        guide = data['guide']
        model.set_input(data)
        model.test()
        output = model.get_output()

		# save results
        util.save(input, guide, target, output, results_path, out_path, opt)
    print ('Results saved in %s'%results_path)
    
