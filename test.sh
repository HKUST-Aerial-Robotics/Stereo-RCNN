CUDA_VISIBLE_DEVICES=5 python test_net.py --net res101 --checksession 1 --checkepoch 8 --checkpoint 13295 --cuda &&
python compensate_empty_test.py
