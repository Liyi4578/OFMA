# -*- coding:utf-8 -*-
# --Based on YOLOX made by Megvii, Inc. and its affiliates.--
import torch

class DataPrefetcher:
    """
    
    改造成了迭代器
    
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.orign_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, = next(self.loader)
        except StopIteration:
            
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            
            self.input_cuda()
            # self.next_target = self.next_target.cuda(non_blocking=True)
            if isinstance(self.next_input,list):
                for i in range(len(self.next_input)):
                    self.next_input[i] = self.next_input[i].cuda(non_blocking=True)
            else:
                self.next_input = self.next_input.cuda(non_blocking=True)
    
    # 迭代器有头有尾
    def __len__(self):
        return len(self.loader)
    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        else:
            self.loader = iter(self.orign_loader)
            self.preload()
            raise StopIteration
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
    
    def __iter__(self):
        return self
    
    def _input_cuda_for_image(self):
        if isinstance(self.next_input,list):
            for i in range(len(self.next_input)):
                self.next_input[i] = self.next_input[i].cuda(non_blocking=True)
        else:
            self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
