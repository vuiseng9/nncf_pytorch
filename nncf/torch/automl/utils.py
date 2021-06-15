import pandas as pd
import numpy as np

class AutoQ_Summarizer:
    def __init__(self, compression_ctrl, nncf_config):
        self._compression_ctrl = compression_ctrl
        self.autoq_bool = self.get_autoq_config(nncf_config)
    
    def bw_dist(self):
        stats = self._compression_ctrl.statistics()
        qdf = pd.DataFrame.from_dict(
            [stats.quantization.bitwidth_distribution_statistics.num_wq_per_bitwidth, 
             stats.quantization.bitwidth_distribution_statistics.num_aq_per_bitwidth]).fillna(0).astype(int).rename(index={0:'WQ',1:'AQ'})
        return qdf

    def get_model_size_ratio(self):
        total_nparam = 0
        total_bit = 0
        for qid, qinfo in self._compression_ctrl.weight_quantizers.items():
            nparam = np.prod(qinfo.quantized_module.weight.shape)
            bw = qinfo.quantizer_module_ref.num_bits
            bit_per_layer = nparam*bw
            total_nparam += nparam
            total_bit += bit_per_layer
        size_ratio = (total_bit)/(total_nparam*32)
        return size_ratio
    
    def write_onnx(self, onnx_pth):
        self._compression_ctrl.export_model(onnx_pth)
        
    def register_final_accuracy(self, compressed_acc):
        self.compressed_accuracy = compressed_acc

    def get_autoq_config(self, nncf_config):
        if nncf_config.get('compression', {}).get('initializer', {}).get('precision', {}).get('type', {}) == 'autoq':
            self.autoq_cfg = nncf_config.get('compression', {}).get('initializer', {}).get('precision', {})
            return True
        return False

