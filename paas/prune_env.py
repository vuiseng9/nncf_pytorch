import os
import torch
import torch.nn as nn

import pandas as pd
import numpy as np

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from nncf.common.utils.logger import logger as nncf_logger
from nncf.pruning.filter_pruning.algo import FilterPruningController
from nncf.graph.graph import PTNNCFGraph

from collections import defaultdict
class PruneEnv:
    def __init__(self, 
        pruning_controller, pruned_model, nncf_cfg, evaluator, train_loader, val_loader):
        if isinstance(pruning_controller, FilterPruningController):
            self.pruning_controller = pruning_controller
            self.pruned_model = pruned_model
            self.nncf_cfg = nncf_cfg
            self.evaluator = evaluator
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            raise ValueError("PruneEnv requires a filter-prune wrapped controller and model")
        self.df = self.extract_prunable_layer_features()
        self.node_type_lut, self.connectivity_lut = self.extract_graph_connectivity()
        self.print_groupwise_nodes()
        self.visualize_groupwise_graph()

    @property
    def original_flops(self):
        return int(self.pruning_controller.full_flops)
    
    @property
    def remaining_flops(self):
        return int(self.pruning_controller.current_flops)
    
    @property
    def flop_ratio(self):
        return self.remaining_flops/self.original_flops

    @property
    def effective_pruning_rate(self):
        return self.pruning_controller._calculate_global_weight_pruning_rate()

    @property
    def groupwise_pruning_rate(self):
        return self.pruning_controller.current_groupwise_pruning_rate

    @property
    def layerwise_stats(self):
        return self.pruning_controller.get_stats_for_pruned_modules()

    def extract_graph_connectivity(self):
        g = self.pruned_model.get_graph()
        nx_digraph = g._get_graph_for_structure_analysis()

        node_type = dict()
        for n in nx_digraph.nodes:
            node_type[n] = g.get_nx_node_by_key(n)['ia_op_exec_context'].operator_name

        edge_connectivity = defaultdict(list) # key: src_node, val: set(dst_nodes)
        for e in nx_digraph.edges:
            edge_connectivity[e[0]].append(e[1])

        return node_type, edge_connectivity

    def evaluate_valset(self, pruning_rate_cfg):
        self.pruning_controller.set_pruning_rate(pruning_rate_cfg)
        return self.evaluator(self.pruned_model, self.val_loader)

    def print_groupwise_nodes(self):
        for cluster in self.pruning_controller.pruned_module_groups_info.get_all_clusters():
            for node in cluster.nodes:
                print("Group {:2d} | {:3d} | {}".format(cluster.id, node.nncf_node_id, node.module_scope.__str__()))
            print("----------------------------------------------------------------------------------------------")


    def extract_prunable_layer_features(self):
        def get_layer_attr(m):        
            feature=defaultdict(lambda: 0)
            if isinstance(m, nn.Conv2d):
                feature['depthwise']  = int(m.in_channels == m.groups) # 1.0 for depthwise, 0.0 for other conv2d
                feature['cin']          = m.in_channels
                feature['cout']         = m.out_channels
                feature['stride']       = m.stride[0]
                feature['kernel']       = m.kernel_size[0]
                feature['param']        = np.prod(m.weight.size())     
                feature['ifm']     = np.prod(m._input_shape[-2:]) # H*W
                feature['ofm']     = np.prod(m._output_shape[-2:]) # H*W
            else:
                raise ValueError("unsupported module {}".format(m.__class__.__name__))
            return feature

        def annotate_learnable_module_io_shape(model):
            def annotate_io_shape(module, input_, output):
                if not isinstance(output, tuple):
                    module._input_shape  = input_[0].shape
                    module._output_shape = output.shape

            hook_list = [m.register_forward_hook(annotate_io_shape) for n, m in model.named_modules()]
            model.do_dummy_forward(force_eval=True)
            for h in hook_list:
                h.remove()
            
        annotate_learnable_module_io_shape(self.pruned_model)

        g = self.pruned_model.get_graph()
        layer_dictlist = []
        for cluster in self.pruning_controller.pruned_module_groups_info.get_all_clusters():
            for node in cluster.nodes:
                node_features=get_layer_attr(node.module)
                node_features['cluster_id'] = cluster.id
                node_features['node_id'] = node.nncf_node_id
                node_features['module_scope'] = node.module_scope.__str__()
                node_features['node_name'] = g.get_nncf_node_by_id(node.nncf_node_id).node_name
                layer_dictlist.append(node_features)
        
        return pd.DataFrame.from_dict(layer_dictlist)

    def visualize_groupwise_graph(self, path=None):
        g = self.pruned_model.get_graph()

        palette = ['yellow', 'pink', 'lightblue']

        out_graph = nx.DiGraph()
        for node_name, node in g._nx_graph.nodes.items():
            ia_op_exec_context = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR]

            attrs_node = {}
            label = str(node[PTNNCFGraph.ID_NODE_ATTR]) + ' ' + str(ia_op_exec_context)
            if 'conv2d' in label.lower():
                label = "*prunable*\n" + label
            tokens=label.split("/")
            new_tokens=[]
            for i, token in enumerate(tokens):
                if (i+1)%2==0:
                    token += "\n"
                new_tokens.append(token)
            attrs_node['label'] = '/'.join(new_tokens)
            if sum(self.df.node_name == node_name) == 1:
                cluster_id = self.df.cluster_id[self.df.node_name == node_name].values[0]
                attrs_node['label'] += "\n(cluster {})".format(cluster_id)
                attrs_node['color'] = palette[cluster_id % len(palette)]
                attrs_node['style'] = 'filled'

            out_graph.add_node(node_name, **attrs_node)

        for u, v in g._nx_graph.edges:
            out_graph.add_edge(u, v, label=g._nx_graph.edges[u, v][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")

        if path is None:
            path = os.path.join(self.nncf_cfg.get("log_dir", "."), "prune_env.dot")
        
        nx.drawing.nx_pydot.write_dot(out_graph, path)
        try:
            A = to_agraph(out_graph)
            A.layout('dot')
            png_path = os.path.splitext(path)[0]+'.png'
            A.draw(png_path)
        except ImportError:
            nncf_logger.warning("Graphviz is not installed - only the .dot model visualization format will be used. "
                                "Install pygraphviz into your Python environment and graphviz system-wide to enable "
                                "PNG rendering.")
