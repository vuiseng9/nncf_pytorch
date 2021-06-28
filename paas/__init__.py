import os,sys
from typing import Protocol
from flask import Flask, g
from flask import request, send_file, abort
import json, random
import numpy as np
import gc
import time
from multiprocessing import Lock, Semaphore
from multiprocessing.sharedctypes import Value
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from natsort import natsorted
import hashlib
lock = Semaphore(1)
mutex= Lock()

from .prune_env import PruneEnv
from examples.torch.classification.main import main as imgnet

from copy import deepcopy
import logging, pandas
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)
def prRed(prt): print("\033[91m {}\033[00m".format(prt),flush=True)
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt),flush=True)
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt),flush=True)

def init_workload():
    if os.environ['workload'] == 'imgnet':
        _args = [
            '--gpu-id', '0', 
            '--workers', '6', 
            '--log-dir', '/tmp/paas-imgnet-log/',  
            '--config', os.environ['config'],
            '--data',   os.environ['data']]
        return imgnet(_args)
    else:
        raise ValueError("Environment variable workload is not valid")

def dummy_response():
    return 	{
        'rc': 0, 
        'done': 1, 
        'reward': random.uniform(0.0, 1.4), 
        'meta_data': {
            'acccuracy': random.uniform(0.0, 0.95), 
            'modelsize': random.uniform(0.0625, 0.25)
            }
        }

def acquire_lock(calling_method):
    global concurrent_requests_value
    global last_lock_time

    try:
        if lock.acquire(block=False):
            last_lock_time = time.time()
            concurrent_requests_value += 1
            prCyan("acquired_lock - " + calling_method+", concurrent_requests_value="+str(concurrent_requests_value))
            return True
        else:
            prRed("Blocked since: " + str(time.time()-last_lock_time))
            return  False
    except:
        prRed("BlockedY")
        pass

def release_lock(calling_method):
    global concurrent_requests_value
    global last_lock_time
    lock.release()
    last_lock_time = time.time()
    concurrent_requests_value -= 1
    prCyan("release_lock - " + calling_method+", concurrent_requests="+str(concurrent_requests_value))
    return  True
        
def create_app() -> Flask:
        
    global concurrent_requests_value
    global max_thread_time
    bEnvReady = False
    concurrent_requests_value = 0

    '''Create an app by initializing components'''
    app = Flask(__name__)
    acquire_lock("WorkloadInit")
    t1 = time.time()
    env = PruneEnv(*(init_workload()))
    max_thread_time = int(5*(time.time()-t1))
    prRed("max_thread_time="+str(max_thread_time)+" sec.")
    release_lock("WorkloadInit")

    bEnvReady = True
    print("{} Pruning Service initialized".format(os.environ['workload']), flush=True)

    @app.route('/ready_state')
    def readystate():
        if bEnvReady is True:
            return {'method':'ready_state', 'rc': 0, 'msg':"Environment {} initialized".format(os.environ['workload']), 'config':os.environ['config']}
        return {'method': 'ready_state', 'rc': 1, 'msg': "Environment {} not yet initialzed".format(os.environ['workload']), 'config':os.environ['config']}

    @app.route('/ready')
    def ready():

        global concurrent_requests_value
        global last_lock_time
        global max_thread_time
        from flask import jsonify
        bOK = True
        if bEnvReady:
            if concurrent_requests_value>0:
                bOK = False
        else:
            prRed("Service yet not initialized")
            bOK = False
        if bOK:
            return jsonify(success=True, concurrent_requests_value=concurrent_requests_value, lock_time=time.time()-last_lock_time), 200
        else:
            if time.time() - last_lock_time > 400:
                prRed("1 Service blocked for {} min.".format((time.time() - last_lock_time) / 60.))
            if (time.time() - last_lock_time) > max_thread_time:
                while (concurrent_requests_value > 0):
                    release_lock("Force release lock")
            return jsonify(success=False, concurrent_requests_value=concurrent_requests_value, lock_time=time.time()-last_lock_time), 300

    @app.route('/layer_graph_data')
    def layer_graph():
        if acquire_lock("creating_layer_wise_graph"):
            try:
                jsonresponse = dict()
                jsonresponse['edges'] = env.connectivity_lut

                prunable_attr = env.df.to_dict()
                prunable_df = pd.DataFrame.from_dict(prunable_attr)
                node2cluster_id = prunable_df[['node_name', 'cluster_id']].set_index('node_name').to_dict()

                node_type = env.node_type_lut

                optype_encoder = OneHotEncoder()
                cluster_id_encoder = OneHotEncoder()

                one_hot_encoded_cluster_id = cluster_id_encoder.fit_transform(
                    np.array(list(node2cluster_id['cluster_id'].values())).reshape(-1, 1)).toarray()

                one_hot_encoded_cluster_id_map = dict(
                    zip(node2cluster_id['cluster_id'].keys(), one_hot_encoded_cluster_id))

                cluster_id_df = pd.Series(data=one_hot_encoded_cluster_id_map.values(),
                                          index=one_hot_encoded_cluster_id_map.keys(), name="cluster_id").to_frame()

                one_hot_encoded_optype = optype_encoder.fit_transform(
                    np.array(list(node_type.values())).reshape(-1, 1)).toarray()

                one_hot_encoded_optype_map = dict(zip(node_type.keys(), one_hot_encoded_optype))

                optype_df = pd.Series(data=one_hot_encoded_optype_map.values(), index=one_hot_encoded_optype_map.keys(),
                                      name="optype").to_frame()

                optype_df = optype_df.reindex(natsorted(optype_df.index))

                prunable_features = prunable_df.drop(columns=['module_scope', 'cluster_id']).set_index('node_name')

                features_per_node = pd.concat([optype_df, prunable_features], axis=1).fillna(0)

                features_per_node = pd.concat([features_per_node, cluster_id_df], axis=1)

                features_per_node['cluster_id'] = \
                    [np.zeros_like(one_hot_encoded_cluster_id[0]) if isinstance(val, float) else val for val in
                     features_per_node['cluster_id'].to_list()]

                for ii, item in enumerate(features_per_node['cluster_id']):
                    features_per_node['cluster_id'][ii] = features_per_node['cluster_id'][ii].tolist()
                    features_per_node['optype'][ii] = features_per_node['optype'][ii].tolist()

                feature_dict = features_per_node.to_dict()

                jsonresponse['node_features'] = feature_dict
                
                jsonresponse['action_space'] = {'continuous': 1}

                jsonresponse['rc'] = 1

                prRed("Sending connectivity and nodes features...")

            finally:
                gc.collect()
                release_lock("creating_layer_wise_graph")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse


    @app.route('/get_model_graph_viz')
    def get_model_graph_viz():
        graph_imgpth = os.path.join(env.nncf_cfg['log_dir'],'prune_env.png')
        try:
            return send_file(graph_imgpth, mimetype='image/png')
        except FileNotFoundError:
            if bEnvReady is True:
                return {'method':'get_model_graph_viz', 'msg':"Env ready, model graph image not found"}
            return {'method': 'get_model_graph_viz', 'msg': "Env not ready, model graph image not found"}

    @app.route('/get_node2optype_map')
    def get_node2optype_map():
        if acquire_lock("get_node2optype_map"):
            try:
                prRed("Sending dictionary of nodetype per nodes...")
                jsonresponse= env.node_type_lut
            finally:
                gc.collect()
                release_lock("get_node2optype_map")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse
    
    @app.route('/get_connectivity_map')
    def get_connectivity_map():
        if acquire_lock("get_node_connectivity_map"):
            try:
                prRed("Sending connectivity per source nodes...")
                jsonresponse= env.connectivity_lut
            finally:
                gc.collect()
                release_lock("get_connectivity_map")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse

    @app.route('/get_prunable_attr')
    def get_prunable_attr():
        if acquire_lock("get_prunable_attr"):
            try:
                prRed("Sending attributes of prunable nodes...")
                jsonresponse= env.df.to_dict()
            finally:
                gc.collect()
                release_lock("get_prunable_attr")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse

    @app.route('/sample_eval_req')
    def sample_eval_req():
        if acquire_lock("sample_eval_req"):
            try:
                prRed("Sending attributes of prunable nodes...")
                jsonresponse= env.groupwise_pruning_rate
            finally:
                gc.collect()
                release_lock("sample_eval_req")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse


    @app.route('/evaluate', methods=['POST'])
    def evaluate():
        #Code snippet for debugging purposes only
        start_time = time.time()
        #=================
        #Validate format of incoming request
        # =================
        try:
            content = request.get_json()
            pruning_rate_cfg = {int(gid): round(pr, 4) for gid, pr in content.items()}
        except:
            prRed("Exception in parsing http request")
            return {'rc': -3, 'msg': 'Exception in parsing http request'}
        
        # =================
        # For debugging purposes return a random response immediately
        # =================
        if 'debug' in content:
            if content['debug']:
                return dummy_response()
        
        if env.nncf_cfg.get('eval_cache', True) is True:
            cache_dir = os.path.join(env.nncf_cfg['log_dir'], 'paas_eval_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            input_cfg_hash = hashlib.md5(json.dumps(pruning_rate_cfg).encode('utf-8')).hexdigest()
            input_cfg_hash_filename = os.path.join(cache_dir, input_cfg_hash+".json")
            
            if os.path.exists(input_cfg_hash_filename):
                prRed("Recycling: {}".format(pruning_rate_cfg))
                prCyan("Recycling results in "+input_cfg_hash_filename)
                with open(input_cfg_hash_filename, 'r') as f:
                    jsonresponse = json.load(f)
                    jsonresponse['cached_eval'] = True
                return jsonresponse

        if acquire_lock("evaluate"):
            try:
                prRed("Evaluating: {}".format(pruning_rate_cfg))
                retval = env.evaluate_valset(pruning_rate_cfg)
                end_time = time.time()

                jsonresponse = dict()
                jsonresponse['rc'] = 0
                jsonresponse['msg'] = 'Prune and inference completed'
                jsonresponse['meta_data'] = {
                    'task_metric': retval,
                    'original_flops': env.original_flops,
                    'remaining_flops': env.remaining_flops,
                    'flop_ratio': env.flop_ratio,
                    'size_ratio': 1-env.effective_pruning_rate, # do note that only conv layers are considered
                    'effective_pruning_rate': env.effective_pruning_rate,
                    'groupwise_pruning_rate': env.groupwise_pruning_rate,
                    'input_pruning_rate': pruning_rate_cfg,
                    'layerwise_stats': env.layerwise_stats}

                jsonresponse['processing_time'] = str(end_time - start_time)

                if env.nncf_cfg.get('eval_cache', True) is True:                  
                    evaluated_cfg_hash = hashlib.md5(json.dumps(env.groupwise_pruning_rate).encode('utf-8')).hexdigest()
                    evaluated_cfg_hash_filename = os.path.join(cache_dir, evaluated_cfg_hash+".json")

                    prCyan("Writing input cfg to " + input_cfg_hash_filename)
                    with open(input_cfg_hash_filename, 'w') as f:
                        json.dump(jsonresponse, f, indent=4)
                    prCyan("Writing evaluated cfg to " + evaluated_cfg_hash_filename)
                    with open(evaluated_cfg_hash_filename, 'w') as f:
                        json.dump(jsonresponse, f, indent=4)

                # restoration of dense state dict must be after stats collection
                # as statistics are only extracted upon collection
                env.restore_dense_model()
            finally:
                gc.collect()
                release_lock("evaluate")
                jsonresponse['cached_eval'] = False
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse

    return app