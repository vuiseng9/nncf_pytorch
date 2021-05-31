import os,sys
from flask import Flask, g
from flask import request
import json, random
import numpy as np
import gc
from multiprocessing import Lock, Semaphore
from multiprocessing.sharedctypes import Value
lock = Semaphore(1)
mutex= Lock()

#sys.path.append('/workspace/nncf_pytorch')
sys.path.append(os.path.dirname(__file__))

from examples.common.sample_config import SampleConfig, create_sample_config
from nncf import NNCFConfig
evaluated_mappings = dict()
import time
import queue
evaluated_mappings_history = queue.Queue()

from imgnet_qenv_wrapper import main as imgnet_qenv
from sseg_qenv_wrapper import main as sseg_qenv
from objdet_qenv_wrapper import main as objdet_qenv

from copy import deepcopy
import logging, pandas
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)
def prRed(prt): print("\033[91m {}\033[00m".format(prt),flush=True)
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt),flush=True)
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt),flush=True)

import string

from collections import defaultdict, OrderedDict

def init_env():

    if os.environ['qenv_name'] == 'imgnet':
        _args = [
            '--gpu-id', '0', 
            '--workers', '8', 
            '--log-dir', '/tmp/qaas-imgnet-log/',  
            '--config', os.environ['config'],
            '--data',   os.environ['data']]
        env, config = imgnet_qenv(_args)

    elif os.environ['qenv_name'] == 'sseg':
        _args = [
            '--gpu-id', '0', 
            '--workers', '8', 
            '--log-dir', '/tmp/qaas-sseg-log/',
            '--dataset', 'camvid',
            '--config', os.environ['config'],
            '--data',   os.environ['data'],
            '--weight', os.environ['weight']]
        env, config = sseg_qenv(_args)

    elif os.environ['qenv_name'] == 'objdet':
        _args = [
            '--gpu-id', '0', 
            '--workers', '8', 
            '--log-dir', '/tmp/qaas-objdet-log/',  
            '--config', os.environ['config'],
            '--data',   os.environ['data'],
            '--weight', os.environ['weight']]
        env, config = objdet_qenv(_args)
    
    elif os.environ['qenv_name'] == 'dlrm':
        from dlrm_qenv_wrapper import main as dlrm_qenv
        _args = [
                "--arch-sparse-feature-size", "16",
                "--arch-mlp-bot", "13-512-256-64-16",
                "--arch-mlp-top", "512-256-1",
                "--data-generation", "dataset",
                "--data-set", "kaggle",
                "--raw-data-file", os.path.join(os.environ['data'], "train.txt"),
                "--processed-data-file", os.path.join(os.environ['data'], "kaggleAdDisplayChallenge_processed.npz"),
                "--loss-function", "bce",
                "--round-targets", "True",
                "--learning-rate", "0.1",
                "--mini-batch-size", "128",
                "--print-freq", "1",
                "--print-time",
                "--test-mini-batch-size", "16384",
                "--test-num-workers", "16",
                "--use-gpu",
                "--dataset-multiprocessing",
                "--test-freq", "1",
                "--load-model",  os.environ['weight'],
                "--inference-only",
                "--nncf_config", os.environ['config'],
                "--log-dir", "/tmp/qaas-dlrm-log/"]

        env, config = dlrm_qenv(_args)
    else:
        raise ValueError("Undefined environment variable qenv_name")
    
    return env

# def get_current_lock_time():
#     T = time.time()
#     return  int(time.time() - last_locked_time.value)
#
# def reset_current_lock_time(calling_method):
#     last_locked_time.value  = time.time()

def acquire_lock(calling_method):
    global conncurrent_requests_value
    global last_lock_time
    #if conncurrent_requests_value==0:
    try:
        if lock.acquire(block=False):
            last_lock_time = time.time()
            conncurrent_requests_value += 1
            prCyan("aquired_lock - " + calling_method+", conncurrent_requests_value="+str(conncurrent_requests_value))
            return True
        else:
            prRed("Blocked since: " + str(time.time()-last_lock_time))
            return  False
    except:
        prRed("BlockedY")
        pass

def release_lock(calling_method):
    global conncurrent_requests_value
    global last_lock_time
    lock.release()
    last_lock_time = time.time()
    conncurrent_requests_value -= 1
    prCyan("release_lock - " + calling_method+", conncurrent_requests="+str(conncurrent_requests_value))
    return  True

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    bEnvReady = False
    global conncurrent_requests_value
    global max_thread_time
    conncurrent_requests_value = 0
    acquire_lock("create app init")
    t1 = time.time()
    env = init_env()
    max_thread_time = int(5*(time.time()-t1))
    prRed("max_thread_time="+str(max_thread_time)+" sec.")
    release_lock("create app init")
    bEnvReady = True
    print("NNCF Quantization Environment initialized",env,flush=True)


    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/ready')
    def ready():

        global conncurrent_requests_value
        global last_lock_time
        global max_thread_time
        from flask import jsonify
        bOK = True
        if bEnvReady:
            if conncurrent_requests_value>0:
                bOK = False
        else:
            prRed("Service yet not initialized")
            bOK = False
        if bOK:
            return jsonify(success=True, conncurrent_requests_value=conncurrent_requests_value, lock_time=time.time()-last_lock_time), 200
        else:
            if time.time() - last_lock_time > 400:
                prRed("1 Service blocked for {} min.".format((time.time() - last_lock_time) / 60.))
            if (time.time() - last_lock_time) > max_thread_time:
                while (conncurrent_requests_value > 0):
                    release_lock("Force release lock")
            return jsonify(success=False, conncurrent_requests_value=conncurrent_requests_value, lock_time=time.time()-last_lock_time), 300

    @app.route('/release_lock')
    def force_release_lock():
        global conncurrent_requests_value
        from flask import jsonify
        while(conncurrent_requests_value>0):
            release_lock("Force release lock")
        return jsonify(conncurrent_requests_value=conncurrent_requests_value), 300

    @app.route('/hello')
    def hello():
        return 'Hello, World! Environment ready: '+str(bEnvReady)

    @app.route('/toggle_adaptbn')
    def toggle_adaptbn():
        env.enable_adaptbn = not env.enable_adaptbn
        return adaptbn_cfg()

    @app.route('/adaptbn_cfg')
    def adaptbn_cfg():
        if env.enable_adaptbn is True:
            if 'batchnorm_adaptation' not in env.qctrl.quantization_config['initializer']:
                env.qctrl.quantization_config['initializer']['batchnorm_adaptation'] = {'num_bn_adaptation_steps': 50}
            return {"enable_adaptbn" : env.enable_adaptbn, 
                    "n_batch": env.qctrl.quantization_config['initializer']['batchnorm_adaptation']['num_bn_adaptation_steps']}
        return {"enable_adaptbn" : env.enable_adaptbn}

    def extract_clipfactor_toggle_from_url(r: request):
        import urllib.parse as urlparse
        from urllib.parse import parse_qs
        parsed = urlparse.urlparse(request.url)
        try:
            bclipfactor = parse_qs(parsed.query)['clipfactor'][0]
            bclipfactor = int(bclipfactor) > 0
        except:
            bclipfactor = True
        print("bclipfactor=", bclipfactor)
        return bclipfactor

    def extract_batchnorm_adaption_from_url(r: request):
        import urllib.parse as urlparse
        from urllib.parse import parse_qs
        parsed = urlparse.urlparse(request.url)
        try:
            bnadapt = parse_qs(parsed.query)['bnadapt'][0]
            bnadapt = int(bnadapt)>0
        except:
            bnadapt = False
        return bnadapt

    def dummy_response():
        return 	{'rc': 0, 'done': 1, 'reward': random.uniform(0.0, 1.4), 'meta_data': {'acccuracy': random.uniform(0.0, 0.95), 'modelsize': random.uniform(0.0625, 0.25)}}

    @app.route('/evaluate_scope_overrides', methods=['POST'])
    def evaluate_scope_overrides():
        #Code snippet for debugging purposes only
        start_time = time.time()
        try:
            content = request.get_json()
        except:
            prRed("Exception in parsing http request")
            return {'rc': -3, 'msg': 'Exception in parsing http request'}

        precision_strategy_as_list = []
        for k in env.scope_override_key_to_master_df_map.keys():
            precision_strategy_as_list.append(content[k]['bits'])

        if acquire_lock("evaluate"):
            try:
                prRed("Evaluating: "+str(precision_strategy_as_list))
                obs, reward, done, info_set = \
                    env.evaluate_strategy(
                        precision_strategy_as_list,
                        clip_strategy_dict = None,
                        skip_wall=True,
                        eval_pipeline=False)
                results = {'reward': reward,'done':1, 'meta_data': info_set}
                end_time = time.time()
                jsonresponse= dict({'rc': 0, 'msg': 'Quantization and inference completed'}, **results)
                results['processing_time'] = str(end_time - start_time)
            finally:
                gc.collect()
                release_lock("evaluate")
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
        except:
            prRed("Exception in parsing http request")
            return {'rc': -3, 'msg': 'Exception in parsing http request'}

        # =================
        # For debugging purposes return a random response immediately
        # =================
        if 'debug' in content:
            if content['debug']:
                return dummy_response()

        # =================
        # Extract parameters from request
        # =================
        bnadapt = extract_batchnorm_adaption_from_url(request)


        import hashlib
        wl_model_name = env.config['model']
        wl_model_name = wl_model_name +"BatchNorm" if bnadapt else  wl_model_name +"NoBatchNorm"

        C = dict()
        for k,v in env.config.items():
            if any([q == k for q in ['tb','nncf_config','config','checkpoint_save_dir','log_dir','intermediate_checkpoints_path','episodic_nncfcfg']]):
                continue
            else:
                C[k] = v

        wl_config_hash = "nncfconfig_"+str(int(hashlib.md5(str(C).encode('utf-8')).hexdigest(), 16))

        rc = "{}"
        results = "{}"
        strategy_dict = content

        #print("======Display Request JSON=====")
        #print(strategy_dict,flush=True)
        #print("=======================")

        precision_strategy_dict = dict()
        clip_strategy_dict = dict()

        for TargetOp, action in strategy_dict.items():
            for k, v in env.master_df.q_nx_nodekey.items():
                if v==TargetOp:
                    if 'precision' in action:
                        precision_strategy_dict[k] = action['precision']
                    if 'clip_min_scaler' in action and 'clip_max_scaler' in action:
                        clip_strategy_dict[k] = dict()
                        clip_strategy_dict[k]['clip_min_scaler'] = np.array(action['clip_min_scaler'])
                        clip_strategy_dict[k]['clip_max_scaler'] = np.array(action['clip_max_scaler'])

        precision_strategy_as_list = [precision_strategy_dict[qlut_id] for qlut_id in env.master_df.index[env.master_df.is_pred | (not env.tie_quantizers)]]
        precision_strategy_str = "precisionstrat"+".".join([str(p) for p in precision_strategy_as_list])

        if len(clip_strategy_dict)< len(precision_strategy_dict):
            clip_strategy_dict = None
            clip_strategy_str = ""
        else:
            clip_strategy_dict = {qlut_id: clip_strategy_dict[qlut_id] for qlut_id in env.master_df.index}
            clip_strategy_str = "clipstrat"+".".join([f"({v['clip_min_scaler']},{v['clip_max_scaler']})" for _,v in clip_strategy_dict.items()])

        effective_mapping_string = "prec+clip-"+f'bnadapt-{bnadapt}'+precision_strategy_str+"-"+clip_strategy_str
        mapping_hash = "map_"+str(int(hashlib.md5(effective_mapping_string.encode('utf-8')).hexdigest(), 16))

        cache_dir = os.path.join("cached_mappings",wl_model_name,wl_config_hash)
        os.makedirs(cache_dir, exist_ok=True)
        
        ##Caching Section
        if env.config.get('qaas_cache', True) is True:
            configuration_filename = str(wl_config_hash)+".json"
            if configuration_filename not in os.listdir(os.path.join("cached_mappings",wl_model_name)):
                try:
                    import json
                    outfilename = os.path.join("cached_mappings",wl_model_name,configuration_filename)
                    with open(outfilename, "w") as f:
                        json.dump(C, f, indent=4, sort_keys=True)
                except:
                    pass




            CacheList = os.listdir(cache_dir)
            mappingcache = dict() #The cached could be sourced to a DB server. At the moment each entry is stored as file.
            try:
                if any([mapping_hash in c for c in CacheList]):
                    import ast
                    infilename = os.path.join(cache_dir, mapping_hash+".json")
                    prCyan("Recycling results in "+infilename)
                    with open(infilename, 'r') as f:
                        mappingcache[mapping_hash] = ast.literal_eval(f.readline())

            except:
                pass

            if mapping_hash in mappingcache:
                rc_dict = mappingcache[mapping_hash]
                rc_dict['processing_time'] = time.time() - t1
                return rc_dict, 200

        if acquire_lock("evaluate"):
            try:
                prRed("Evaluating: "+str(effective_mapping_string))
                env.enable_adaptbn = bnadapt
                adaptbn_cfg()
                obs, reward, done, info_set = \
                    env.evaluate_strategy(
                        precision_strategy_as_list,
                        clip_strategy_dict = clip_strategy_dict,
                        skip_wall=True)
                results = {'reward': reward,'done':1, 'meta_data': info_set}
                end_time = time.time()
                jsonresponse= dict({'rc': 0, 'msg': 'Quantization and inference completed'}, **results)
                outfilename = os.path.join(cache_dir, mapping_hash + ".json")
                prCyan("Mapping results saved to: "+outfilename)
                try:
                    with open(outfilename, 'w') as f:
                        f.write(str(jsonresponse))
                except:
                    pass
                results['processing_time'] = str(end_time - start_time)
            finally:
                gc.collect()
                release_lock("evaluate")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse

    @app.route('/model_initial_clip_threshold')
    def model_initial_clip_threshold():
        d = {}
        for v in env.master_df.index[env.master_df.is_pred | (not env.tie_quantizers)]:
            qmod = env.master_df.qmodule[v]
            
            d[str(v)] = {
                'qtype'      : qmod.__class__.__name__,
                'per_channel': qmod.per_channel,
                'clip_min'   : qmod.init_clip_threshold['min'].reshape(-1).tolist(),
                'clip_max'   : qmod.init_clip_threshold['max'].reshape(-1).tolist(),
            }
        return d

    @app.route('/lock_state')
    def getlockstate():
        if lock.acquire():
            lock.release()
            return {'me'
                    'thod': 'lock_state', 'rc': 0, 'msg': "Lock not locked"}
        else:
            return {'method': 'lock_state', 'rc': -1, 'msg': "Lock locked"}

    # @app.route('/release_lock')
    # def releaselock():
    #     try:
    #         reset_current_lock_time("REST /release_lock "+str(conncurrent_requests.value))
    #         lock.release()
    #         return {'method': 'releaselock', 'rc': 0, 'msg': "Lock release"}
    #     except:
    #         return {'method': 'releaselock', 'rc': -1, 'msg': "Lock already released"}


    @app.route('/ready_state')
    def readystate():
        config = os.environ['config']
        wl_name = config.split('/')[-1].split('.')[-2]

        if bEnvReady:
            return {'method':'ready_state', 'rc': 0, 'msg':"Environment {} initialzed".format(wl_name),'config':config}
        return {'method': 'ready_state', 'rc': 1, 'msg': "Environment {} not yet initialzed".format(wl_name),'config':config}

    def get_ops_onehotencoding(ops_set):
        global_ops_set = {'RELU','__add__','adaptive_avg_pool2d','asymmetric_quantize','batch_norm','bmm','cat','conv2d','dropout','embedding_bag','hardtanh','linear','max_pool2d','nncf_model_input','sigmoid','symmetric_quantize'}.union(ops_set)
        print("current ops set=",ops_set)
        lst  = lambda: [0]*len(global_ops_set)
        ops_onehotencoding = defaultdict(lst )
        for i, v in enumerate(global_ops_set):
            ops_onehotencoding[v] = [0] * len(global_ops_set)
            ops_onehotencoding[v][i] = 1
        return  ops_onehotencoding

    def get_op_type(op):
        g = env.qctrl._model.get_graph()
        nx_digraph = g._get_graph_for_structure_analysis()
        return g.get_nx_node_by_key(op)['op_exec_context'].operator_name

    @app.route('/get_dataframe')
    def df_send():
        if acquire_lock("get_dataframe"):
            bUseClipFactor = extract_clipfactor_toggle_from_url(request)
            op_2_feature = defaultdict(list)
            ops_set = set()
            nx_digraph = env.qctrl._model.get_graph()._get_graph_for_structure_analysis()
            feature_dim = 0

            for k, v in env.master_df.q_nx_nodekey.items():
                node_feature = env.master_df.loc[k, env.state_list]
                feature_vetor = [v1 for _,v1 in node_feature.items()]
                op_type = k.replace(']', '/').split('/')[-1].rstrip(string.digits).rstrip("_")
                op_type = get_op_type(v)
                ops_set.add(op_type)
                #ToDo: filter our prev action
                op_2_feature[v] = [feature_vetor,op_type]


            print("ops_set1=",ops_set)
            ops_set = set()
            edge_connectivity = defaultdict(set)
            for e in nx_digraph.edges:
                edge_connectivity[e[0]].add(e[1])
                for _e in e:
                    op_type = _e.replace(']', '/').split('/')[-1].rstrip(string.digits).rstrip("_")
                    op_type = get_op_type(_e)
                    ops_set.add(op_type)
                    if _e not in op_2_feature:
                        op_2_feature[_e] = [[0] * len(feature_vetor),op_type]
                    else:
                        op_2_feature[_e][1] = op_type
            print("ops_set2=", ops_set)
            ops_onehotencoding = get_ops_onehotencoding(ops_set)
            for _, op in op_2_feature.items():
                op_type = op[1]
                print(op_type,ops_onehotencoding[op_type])
            ops_dict = {}
            for op in op_2_feature:
                op_type = op.replace(']', '/').split('/')[-1].rstrip(string.digits).rstrip("_")
                op_type = get_op_type(op)
                ops_dict[op] = [list(edge_connectivity[op]),
                                ops_onehotencoding[op_type]+op_2_feature[op][0]]

            df = pandas.DataFrame.from_dict(ops_dict, orient='index')
            print('df', df)

            df.columns = ["dst_op", "features"]
            df = df.reindex(index=sorted(df.index))
            out = df.to_dict()

            if env.tie_quantizers == False:
                quantization_action_space = ([1, 2, 4, 8], 'discrete')
            else:
                quantization_action_space = ([2, 4, 8], 'discrete')

            out['action_space'] = {
                'action_type': {'precision': quantization_action_space}}

            global max_thread_time
            out['recommended_request_timeout'] = str(max_thread_time)+" sec."
            
            config = NNCFConfig.from_json(os.environ['config'])

            if bUseClipFactor:
                if "clip_scaler" in config:
                    if config['clip_scaler']:
                        clip_min_scaler_action_space = (['0.5', '0.7', '0.9', '1.0'], 'discrete')
                        clip_max_scaler_action_space = (['0.5', '0.7', '0.9', '1.0'], 'discrete')
                        out['action_space']['action_type']['clip_min_scaler'] = clip_min_scaler_action_space
                        out['action_space']['action_type']['clip_max_scaler'] = clip_max_scaler_action_space

            release_lock("get_normalized_state_dataframe")
            for k, v in out['features'].items():
                out['features'][k] = [float(a) for a in v]
            return {'normalized_state_df': out, 'rc': 0}
        else:
            return {'rc': -1, 'msg': 'Server busy'}

    return app
