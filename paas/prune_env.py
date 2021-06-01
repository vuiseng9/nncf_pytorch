from nncf.pruning.filter_pruning.algo import FilterPruningController

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

    def evaluate_valset(self, pruning_rate_cfg):
        self.pruning_controller.set_pruning_rate(pruning_rate_cfg)
        return self.evaluator(self.pruned_model, self.val_loader)