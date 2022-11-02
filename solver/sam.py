import math
import torch
import torch.optim

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class SAM(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, rho) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        self.rho = rho
        super(SAM, self).__init__(params, dict(rho=rho))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
    
    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
        }
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."
        
        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


@OPTIMIZER_REGISTRY.register()
class SSAMF(SAM):
    @configurable()
    def __init__(self, params, base_optimizer, rho, sparsity, num_samples, update_freq) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0.0 <= sparsity <= 1.0, f"sparsity should between 0 and 1: {sparsity}"
        assert 1.0 <= num_samples, f"num_samples should be greater than 1: {num_samples}"
        assert 1.0 <= update_freq , f"update_freq should be greater than 1: {update_freq}"
        self.rho = rho
        self.sparsity = sparsity
        self.num_samples = num_samples
        self.update_freq = update_freq
        super(SSAMF, self).__init__(params, base_optimizer, rho)

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["sparsity"] = sparsity
            group["num_samples"] = num_samples
            group["update_freq"] = update_freq

        self.init_mask()

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
            "sparsity": args.sparsity,
            "num_samples": args.num_samples,
            "update_freq": args.update_freq,
        }
    
    @torch.no_grad()
    def init_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = torch.zeros_like(p, requires_grad=False).to(p)

    @torch.no_grad()
    def update_mask(self, model, train_data, **kwargs):
        fisher_value_dict = {}
        fisher_mask_dict = {}
        for group in self.param_groups:
            for p in group['params']:
                fisher_value_dict[id(p)] = torch.zeros_like(p, requires_grad=False).to(p)
                fisher_mask_dict[id(p)] = torch.zeros_like(p, requires_grad=False).to(p)

        criterion = torch.nn.CrossEntropyLoss()
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=1,
            num_workers=4,
            shuffle=True,
        )
        # cal fisher value
        with torch.enable_grad():
            for idx, (image, label) in enumerate(train_dataloader):
                if idx >= self.num_samples: break
                if idx % (self.num_samples // 10) == 0: print('Updating Mask: [{}/{}]..'.format(idx, self.num_samples))
                image, label = image.cuda(), label.cuda()
                
                output = model(image)
                loss = criterion(output, label)
                loss.backward()

                for group in self.param_groups:
                    for p in group["params"]:
                        fisher_value_dict[id(p)] += torch.square(p.grad).data
                model.zero_grad()
        
        # topk fisher value 
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict.values()])
        
        keep_num = int(len(fisher_value_list) * (1 - self.sparsity))
        _value, _index = torch.topk(fisher_value_list, keep_num)
        
        mask_list = torch.zeros_like(fisher_value_list)
        mask_list.scatter_(0, _index, torch.ones_like(_value))

        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start_index == len(mask_list)
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None, model=None, epoch=None, batch_idx=None, train_data=None, logger=None, **kwargs):
        super().step(closure, **kwargs)
        assert model is not None
        assert train_data is not None
        assert epoch is not None
        assert batch_idx is not None
        assert logger is not None
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            logger.log('Update Mask!')
            self.update_mask(model, train_data)
            logger.log('Mask Lived Weight: {:.4f}'.format(self.mask_info()))
            
    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item() 
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num

@OPTIMIZER_REGISTRY.register()
class SSAMD(SAM):
    @configurable()
    def __init__(self, params, base_optimizer, 
        rho, sparsity, drop_rate, drop_strategy, growth_strategy, update_freq, T_start, T_end) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0.0 <= sparsity <= 1.0, f"sparsity should between 0 and 1: {sparsity}"
        assert 0.0 <= drop_rate <= 1.0, f"drop_rate should between 0 and 1: {drop_rate}"
        assert 1.0 <= update_freq , f"update_freq should be greater than 1: {update_freq}"
        self.rho = rho
        self.sparsity = sparsity
        self.drop_rate = drop_rate
        self.drop_strategy = drop_strategy
        self.growth_strategy = growth_strategy
        self.update_freq = update_freq
        self.T_start = T_start
        self.T_end = T_end
        super(SSAMD, self).__init__(params, base_optimizer, rho)

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["sparsity"] = sparsity
            group["drop_rate"] = drop_rate
            group["drop_strategy"] = drop_strategy
            group["growth_strategy"] = growth_strategy
            group["update_freq"] = update_freq
            group["T_end"] = T_end
        self.init_mask()

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
            "sparsity": args.sparsity,
            "drop_rate": args.drop_rate,
            "drop_strategy": args.drop_strategy,
            "growth_strategy": args.growth_strategy,
            "update_freq": args.update_freq,
            "T_end": args.epochs,
            "T_start": 0,
        }
    
    @torch.no_grad()
    def init_mask(self):
        random_scores = []
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]['score'] = torch.rand(size=p.shape).cpu().data
                random_scores.append(self.state[p]['score'])
        random_scores = torch.cat([torch.flatten(x) for x in random_scores])
        live_num = len(random_scores) - math.ceil(len(random_scores) *self.sparsity)
        _value, _index = torch.topk(random_scores, live_num)

        mask_list = torch.zeros_like(random_scores)
        mask_list.scatter_(0, _index, torch.ones_like(_value))
        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                self.state[p]['mask'] = self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                del self.state[p]['score']
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start_index == len(mask_list)
        
    @torch.no_grad()
    def DeathRate_Scheduler(self, epoch):
        dr = (self.drop_rate) * (1 + math.cos(math.pi * (float(epoch - self.T_start) / (self.T_end - self.T_start)))) / 2 
        return dr           

    @torch.no_grad()
    def update_mask(self, epoch, **kwargs):
        death_scores = []
        growth_scores =[]
        for group in self.param_groups:
            for p in group['params']:
                death_score = self.get_score(p, self.drop_strategy)
                death_scores.append((death_score + 1e-7) * self.state[p]['mask'].cpu().data)

                growth_score = self.get_score(p, self.growth_strategy)
                growth_scores.append((growth_score + 1e-7) * (1 - self.state[p]['mask'].cpu().data))
        '''
            Death 
        '''
        death_scores = torch.cat([torch.flatten(x) for x in death_scores])
        death_rate = self.DeathRate_Scheduler(epoch=epoch)
        death_num = int(min((len(death_scores) - len(death_scores) * self.sparsity)* death_rate, len(death_scores) * self.sparsity))
        d_value, d_index = torch.topk(death_scores, int((len(death_scores) - len(death_scores) * self.sparsity) * (1 - death_rate)))

        death_mask_list = torch.zeros_like(death_scores)
        death_mask_list.scatter_(0, d_index, torch.ones_like(d_value))
        '''
            Growth
        '''
        growth_scores = torch.cat([torch.flatten(x) for x in growth_scores])
        growth_num = death_num
        g_value, g_index = torch.topk(growth_scores, growth_num)
        
        growth_mask_list = torch.zeros_like(growth_scores)
        growth_mask_list.scatter_(0, g_index, torch.ones_like(g_value))

        '''
            Mask
        '''
        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                death_mask = death_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                growth_mask = growth_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                
                self.state[p]['mask'] = death_mask + growth_mask
                self.state[p]['mask'] = self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
                
                
                    
        assert start_index == len(death_mask_list)

    def get_score(self, p, score_model=None):
        if score_model == 'weight':
            return torch.abs(p.clone()).cpu().data
        elif score_model == 'gradient':
            return torch.abs(p.grad.clone()).cpu().data
        elif score_model == 'random':
            return torch.rand(size=p.shape).cpu().data
        else:
            raise KeyError    
  
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None, epoch=None, batch_idx=None, logger=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."
        assert epoch is not None
        assert batch_idx is not None
        assert logger is not None

        self.first_step()
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            logger.log('Update Mask!')
            self.update_mask(epoch)
            logger.log('Mask Lived Weight: {:.4f}'.format(self.mask_info()))
        self.zero_grad()
        with torch.enable_grad():
            closure()
        self.second_step()

    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item() 
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num
