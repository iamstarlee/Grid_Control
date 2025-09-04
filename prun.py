import math
from typing import List, Dict, Optional
import torch
from torch import nn

class StochasticMagnitudePruner:
    def __init__(
        self,
        model: nn.Module,           # 要剪枝的神经网络模型
        drop_ratio: float,          # 每次剪枝步骤尝试丢弃的参数比例（占总参数的分数）
        min_density: float,         # 剪枝后确保非零参数的最小比例（占总参数的分数）确保剪枝后非零参数不低于总参数量的%n
        exclude_biases: bool = False, # 是否排除偏置参数不进行剪枝
        normalize: str = "global",  # 权重归一化方式："global"全局归一化 或 "per_tensor"按张量归一化
        device: Optional[torch.device] = None, 
        seed: Optional[int] = None, 
    ):

        assert 0.0 <= drop_ratio <= 1.0
        assert 0.0 <= min_density <= 1.0
        assert normalize in ("global", "per_tensor")

        self.model = model
        self.drop_ratio = float(drop_ratio)
        self.min_density = float(min_density)
        self.exclude_biases = exclude_biases
        self.normalize = normalize
        self.device = device or next(model.parameters()).device

        if seed is not None:
            torch.manual_seed(seed)

        self.param_infos: List[Dict] = []
        for name, p in model.named_parameters():
            # 跳过不需要梯度的参数
            if not p.requires_grad:
                continue
            # 如果设置排除偏置且当前参数是偏置（1维），则跳过
            if self.exclude_biases and p.ndim == 1:
                continue
            # 记录参数信息：名称、参数对象、掩码
            self.param_infos.append({
                "name": name,
                "param": p,
                "mask": torch.ones_like(p, device=p.device, dtype=p.dtype),  # 初始化为全1的浮点掩码
            })

        # 为每个参数注册梯度钩子，确保被剪枝的权重在反向传播时梯度保持为零
        for info in self.param_infos:
            mask = info["mask"]
            p = info["param"]
            p.register_hook(lambda grad, m=mask: grad * m)

        self._total = sum(info["mask"].numel() for info in self.param_infos)

    def total_params(self) -> int:
        return self._total

    def count_nonzero(self) -> int:
        with torch.no_grad():
            return int(sum(info["mask"].gt(0).sum().item() for info in self.param_infos))

    def density(self) -> float:
        return self.count_nonzero() / (self._total + 1e-12)

    @torch.no_grad() 
    def apply_mask(self):
        for info in self.param_infos:
            info["param"].data.mul_(info["mask"])

    @torch.no_grad()
    def prune_and_maybe_regrow(self):
        """
        执行一次完整的剪枝
        """
        # 边界情况处理：如果没有可剪枝参数，直接返回
        if self._total == 0:
            return

        self._stochastic_prune()

        self.apply_mask()

        # 检查是否需要重新生长权重
        current_nz = self.count_nonzero()  
        min_required = int(math.ceil(self.min_density * self._total))  
        need = max(0, min_required - current_nz) 
        if need > 0:
            self._regrow_random(need)  # 随机重新生长指定数量的参数
            self.apply_mask() 

    @torch.no_grad()
    def _stochastic_prune(self):
        # 如果使用全局归一化，收集全局最小值和最大值
        global_min = None
        global_max = None

        if self.normalize == "global":
            mins = []
            maxs = []
            for info in self.param_infos:
                p = info["param"].data
                m = info["mask"].gt(0)  # 获取活跃（非零）位置的掩码
                if m.any():  # 如果存在活跃位置
                    vals = p.abs()[m]  
                    mins.append(vals.min())  
                    maxs.append(vals.max())  
            if len(maxs) == 0:
                return  
            global_min = torch.stack(mins).min()  
            global_max = torch.stack(maxs).max()  
        # 计算目标丢弃数量
        target_drops = int(round(self.drop_ratio * self._total))

        sum_pdrop = 0.0
        per_tensor_stats = []  # 存储统计信息以避免第二遍遍历时重复计算

        for info in self.param_infos:
            p = info["param"].data.view(-1)  
            m = info["mask"].view(-1).gt(0)  
            active = p[m].abs()  
            if active.numel() == 0:
                per_tensor_stats.append((None, None, None, m))
                continue

            # 根据归一化方式确定最小值和最大值
            if self.normalize == "per_tensor":
                mn = active.min()  
                mx = active.max()  
            else:
                mn = global_min  
                mx = global_max  

            denom = (mx - mn).clamp_min(1e-12)  # 避免除零
            norm = ((active - mn) / denom).clamp_(0, 1)
            pdrop = 1.0 - norm
            sum_pdrop += float(pdrop.sum().item())
            per_tensor_stats.append((mn, denom, pdrop, m))  # 存储统计信息

        # 如果没有可丢弃的内容或目标丢弃数量为零，直接返回
        if sum_pdrop <= 0.0 or target_drops <= 0:
            return

        # 缩放概率
        alpha = min(1.0, target_drops / (sum_pdrop + 1e-12))

        for info, stats in zip(self.param_infos, per_tensor_stats):
            mn, denom, pdrop, m = stats
            if pdrop is None:
                continue
            p_scaled = (alpha * pdrop).clamp_(0, 1)
            drops = torch.rand_like(p_scaled, device=p_scaled.device) < p_scaled

            mask_flat = info["mask"].view(-1)
            active_idx = m.nonzero(as_tuple=False).squeeze(1)  # 获取活跃位置的索引
            drop_idx = active_idx[drops]
            mask_flat[drop_idx] = 0.0

    @torch.no_grad()
    def _regrow_random(self, k: int):
        """
        生长
        """
        if k <= 0:
            return

        # 统计每个张量的零值数量
        zero_counts = []
        for info in self.param_infos:
            zero_counts.append(int(info["mask"].numel() - info["mask"].gt(0).sum().item()))
        total_zeros = sum(zero_counts)
        if total_zeros == 0:
            return

        alloc = []
        remaining = k
        for zc in zero_counts[:-1]:
            ki = int(round((zc / max(1, total_zeros)) * k))
            ki = min(ki, remaining)
            alloc.append(ki)
            remaining -= ki
        alloc.append(remaining)
        for info, ki in zip(self.param_infos, alloc):
            if ki <= 0:
                continue
            mask_flat = info["mask"].view(-1)
            p_flat = info["param"].data.view(-1)
            zero_idx = (mask_flat <= 0).nonzero(as_tuple=False).squeeze(1)
            if zero_idx.numel() == 0:
                continue
            if ki > zero_idx.numel():
                ki = zero_idx.numel()
            perm = torch.randperm(zero_idx.numel(), device=zero_idx.device)
            regrow_idx = zero_idx[perm[:ki]]
            mask_flat[regrow_idx] = 1.0
            active_vals = p_flat[(mask_flat > 0).nonzero(as_tuple=False).squeeze(1)]
            if active_vals.numel() > 0:
                std = float(active_vals.std().item())
                if not math.isfinite(std) or std == 0.0:
                    std = 0.02
            else:
                std = 0.02
            new_vals = torch.randn(ki, device=p_flat.device, dtype=p_flat.dtype) * std
            p_flat[regrow_idx] = new_vals

    def state_dict(self):
        return {
            "drop_ratio": self.drop_ratio,
            "min_density": self.min_density,
            "exclude_biases": self.exclude_biases,
            "normalize": self.normalize,
            "masks": {info["name"]: info["mask"].detach().cpu() for info in self.param_infos},
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        self.drop_ratio = state["drop_ratio"]
        self.min_density = state["min_density"]
        self.exclude_biases = state["exclude_biases"]
        self.normalize = state["normalize"]
        name_to_info = {info["name"]: info for info in self.param_infos}
        for name, mask in state["masks"].items():
            if name in name_to_info:
                info = name_to_info[name]
                info["mask"].copy_(mask.to(info["mask"].device, dtype=info["mask"].dtype))
        self.apply_mask()
