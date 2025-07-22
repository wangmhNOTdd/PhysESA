"""
阶段一：简化的PhysESA模型实现
基于ESA架构，只使用MAB层进行局部相互作用学习
优化版本：使用Flash Attention提升内存效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from typing import Dict, Optional, Tuple
import math

# 尝试导入Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("[信息] Flash Attention 可用，将使用优化的注意力实现")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("[警告] Flash Attention 未安装，将使用标准注意力实现")

class MaskedSelfAttention(nn.Module):
    """掩码自注意力模块 - Flash Attention优化版本"""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        使用Flash Attention的前向传播（如果可用且条件满足）
        
        Args:
            x: [batch_size, seq_len, embed_dim] 或 [seq_len, embed_dim]
            mask: [batch_size, seq_len, seq_len] 注意力掩码
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加batch维度
            
        batch_size, seq_len, embed_dim = x.shape
        
        # 计算Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 根据条件选择注意力实现
        use_flash = (FLASH_ATTN_AVAILABLE and 
                    x.device.type == 'cuda' and 
                    x.dtype in [torch.float16, torch.bfloat16] and
                    seq_len <= 8192 and  # Flash Attention 序列长度限制
                    mask is None)  # 目前跳过掩码以使用Flash Attention
        
        if use_flash:
            # 使用Flash Attention
            out = flash_attn_func(
                q, k, v, 
                dropout_p=self.dropout_p if self.training else 0.0
            )
        else:
            # 使用标准注意力实现
            out = self._standard_attention(q, k, v, mask)
        
        # 重塑输出并应用输出投影
        out = out.reshape(batch_size, seq_len, embed_dim)  # 使用reshape而不是view
        out = self.out_proj(out)
        
        if batch_size == 1:
            out = out.squeeze(0)  # 移除batch维度
            
        return out
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """标准注意力实现作为回退，基于ESA原版实现优化"""
        batch_size, seq_len = q.shape[:2]
        
        # 转换为标准的多头注意力格式: [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 使用PyTorch的scaled_dot_product_attention，这是内存优化的
        if mask is not None:
            # ESA风格的掩码处理
            if mask.dim() == 2:
                # 扩展掩码到正确的形状 [batch_size, num_heads, seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = mask.expand(batch_size, self.num_heads, -1, -1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # 使用PyTorch的内存优化注意力
            try:
                # 使用EFFICIENT_ATTENTION后端，这样可以自动选择最优的实现
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q, k, v, 
                        attn_mask=mask, 
                        dropout_p=self.dropout_p if self.training else 0.0, 
                        is_causal=False
                    )
            except ImportError:
                # 如果没有新版PyTorch，使用手动实现
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_weights = self.dropout(attn_weights)
                out = torch.matmul(attn_weights, v)
        else:
            # 没有掩码的情况，使用标准的scaled_dot_product_attention
            try:
                out = F.scaled_dot_product_attention(
                    q, k, v, 
                    dropout_p=self.dropout_p if self.training else 0.0
                )
            except AttributeError:
                # 如果PyTorch版本太老
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_weights = self.dropout(attn_weights)
                out = torch.matmul(attn_weights, v)
        
        # 转换回: [batch_size, seq_len, num_heads, head_dim]
        out = out.transpose(1, 2)
        
        return out

class MABLayer(nn.Module):
    """掩码自注意力块 (MAB)"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 掩码多头注意力
        self.attn = MaskedSelfAttention(embed_dim, num_heads, dropout)
        
        # 前馈网络
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 第一个残差连接：注意力
        x = x + self.attn(self.norm1(x), mask)
        
        # 第二个残差连接：前馈网络
        x = x + self.mlp(self.norm2(x))
        
        return x

class PoolingByMultiHeadAttention(nn.Module):
    """多头注意力池化模块 (PMA)"""
    
    def __init__(
        self,
        embed_dim: int,
        num_seeds: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_seeds = num_seeds
        
        # 可学习的种子向量
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, embed_dim))
        
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim] 或 [seq_len, embed_dim]
        Returns:
            [batch_size, num_seeds, embed_dim] 或 [num_seeds, embed_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加batch维度
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = x.shape[0]
        
        # 扩展种子向量到batch维度
        seeds = self.seed_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 交叉注意力：seeds作为query，x作为key和value
        pooled, _ = self.cross_attn(
            query=seeds,
            key=x,
            value=x,
            key_padding_mask=mask
        )
        
        pooled = self.norm(pooled)
        
        if squeeze_output:
            pooled = pooled.squeeze(0)
            
        return pooled

class SimplePhysESA(nn.Module):
    """
    简化的PhysESA模型 - 阶段一版本
    只使用MAB层，不使用SAB层
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_mab_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        cutoff_radius: float = 5.0
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_mab_layers = num_mab_layers
        self.cutoff_radius = cutoff_radius
        
        # 输入投影层：将节点和边特征投影到统一维度
        edge_input_dim = 2 * node_dim + edge_dim  # concat(n_i, n_j, e_ij)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)
        
        # MAB层堆栈
        self.mab_layers = nn.ModuleList([
            MABLayer(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_mab_layers)
        ])
        
        # 池化层
        self.pooling = PoolingByMultiHeadAttention(
            embed_dim=hidden_dim,
            num_seeds=1,  # 单一图表示
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def create_edge_adjacency_mask(
        self, 
        edge_index: torch.Tensor, 
        num_edges: int
    ) -> torch.Tensor:
        """
        创建边邻接掩码：基于ESA原版实现，内存优化版本
        """
        device = edge_index.device
        
        # 获取源节点和目标节点
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # 使用ESA的高效实现：直接创建扩展张量
        expanded_source_nodes = source_nodes.unsqueeze(1).expand(-1, num_edges)
        expanded_target_nodes = target_nodes.unsqueeze(1).expand(-1, num_edges)
        
        # 计算边的邻接关系（一次性计算所有条件）
        source_adjacency = expanded_source_nodes == expanded_source_nodes.t()
        target_adjacency = expanded_target_nodes == expanded_target_nodes.t()
        cross_adjacency = (expanded_source_nodes == expanded_target_nodes.t()) | \
                         (expanded_target_nodes == expanded_source_nodes.t())
        
        # 合并所有邻接关系
        adjacency_mask = source_adjacency | target_adjacency | cross_adjacency
        
        # ESA风格：使用0表示False（对角线设为0表示不自注意）
        adjacency_mask.fill_diagonal_(0)
        
        # 转换为布尔类型（与ESA保持一致）
        adjacency_mask = adjacency_mask.bool()
        
        return adjacency_mask
    
    def prepare_edge_features(self, data: Data) -> torch.Tensor:
        """
        准备边特征：concat(n_i, n_j, e_ij)
        """
        edge_index = data.edge_index
        node_features = data.x
        edge_features = data.edge_attr
        
        # 获取源节点和目标节点的特征
        src_features = node_features[edge_index[0]]  # [num_edges, node_dim]
        dst_features = node_features[edge_index[1]]  # [num_edges, node_dim]
        
        # 拼接：[src_features, dst_features, edge_features]
        edge_input = torch.cat([src_features, dst_features, edge_features], dim=-1)
        
        return edge_input
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyTorch Geometric Data对象
            
        Returns:
            预测的结合亲和力 [batch_size, 1]
        """
        # 准备边输入特征
        edge_input = self.prepare_edge_features(data)  # [num_edges, edge_input_dim]
        
        # 投影到隐藏维度
        edge_repr = self.edge_proj(edge_input)  # [num_edges, hidden_dim]
        
        # 创建边邻接掩码
        num_edges = edge_input.shape[0]
        edge_mask = self.create_edge_adjacency_mask(data.edge_index, num_edges)
        
        # 通过MAB层
        for mab_layer in self.mab_layers:
            edge_repr = mab_layer(edge_repr, mask=edge_mask)
        
        # 池化到图级别表示
        graph_repr = self.pooling(edge_repr)  # [1, hidden_dim]
        
        # 预测结合亲和力
        prediction = self.predictor(graph_repr.squeeze(0))  # [1]
        
        return prediction

def create_model(node_dim: int, edge_dim: int, **kwargs) -> SimplePhysESA:
    """创建模型实例的工厂函数"""
    return SimplePhysESA(
        node_dim=node_dim,
        edge_dim=edge_dim,
        **kwargs
    )

if __name__ == "__main__":
    # 测试模型
    from molecular_graph import MolecularGraphBuilder
    import os
    
    # 创建图构建器
    builder = MolecularGraphBuilder(cutoff_radius=5.0, num_gaussians=16)
    dims = builder.get_feature_dimensions()
    
    # 创建模型
    model = create_model(
        node_dim=dims['node_dim'],
        edge_dim=dims['edge_dim'],
        hidden_dim=128,
        num_mab_layers=4
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播（如果有数据）
    complex_id = "1a28"
    pdb_file = f"./datasets/pdbbind/pdb_files/{complex_id}/{complex_id}.pdb"
    sdf_file = f"./datasets/pdbbind/pdb_files/{complex_id}/{complex_id}_ligand.sdf"
    
    if os.path.exists(pdb_file) and os.path.exists(sdf_file):
        try:
            data = builder.build_graph(complex_id, pdb_file, sdf_file)
            print(f"测试数据: 原子数={data.x.shape[0]}, 边数={data.edge_index.shape[1]}")
            
            model.eval()
            with torch.no_grad():
                prediction = model(data)
                print(f"预测结果: {prediction.item():.4f}")
                
        except Exception as e:
            print(f"测试时出错: {e}")
    else:
        print("测试数据不存在，跳过前向传播测试")

# 为了向后兼容，提供别名
PhysESA = SimplePhysESA
