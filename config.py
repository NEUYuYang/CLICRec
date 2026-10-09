"""CLICRec 模型与损失配置；默认值不代表论文最优实验配置。"""

from dataclasses import dataclass

@dataclass
class CLICRecConfig:
    num_users: int
    num_items: int
    embedding_size: int = 64
    short_seq_length: int = 50
    num_layers: int = 2
    num_heads: int = 2
    gat_num_heads: int = 2
    gat_hidden_dim: int = 64
    mlp_hidden_size: tuple = (128, 64)
    dropout_prob: float = 0.1
    gat_dropout: float = 0.1
    long_clusters: int = 20
    short_clusters: int = 20
    long_temperature: float = 0.2
    short_temperature: float = 0.2
    lambda_link: float = 0.1
    lambda_cl: float = 0.1
    lambda_consistency: float = 0.1
    lambda_l2: float = 0.0
    consistency_delta: float = 0.1
    consistency_margin: float = 0.0
    decorrelation_beta: float = 1.0
    time_epsilon: float = 1e-6
    kmeans_iterations: int = 50
    kmeans_restarts: int = 5
    kmeans_seed: int = 42

    def __post_init__(self):  # 在数据类自动初始化后检查配置值的有效性
        for name in ('num_users', 'num_items', 'embedding_size', 'short_seq_length',
                     'num_layers', 'num_heads', 'gat_num_heads', 'gat_hidden_dim',
                     'long_clusters', 'short_clusters', 'kmeans_iterations', 'kmeans_restarts'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if min(self.num_users, self.num_items) < 2:
            raise ValueError('ID counts must include padding and at least one real ID')
        if self.embedding_size % self.num_heads:
            raise ValueError('embedding_size must be divisible by num_heads')
        if not 0 <= self.consistency_delta < 1 or not 0 <= self.consistency_margin < 1:
            raise ValueError('delta and margin must be in [0,1)')
        if min(self.long_temperature, self.short_temperature, self.time_epsilon) <= 0:
            raise ValueError('temperatures and time_epsilon must be positive')
        if any(getattr(self, name) < 0 for name in
               ('lambda_link', 'lambda_cl', 'lambda_consistency', 'lambda_l2', 'decorrelation_beta')):
            raise ValueError('loss weights must be nonnegative')
        if not 0 <= self.dropout_prob < 1 or not 0 <= self.gat_dropout < 1:
            raise ValueError('dropout must be in [0,1)')