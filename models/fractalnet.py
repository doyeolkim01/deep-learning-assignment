import torch
import torch.nn as nn
import torch.nn.functional as F

# (BN -> ReLU -> Conv) Block
class Block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    out = self.bn(x)
    out = self.relu(out)
    out = self.conv(out)
    return out


# FractalNode: recursive fractal structure
class FractalNode(nn.Module):
  # Fractal Recursion)
  # f{1}(x) = Block(x)
  # f{d+1}(x) = join( f{d}(f{d}(x)), f{1}(x) )

  # depth = num_columns - 1
  def __init__(self, in_channels, depth, drop_p=0.0):
    super().__init__()
    self.depth = depth
    self.drop_p = drop_p

    if depth == 0:
      # f{1}
      self.f1 = Block(in_channels, in_channels)
    else:
      # shallow path: f{1}
      self.shallow = Block(in_channels, in_channels)
      # deep path: f{d}(f{d}) (composite function, each uses different weights)
      self.deep1 = FractalNode(in_channels, depth - 1, drop_p=drop_p)
      self.deep2 = FractalNode(in_channels, depth - 1, drop_p=drop_p)
  
  def _join_list(self, xs):
    # eval mode or not using local drop-path
    if (not self.training) or self.drop_p <= 0.0:
      return torch.stack(xs, dim=0).mean(dim=0)

    # train mode: local drop-path (keep at least one branch)
    kept = []
    
    for x in xs:
      if torch.rand(1).item() >= self.drop_p:
        kept.append(x)

    # if all dropped, randomly keep one
    if len(kept) == 0:
      kept.append(xs[torch.randint(0, len(xs), (1,)).item()])
    
    if len(kept) == 1:
      return kept[0]
    
    return torch.stack(kept, dim=0).mean(dim=0)

  def forward_global(self, x, col_idx):
    if self.depth == 0:
        return self.f1(x)
    
    if col_idx == 0:
        return self.shallow(x)
    
    y = self.deep1.forward_global(x, col_idx - 1)
    y = self.deep2.forward_global(y, col_idx - 1)
    return y

    
  # x -> [y1, y2, y3, ..., y_depth]
  def forward(self, x, mode = 'local', col_idx = None):
    # global mode: use a single-column branch
    if (mode == 'global') and (self.training):
      if col_idx is None:
        raise ValueError('global mode requires col_idx!')
      return self.forward_global(x, col_idx)
    
    if self.depth == 0:
      # base case
      return [self.f1(x)]

    # local mode: join branches with local drop-path
    if (mode == 'local') or (not self.training):
      shallow_output = self.shallow(x)
      
      # first f{d}
      deep_outputs = self.deep1(x, mode='local')
      merged = self._join_list(deep_outputs)
      
      # second f{d}
      deep_outputs = self.deep2(merged, mode = 'local')
      
      return [shallow_output] + deep_outputs

    

# FractalBlock: a fractal module with C columns
class FractalBlock(nn.Module):
  # A fractal block with C columns (depth = C - 1)
  def __init__(self, in_channels, num_columns = 3, drop_p = 0.15):
    super().__init__()
    self.num_columns = num_columns
    self.drop_p = drop_p
    self.root = FractalNode(in_channels, depth = num_columns - 1, drop_p = drop_p)

  def forward(self, x ,mode = 'local', col_idx = None):
    if (not self.training) or (mode == 'local'):
        cols = self.root(x, mode = 'local')
        return self.root._join_list(cols)
    
    if mode == 'global':
      if col_idx is None:
        raise ValueError('global mode requires col_idx!')
      return self.root.forward_global(x, col_idx)


class FractalNet(nn.Module):
  # num of fractal blocks: 5
  # channels: 64 -> 128 -> 256 -> 512 -> 512
  # max-pooling: applied after each block
  # final: global average pooling + linear classifier
  def __init__(self, num_classes = 10, num_columns = 3, drop_p = 0.15, mix_p = 0.5):
    super().__init__()

    self.num_columns = num_columns
    self.mix_p = mix_p

    channels = [64, 128, 256, 512, 512]
    last_channels = channels[-1]

    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = channels[0], kernel_size = 3, stride = 1, padding = 1)

    self.projs = nn.ModuleList()
    self.blocks = nn.ModuleList()
    self.pools = nn.ModuleList()

    in_channels = channels[0]
    for out_channels in channels:

      # 1x1 convolution for channel adjustment
      if in_channels != out_channels:
        self.projs.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0))
      else:
        self.projs.append(nn.Identity())

      self.blocks.append(FractalBlock(in_channels = out_channels, num_columns = num_columns, drop_p = drop_p))
      self.pools.append(nn.MaxPool2d(kernel_size = 2, stride = 2))

      in_channels = out_channels

    self.bn = nn.BatchNorm2d(last_channels)
    self.relu = nn.ReLU()
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(p=0.2)
    
    self.num_features = last_channels
    self.fc = nn.Linear(last_channels, num_classes)

  def forward_features(self, x):
    # selects local / global mode
    if self.training and torch.rand(1).item() >= self.mix_p:
      mode = 'global'
      col_idx = torch.randint(0, self.num_columns, (1, )).item()
    else:
      mode = 'local'
      col_idx = None

    out = self.conv1(x)

    for proj, block, pool in zip(self.projs, self.blocks, self.pools):
      out = proj(out)
      out = block(out, mode = mode, col_idx = col_idx)
      out = pool(out)

    out = self.bn(out)
    out = self.relu(out)
    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = self.dropout(out)
    return out

  def forward(self, x):
    features = self.forward_features(x)
    logits = self.fc(features)
    return logits

class FractalNetEncoder(nn.Module):
    def __init__(self, num_columns = 3, drop_p = 0.15, mix_p = 0.5):
        super().__init__()
        self.backbone = FractalNet(num_classes = 10,num_columns = 3, drop_p = 0.15, mix_p = 0.5)
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone.forward_features(x)


def fractalnet_encoder():
    return FractalNetEncoder(num_columns = 3, drop_p = 0.15, mix_p = 0.5)