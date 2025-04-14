# Text-generation-based-on-CharLSTM
This paper introduces a text generation program of CharLSTM and provides the source code.
# 基于CharLSTM的文本生成程序介绍

## 一、程序概述

该程序构建了一个基于字符级长短期记忆网络（CharLSTM）的文本生成模型，能够学习给定文本数据的模式，并生成具有相似风格的新文本。其核心流程包括数据处理、模型定义与训练、文本生成以及结果保存。

## 二、数据处理

### （一）数据加载

程序从本地文件加载莎士比亚文本数据。通过以下代码实现：

```python
try:
    with open(Config.data_url, 'r', encoding='utf-8') as f:
        return f.read()
except FileNotFoundError:
    print(f"文件 {Config.data_url} 未找到，请确保已将数据文件下载到本地。")
    raise
```

### （二）构建字符映射表

构建字符到索引（`char2idx`）以及索引到字符（`idx2char`）的映射表，以便将文本数据转换为数值序列，方便模型处理。例如：

```python
self.chars = sorted(list(set(self.text)))
self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
self.vocab_size = len(self.chars)
```

### （三）创建滑动窗口数据集

将数值序列数据构建为滑动窗口数据集，用于模型训练。每个样本包含一个长度为`seq_length`的输入序列和对应的目标序列（输入序列的下一个字符）。

```python
class ShakespeareDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.seq_length],
            self.data[idx + 1:idx + self.seq_length + 1]
        )
```

## 三、模型定义

### （一）CharLSTM模型结构

模型由嵌入层、LSTM层和全连接层组成。

1. **嵌入层（Embedding Layer）**：将字符索引转换为低维向量表示，公式为：
    \[ \mathbf{e}_i = \mathbf{W}_e \mathbf{x}_i \]
    其中，\( \mathbf{e}_i \) 是第 \( i \) 个字符的嵌入向量，\( \mathbf{W}_e \) 是嵌入矩阵，\( \mathbf{x}_i \) 是字符索引。在程序中通过以下代码实现：

```python
self.embed = nn.Embedding(vocab_size, embed_dim)
```

2. **LSTM层（Long - Short Term Memory Layer）**：用于处理序列数据，能够捕捉文本的长期依赖关系。LSTM单元的计算公式较为复杂，主要包括输入门（Input Gate）、遗忘门（Forget Gate）、输出门（Output Gate）和记忆单元（Cell State）的更新。
    输入门：
    \[ \mathbf{i}_t = \sigma(\mathbf{W}_{ii} \mathbf{x}_t + \mathbf{W}_{hi} \mathbf{h}_{t - 1} + \mathbf{b}_i) \]
    遗忘门：
    \[ \mathbf{f}_t = \sigma(\mathbf{W}_{if} \mathbf{x}_t + \mathbf{W}_{hf} \mathbf{h}_{t - 1} + \mathbf{b}_f) \]
    输出门：
    \[ \mathbf{o}_t = \sigma(\mathbf{W}_{io} \mathbf{x}_t + \mathbf{W}_{ho} \mathbf{h}_{t - 1} + \mathbf{b}_o) \]
    记忆单元：
    \[ \tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_{ic} \mathbf{x}_t + \mathbf{W}_{hc} \mathbf{h}_{t - 1} + \mathbf{b}_c) \]
    \[ \mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t - 1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \]
    隐藏状态：
    \[ \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \]
    其中，\( \sigma \) 是Sigmoid函数，\( \odot \) 是逐元素相乘，\( \mathbf{W} \) 是权重矩阵，\( \mathbf{b} \) 是偏置向量，\( \mathbf{x}_t \) 是当前时刻的输入，\( \mathbf{h}_{t - 1} \) 是上一时刻的隐藏状态，\( \mathbf{C}_{t - 1} \) 是上一时刻的记忆单元。在程序中：

```python
self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
```

3. **全连接层（Fully - Connected Layer）**：将LSTM层的输出映射回字符空间，得到每个字符的预测概率。公式为：
    \[ \mathbf{y}_t = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o \]
    其中，\( \mathbf{y}_t \) 是预测的字符概率分布，\( \mathbf{W}_o \) 是权重矩阵，\( \mathbf{b}_o \) 是偏置向量，\( \mathbf{h}_t \) 是LSTM层的输出。在程序中：

```python
self.fc = nn.Linear(hidden_dim, vocab_size)
```

### （二）模型前向传播

前向传播过程将输入序列依次通过嵌入层、LSTM层和全连接层，得到预测结果。

```python
def forward(self, x, hidden=None):
    x = self.embed(x)  # (B, S) -> (B, S, E)
    out, hidden = self.lstm(x, hidden)  # (B, S, H)
    out = self.fc(out)  # (B, S, V)
    return out, hidden
```

其中，\( B \) 是批大小，\( S \) 是序列长度，\( E \) 是嵌入维度，\( H \) 是LSTM隐藏层维度，\( V \) 是词汇表大小。

## 四、模型训练

### （一）训练流程

1. 初始化模型、损失函数（交叉熵损失）和优化器（Adam优化器）。

```python
model = CharLSTM(processor.vocab_size, Config.embed_dim, Config.hidden_dim).to(Config.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
```

2. 在每个训练轮次（epoch）中，遍历数据加载器，将输入数据和目标数据传入模型进行前向传播、计算损失、反向传播和参数更新。

```python
for epoch in tqdm(range(Config.epochs), desc="Training Epochs"):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False):
        inputs = inputs.to(Config.device)
        targets = targets.to(Config.device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs.view(-1, processor.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
```

3. 记录每个epoch的平均损失，并绘制训练损失曲线。

```python
avg_loss = total_loss / len(dataloader)
losses.append(avg_loss)
print(f"Epoch [{epoch + 1}/{Config.epochs}] Loss: {avg_loss:.4f}")

# 绘制训练曲线
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
```

### （二）损失函数

使用交叉熵损失（Cross - Entropy Loss）来衡量模型预测结果与真实标签之间的差异，公式为：
\[ L = - \frac{1}{N} \sum_{i = 1}^{N} \sum_{j = 1}^{C} y_{ij} \log(\hat{y}_{ij}) \]
其中，\( N \) 是样本数量，\( C \) 是类别数量（即词汇表大小），\( y_{ij} \) 是真实标签（如果第 \( i \) 个样本的真实类别是 \( j \)，则 \( y_{ij} = 1 \)，否则 \( y_{ij} = 0 \)），\( \hat{y}_{ij} \) 是模型预测的第 \( i \) 个样本属于类别 \( j \) 的概率。在程序中通过 `nn.CrossEntropyLoss()` 实现。

## 五、文本生成

### （一）温度采样策略

为了生成多样化的文本，使用温度采样策略。首先将模型输出的logits除以温度系数（temperature），然后通过Softmax函数得到概率分布，最后从该概率分布中采样生成下一个字符。
温度采样函数为：

```python
def temperature_sampling(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### （二）文本生成过程

从输入文本中随机选择一个起始序列，然后在每个时间步，根据模型预测的概率分布（经过温度采样）生成下一个字符，并将生成的字符加入到输入序列中，继续生成下一个字符，直到生成指定长度的文本。

```python
def generate_text(model, processor, temperature=1.0):
    model.eval()
    start_idx = torch.randint(0, len(processor.data) - Config.seq_length, (1,)).item()
    input_seq = processor.data[start_idx:start_idx + Config.seq_length].unsqueeze(0).to(Config.device)

    generated = []
    for _ in tqdm(range(Config.gen_length), desc=f"Generating text with temp {temperature}"):
        with torch.no_grad():
            outputs, _ = model(input_seq)
            logits = outputs[:, -1, :]
            next_idx = temperature_sampling(logits, temperature)
            generated.append(next_idx.item())
            input_seq = torch.cat((input_seq[:, 1:], next_idx), dim=1)

    return "".join([processor.idx2char[idx] for idx in generated])
```

## 六、结果保存

### （一）保存训练损失曲线

将训练损失曲线保存为图像文件，便于观察模型的训练过程。

```python
output_dir = os.path.dirname(Config.loss_image_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(Config.loss_image_path)
```

### （二）保存生成文本

将生成的文本保存为文本文件，方便查看和分析。

```python
output_dir = os.path.dirname(Config.output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(Config.output_file_path, 'w', encoding='utf-8') as f:
    f.write(output_text)
```

### （三）保存模型结构图（可选）

如果安装了`torchviz`库，程序可以保存模型的结构图，有助于理解模型的架构。

```python
try:
    from torchviz import make_dot
    x = torch.zeros(1, Config.seq_length, dtype=torch.long).to(Config.device)
    out, _ = model(x)
    dot = make_dot(out, params=dict(model.named_parameters()))
    output_dir = os.path.dirname(Config.model_structure_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dot.render(Config.model_structure_path, format="png")
except ImportError:
    print("未安装torchviz，跳过模型结构图生成")
```

通过上述各个模块的协同工作，该程序能够有效地训练一个文本生成模型，并提供了丰富的功能来展示和保存训练与生成的结果。
