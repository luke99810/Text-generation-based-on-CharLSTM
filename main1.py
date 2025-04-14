import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import dummy
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 设置matplotlib后端
import matplotlib

matplotlib.use('TkAgg')


# ==================== 配置参数 ====================
class Config:
    # 数据路径（请确认实际路径存在）
    data_path = r"C:\Users\宿心\Desktop\37\input.txt"

    # 模型结构
    seq_length = 100  # 输入序列长度
    embed_dim = 64  # 嵌入层维度
    hidden_dim = 128  # LSTM隐藏层维度

    # 训练参数
    batch_size = 128
    epochs = 5  # 增加训练轮次以提高效果
    learning_rate = 0.003
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成参数
    gen_length = 1000  # 生成文本长度
    temperatures = [0.5, 1.0, 1.5]  # 不同温度系数

    # 输出路径（自动创建目录）
    output_dir = r"C:\Users\宿心\Desktop\37"
    output_text = os.path.join(output_dir, "generated.txt")
    loss_plot = os.path.join(output_dir, "training_loss.png")
    model_struct = os.path.join(output_dir, "model_architecture")


# ==================== 数据预处理 ====================
class TextProcessor:
    def __init__(self):
        self.text = self.load_text()
        self.build_vocab()

    def load_text(self):
        """加载文本数据"""
        try:
            with open(Config.data_path, 'r', encoding='utf-8') as f:
                print(f"成功加载数据，文本长度：{len(f.read())}字符")
                f.seek(0)
                return f.read()
        except Exception as e:
            print(f"文件加载失败: {str(e)}")
            raise

    def build_vocab(self):
        """构建字符词典"""
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        print(f"构建词典完成，共{self.vocab_size}个唯一字符")

        # 创建字符映射
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}

        # 转换为张量
        self.data = torch.tensor(
            [self.char2idx[c] for c in self.text],
            dtype=torch.long
        )


class ShakespeareDataset(Dataset):
    """滑动窗口数据集"""

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.data[idx: idx + self.seq_len],  # 输入序列
            self.data[idx + 1: idx + self.seq_len + 1]  # 目标序列（右移一位）
        )


# ==================== 模型定义 ====================
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # 输入形状: (batch_size, seq_len)
        x = self.embed(x)  # (B, S) => (B, S, E)
        out, hidden = self.lstm(x, hidden)  # (B, S, H)
        logits = self.fc(out)  # (B, S, V)
        return logits, hidden


# ==================== 训练流程 ====================
def train_model():
    # 初始化数据
    processor = TextProcessor()
    dataset = ShakespeareDataset(processor.data, Config.seq_length)
    loader = DataLoader(dataset,
                        batch_size=Config.batch_size,
                        shuffle=True,
                        pin_memory=True)

    # 创建模型
    model = CharLSTM(processor.vocab_size,
                     Config.embed_dim,
                     Config.hidden_dim).to(Config.device)
    print(f"模型已创建，参数数量：{sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    losses = []

    # 训练循环
    print("开始训练...")
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{Config.epochs}")

        for inputs, targets in progress:
            # 数据移至设备
            inputs = inputs.to(Config.device)
            targets = targets.to(Config.device)

            # 前向传播
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, processor.vocab_size),
                             targets.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # 梯度裁剪
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        # 保存平均损失
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

    # 保存训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'o-')
    plt.title("Training Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    os.makedirs(Config.output_dir, exist_ok=True)
    plt.savefig(Config.loss_plot)
    plt.close()
    print(f"训练曲线已保存至：{Config.loss_plot}")

    return model, processor


# ==================== 文本生成 ====================
def generate_text(model, processor, temperature=1.0):
    model.eval()
    # 随机选择起始位置
    start_idx = torch.randint(0, len(processor.data) - Config.seq_length, (1,)).item()
    input_seq = processor.data[start_idx:start_idx + Config.seq_length]
    input_seq = input_seq.unsqueeze(0).to(Config.device)  # 添加批次维度

    hidden = None  # 初始化隐藏状态
    generated = []

    progress = tqdm(range(Config.gen_length),
                    desc=f"生成中 (温度={temperature})")
    for _ in progress:
        with torch.no_grad():
            # 前向传播
            logits, hidden = model(input_seq, hidden)

            # 获取最后一个字符的概率
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # 采样
            next_idx = torch.multinomial(probs, num_samples=1)
            generated.append(next_idx.item())

            # 更新输入序列
            input_seq = torch.cat((input_seq[:, 1:], next_idx), dim=1)

    # 转换为文本
    return ''.join([processor.idx2char[idx] for idx in generated])


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(Config.output_dir, exist_ok=True)

    try:
        # 训练模型
        model, processor = train_model()

        # 生成不同温度的文本
        results = []
        for temp in Config.temperatures:
            text = generate_text(model, processor, temp)
            results.append(f"\n{'=' * 40}\n温度 {temp}:\n{text}\n")

        # 保存结果
        with open(Config.output_text, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))
        print(f"生成结果已保存至：{Config.output_text}")

        # 保存模型结构（需要安装torchviz）
        try:
            from torchviz import make_dot

            dummy.zeros(1, Config.seq_length, dtype=torch.long).to(Config.device)
            outputs, _ = model(dummy)
            make_dot(outputs.mean(), params=dict(model.named_parameters())).render(
                Config.model_struct, format="png", cleanup=True)
            print(f"模型结构图已保存至：{Config.model_struct}.png")
        except ImportError:
            print("提示：安装torchviz可生成模型结构图：pip install torchviz graphviz")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")