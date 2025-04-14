import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import requests
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 设置matplotlib使用TkAgg后端
import matplotlib

matplotlib.use('TkAgg')


# ==================== 配置参数 ====================
class Config:
    # 数据参数
    # 这里修改为本地文件路径，需要手动下载 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt 到本地
    data_url = r"C:\Users\宿心\Desktop\37\input.txt"
    seq_length = 100  # 滑动窗口长度
    batch_size = 128  # 批大小

    # 模型参数
    embed_dim = 64  # 嵌入维度
    hidden_dim = 128  # LSTM隐藏层维度

    # 训练参数
    epochs = 2  # 训练轮次
    learning_rate = 0.003  # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成参数
    gen_length = 1000  # 生成文本长度
    temperatures = [0.5, 1.0, 2.0]  # 温度系数

    # 新增：指定输出文件完整路径
    output_file_path = r"C:\Users\宿心\Desktop\37\generated_text_results.txt"
    # 新增：指定训练损失图片完整路径
    loss_image_path = r"C:\Users\宿心\Desktop\37\training_loss.png"
    # 新增：指定模型结构图片完整路径
    model_structure_path = r"C:\Users\宿心\Desktop\37\model_structure"


# ==================== 数据处理模块 ====================
class TextProcessor:
    def __init__(self):
        self.text = self._load_data()
        self._build_vocab()

    def _load_data(self):
        """加载莎士比亚数据集"""
        try:
            with open(Config.data_url, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"文件 {Config.data_url} 未找到，请确保已将数据文件下载到本地。")
            raise

    def _build_vocab(self):
        """构建字符映射表"""
        self.chars = sorted(list(set(self.text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # 转换为数值序列
        self.data = torch.tensor(
            [self.char2idx[ch] for ch in self.text],
            dtype=torch.long
        )


class ShakespeareDataset(Dataset):
    """滑动窗口数据集"""

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


# ==================== 模型定义 ====================
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)  # (B, S) -> (B, S, E)
        out, hidden = self.lstm(x, hidden)  # (B, S, H)
        out = self.fc(out)  # (B, S, V)
        return out, hidden


# ==================== 训练流程 ====================
def train_model():
    processor = TextProcessor()
    dataset = ShakespeareDataset(processor.data, Config.seq_length)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    model = CharLSTM(processor.vocab_size, Config.embed_dim, Config.hidden_dim).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    # 记录训练损失
    losses = []

    print("开始训练...")
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

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{Config.epochs}] Loss: {avg_loss:.4f}")

    # 绘制训练曲线
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 确保输出目录存在
    output_dir = os.path.dirname(Config.loss_image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(Config.loss_image_path)

    # 显示图像
    plt.show()
    plt.close()

    return model, processor


# ==================== 文本生成 ====================
def temperature_sampling(logits, temperature=1.0):
    """温度采样函数"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(model, processor, temperature=1.0):
    """生成文本"""
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


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 训练模型
    model, processor = train_model()

    # 生成示例文本并保存到文件
    output_text = ""
    print("\n生成结果对比：")
    for temp in Config.temperatures:
        text = generate_text(model, processor, temperature=temp)
        output_text += f"\nTemperature {temp}:\n{'=' * 50}\n{text}\n"
        print(f"\nTemperature {temp}:\n{'=' * 50}\n{text[:500]}...\n")

    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(Config.output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(Config.output_file_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"生成的文本已保存到 {Config.output_file_path}")
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
    except PermissionError as e:
        print(f"权限错误: {e}")
    except UnicodeEncodeError as e:
        print(f"编码错误: {e}")
    except Exception as e:
        print(f"保存生成文本时发生其他错误: {e}")

    # 保存模型结构图（需安装torchviz）
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
