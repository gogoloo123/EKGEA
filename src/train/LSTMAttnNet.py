import torch as t


class LSTMAttnNet(t.nn.Module):
    def __init__(self, embed_dim, hidden_dim, hidden_layers, dropout=0, device: t.device = 'cpu'):
        super(LSTMAttnNet, self).__init__()  # 传入子类，调用父类方法
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim # 隐藏层输出矩阵的特征数
        self.hidden_layers = hidden_layers
        self.device = device

        self.build_model(dropout)
        self.out_dim = hidden_dim

    def build_model(self, dropout):
        # 定义一个神经网络层，包含了一个线性层和一个ReLU激活函数
        self.attn_layer = t.nn.Sequential(
            t.nn.Linear(self.hidden_dim, self.hidden_dim),
            t.nn.ReLU(inplace=True) # inplace代表是否将激活函数的计算结果直接替换原始输入的内存位置
        ).to(self.device)
        # 创建了一个双向gru，并保存在了self.gru_layer
        # 输入参数：
        # embed_dim：输入特征维度，hidden_dim:隐藏状态维度，即输出特征维度，hidden_layers:GRU层数，bidirectional:指定为双向gru
        # batch_first：通常我们输入的数据shape=(batch_size,seq_length,embedding_dim),而batch_first默认是False,所以我们的输入数据最好送进LSTM之前将batch_size与seq_length这两个维度调换
        # dropout:控制在LSTM层中随机丢弃输入单元和隐藏单元的比例
        # 输出：output, (hn, cn) = rnn(input, (h0, c0))
        # output：包括当前时刻以及之前时刻所有hn的输出值
        # （hn，cn）：分别是当前时刻LSTM层的输出结果，记忆单元中的值
        self.lstm_layer = t.nn.LSTM(self.embed_dim, self.hidden_dim, self.hidden_layers, bidirectional=True, batch_first=True, dropout=dropout).to(self.device)
        self.lstm_layer_attn_w = t.nn.LSTM(self.embed_dim, self.hidden_dim, self.hidden_layers, bidirectional=True, batch_first=True, dropout=dropout).to(self.device)

    def attn_net_with_w(self, rnn_out, rnn_hn, neighbor_mask: t.Tensor, x):
        """
        :param rnn_out: [batch_size, seq_len, n_hidden * 2]
        :param rnn_hn: [batch_size, num_layers * num_directions, n_hidden]
        :return:
        """
        neighbor_mask_dim = neighbor_mask.unsqueeze(2) # 维度扩展，在第二个维度上扩展一个新的维度
        neighbor_mask_dim = neighbor_mask_dim.repeat(1, 1, self.hidden_dim) # 将张量进行复制，表示在第一个维度上复制一次，第二个维度复制一次，第三个维度复制self.hidden_dim次
        neighbor_mask_dim = neighbor_mask_dim.cuda()
        lstm_tmp_out = t.chunk(rnn_out, 2, -1)  # 把最后一维度分成两份 # 将张量 rnn_out 沿着指定维度进行分块操作，分成多个块，沿着最后一个维度分为两块。
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        assert h.shape == neighbor_mask_dim.shape
        h = t.where(neighbor_mask_dim == 1, h, neighbor_mask_dim)
        lstm_hidden = t.sum(rnn_hn, dim=1, keepdim=True) # 按维度求和
        atten_w = self.attn_layer(lstm_hidden)
        m = t.nn.Tanh()(h)
        atten_context = t.bmm(atten_w, m.transpose(1, 2)) # bmm批次中每一个step的矩阵乘法， transpose交换两个维度
        softmax_w = t.nn.functional.softmax(atten_context, dim=-1)  # 把最后一维度映射到[0,1]
        # 序列结果加权
        # context [batch_size, 1, hidden_dims]
        context = t.bmm(softmax_w, h)
        result = context.squeeze_(1)     #squeeze(arg)表示若第arg维的维度值为1，则去掉该维度。否则tensor不变。
        return result, softmax_w

    def forward(self, x, neighbor_mask):
        # x:[seq_length, batch_size, input_size]
        rnn_out, _ = self.lstm_layer(x) # 获取之前所有时刻lstm模型的输出结果
        # attention
        _, hc = self.lstm_layer_attn_w(x)
        hn = hc[0].permute(1, 0, 2) # 改变张量的维度顺序，从0,1,2变为1,0,2
        hn: t.Tensor
        out, weights = self.attn_net_with_w(rnn_out, hn, neighbor_mask, x)
        return out, weights
