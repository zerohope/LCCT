import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
batch_size=2
class MyData(Dataset):
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        return self.train_x[item]

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=40,
            hidden_size=44,
            num_layers=1,
        )
        self.hidden2value = nn.Linear(128, 1)
    def forward(self,x,act):
        # 以下关于shape的注释只针对单向
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        hidden = (torch.randn(1, batch_size, 44),
                  torch.randn(1,batch_size, 44))
        _,(h_n,c_n)=self.rnn(x,hidden)
        brench_embeding=torch.cat((h_n,c_n),dim=2)
        action,act_length=act
        rs = []
        for idx,data in enumerate(act_length):
         act, length = action[idx],data
         temp=[]
         for i in range(length):
             value_input=torch.cat((brench_embeding[0][idx],act[i]),dim=0)
             val=self.hidden2value(value_input)
             temp.append(val)
         rs.append(temp)
        return rs
        #return (h_n,c_n)

#TODO 模拟
net=RNN()
train_x=[torch.randn(5,40),
         torch.randn(6,40),
         torch.randn(7,40),
         torch.randn(8,40),
         torch.randn(9,40),
         torch.randn(10,40)
]

train_action=[torch.randn(5,40),
         torch.randn(6,40),
         torch.randn(7,40),
         torch.randn(8,40),
         torch.randn(9,40),
         torch.randn(10,40)
]


def collate_fn(train_data):
    #train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    train_data = nn.utils.rnn.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data, data_length



train_seq = MyData(train_x)
train_action=MyData(train_action)
seq_dataloader=DataLoader(train_seq, batch_size=batch_size, collate_fn=collate_fn)
act_dataloader=DataLoader(train_action, batch_size=batch_size, collate_fn=collate_fn)


#TODO 模拟 数据未分割
for seq, act in zip(seq_dataloader, act_dataloader):
    seq_data, seq_length = seq
    seq_pack_data = nn.utils.rnn.pack_padded_sequence(seq_data, seq_length,batch_first=True,enforce_sorted=False)
    output = net(seq_pack_data,act)
    print(output)
    #print(torch.max(output,0))





'''
loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
net.zero_grad()
'''

#rs=net(input,actions)
#print(rs)
'''
loss=loss_function(rs[0],torch.tensor([1.0]))
loss.backward()
optimizer.step()
'''