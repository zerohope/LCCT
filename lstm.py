import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
            hidden_size=40,
            num_layers=1,
            #batch_first=True
        )
        self.hidden2value = nn.Linear(120, 1)
    def forward(self,x,action):
        # 以下关于shape的注释只针对单向
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        hidden = (torch.randn(1, 1, 40),
                  torch.randn(1, 1, 40))
        output,(h_n,c_n)=self.rnn(x,hidden)
        brench_embeding=torch.cat((h_n[0][0],c_n[0][0]))
        rs=[]
        for a in action:
             value_input=torch.cat((brench_embeding,a))
             val=self.hidden2value(value_input)
             rs.append(val)
        return rs

#TODO 批训练，变长
net=RNN()
train_x=[torch.randn(5,40),
         torch.randn(6,40),
         torch.randn(7, 40),
         torch.randn(8, 40),
         torch.randn(9, 40),
         torch.randn(10, 40)
]
def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    train_data = nn.utils.rnn.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data, data_length

train_data = MyData(train_x)
train_dataloader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)
for data, length in train_dataloader:
    data = nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True)
    print(data)


actions=[torch.randn(40) for _ in range(5)]
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