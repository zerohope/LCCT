import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=40,
            hidden_size=40,
            num_layers=1,
            batch_first=True
        )
        self.hidden2value = nn.Linear(120, 1)
    def forward(self,x,action):
        # 以下关于shape的注释只针对单向
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output,(h_n,c_n)=self.rnn(x)
        output_in_last_timestep=output[:,-1,:] # 也是可以的
        #output_in_last_timestep=h_n[-1,:,:]
        c_nlast=c_n[:,-1,:]
        output_last=output_in_last_timestep[-1]
        brench_embeding=torch.cat((c_nlast[0],output_last))
        rs=[]
        for a in action:
            value_input=torch.cat((brench_embeding,a[0]))
            val=self.hidden2value(value_input)
            rs.append(val)
        return rs


net=RNN()
input=[torch.randn(1,40) for _ in range(5)]
inputs = torch.cat(input).view(len(input), 1, 40)
actions=[torch.randn(1,40) for _ in range(5)]


loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
net.zero_grad()
rs=net(inputs,actions)

loss=loss_function(rs[0],torch.tensor([1.0]))
loss.backward()
optimizer.step()