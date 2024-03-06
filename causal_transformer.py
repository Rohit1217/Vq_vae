import torch.nn as nn
import torch.nn.functional as F
import torch
#from Causal_atten_block import block
import math

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
Device=get_device()

def Positional_encoding(x):
  b,T,C=x.shape
  pos=torch.zeros(T,C).to(Device)
  for j in range(T):
    for i in range(C):
        if i%2==0:
            denom=math.pow(10000,(2*i)/(C))
            pos[j,i]=math.sin(j/denom)
        else:
            denom=math.pow(10000,(2*(i-1))/(C))
            pos[j,i]=math.cos(j/denom)
  return pos

def mask_creator(size):
  mask=torch.ones(size,size)
  mask=torch.tril(mask)
  mask=mask.view(1,1,size,size)
  #mask[mask== 0] = float('-inf')
  return  mask



class Self_Attention(nn.Module):
  def __init__(self,emb_dim,n_head,T):
    super(Self_Attention,self).__init__()

    self.emb_dim=emb_dim
    self.n_head=n_head
    self.dim=emb_dim//n_head

    self.fc_q=nn.Linear(self.emb_dim,self.emb_dim).to(Device)
    self.fc_k=nn.Linear(self.emb_dim,self.emb_dim).to(Device)
    self.fc_v=nn.Linear(self.emb_dim,self.emb_dim).to(Device)
    self.fc_o=nn.Linear(self.emb_dim,self.emb_dim).to(Device)

    self.mask1=mask_creator(T).to(Device)
    self.register_buffer('mask',self.mask1)

  def forward(self,x):

    B,T,C=x.shape
    q,k,v=self.fc_q(x),self.fc_k(x),self.fc_v(x)
    q,k,v=q.view(B,T,self.n_head,self.dim),k.view(B,T,self.n_head,self.dim),v.view(B,T,self.n_head,self.dim)
    q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)

    att=torch.matmul(q,k.transpose(2,3))/torch.sqrt(torch.tensor(self.dim))
    att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

    att=F.softmax(att,dim=-1)
    att=torch.matmul(att,v)
    att=att.transpose(1,2)
    att=att.contiguous().view(B,T,self.emb_dim)
    x=self.fc_o(att)
    return x

class block(nn.Module):
  def __init__(self,emb_dim,n_head,T):
    super(block,self).__init__()
    self.emb_dim=emb_dim
    self.n_head=n_head
    self.T=T
    self.atten=Self_Attention(emb_dim,n_head,T)
    self.fc1=nn.Linear(emb_dim,4*emb_dim).to(Device)
    self.fc2=nn.Linear(4*emb_dim,emb_dim).to(Device)
    self.ln1=nn.LayerNorm(emb_dim).to(Device)
    self.ln2=nn.LayerNorm(emb_dim).to(Device)
    self.d1=nn.Dropout(p=0.1)
    self.d2=nn.Dropout(p=0.1)

  def forward(self,x):
    x = x + self.atten(self.ln1(self.d1(x)))
    residual=x
    x=self.ln2(self.d2(x))
    x=F.gelu(self.fc1(x))
    x=F.gelu(self.fc2(x))
    x=x+residual
    return x

'''x=torch.randn(2,3,4)
print(Positional_encoding(x))'''

class iGPT(nn.Module):
  def __init__(self,emb_dim,n_head,T,n_blocks,out_dim):
    super(iGPT,self).__init__()
    self.n=n_blocks
    self.emb_dim=emb_dim
    self.n_head=n_head
    self.T=T
    self.block_list = nn.ModuleList([block(self.emb_dim, self.n_head, self.T) for i in range(self.n)])
    self.embedding=nn.Embedding(num_embeddings=101,embedding_dim=emb_dim)
    #self.embedding1= nn.Linear(1,self.emb_dim)
    self.fc1= nn.Linear(self.emb_dim,self.emb_dim)
    self.fc2= nn.Linear(self.emb_dim,out_dim)

  def forward(self,x):
    B,T=x.shape
    x=self.embedding(x)
    pos=Positional_encoding(x)
    x=pos+x
    for block in self.block_list:
      x=block(x)
    x=F.gelu(self.fc1(x))
    x=self.fc2(x)
    return x
  
'''x=torch.zeros(2,3)
x=x.long().detach()

igt=iGPT(2,2,3,2,2)
print(igt(x).shape)'''
