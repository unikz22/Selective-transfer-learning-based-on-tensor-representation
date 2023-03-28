import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import scipy.io as sio
import pprint

cuda =torch.cuda.is_available()
if cuda:
    device='cuda'
else:
    device='cpu'

class AutoEncoder(nn.Module,):
    def __init__(self, n_in, encoder_units):
        super(AutoEncoder, self).__init__()
        
        self.encoder_units = encoder_units
        self.decoder_units = list(reversed(encoder_units))
        self.in_size=n_in
        self.encoder = nn.Sequential(
            nn.Linear(self.in_size,self.encoder_units[0]),
            nn.ReLU(),
            nn.Linear(self.encoder_units[0], self.encoder_units[1]),
            nn.ReLU(),
            nn.Linear(self.encoder_units[1], self.encoder_units[2]),
            nn.ReLU(),
            nn.Linear(self.encoder_units[2], self.encoder_units[3]),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_units[0], self.decoder_units[1]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[1], self.decoder_units[2]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[2], self.decoder_units[3]),
            nn.ReLU(),
            nn.Linear(self.decoder_units[3], self.in_size)
        )

    def forward(self,x):
        en = self.encoder(x)
        de = self.decoder(en)
        return en,de
    
# path='F:/zw_work/work1/dataset/tucker_XJTU_PHM/'
# X11 = sio.loadmat(path+'XJB1_1.mat')['XB1_1']
# X12 = sio.loadmat(path+'XJB1_2.mat')['XB1_2']
# X13 = sio.loadmat(path+'XJB1_3.mat')['XB1_3']
# X14 = sio.loadmat(path+'XJB1_4.mat')['XB1_4']
# X15 = sio.loadmat(path+'XJB1_5.mat')['XB1_5']
    


# # TrX = np.vstack([X11,X12,X13,X14,X15])
# # Xtr=datanorm(TrX,TrX)
# # X11=datanorm(X11,TrX)
# # X12=datanorm(X12,TrX)
# # X13=datanorm(X13,TrX)
# # X14=datanorm(X14,TrX)
# # X15=datanorm(X15,TrX)

# Xtr=[X11,X12,X13,X14,X15]

# autoencoder = AutoEncoder(2560,[1024,512,128,25])
# optimizer = optim.Adam(autoencoder.parameters(),lr=0.001)
# criterion = nn.MSELoss()
# ploss=[]

# for epoch in range(100):
#     print("epoch:",epoch)
#     for i in range(len(Xtr)):
#         optimizer.zero_grad()
#         # xx=np.double(Xtr[i])
#         # xx = torch.from_numpy(xx)
#         xx=torch.Tensor(Xtr[i]).float().cpu()
#         enc, dec = autoencoder(xx)
        
#         loss_ae = criterion(dec, xx)
        
#         loss_ae.backward()
        
#         optimizer.step()
#     print("loss:",loss_ae.item())
#     ploss.append(loss_ae.item())
    
# sio.savemat("loss.mat",{"ploss":ploss})
# # # 查看中间层低维的输出
# with torch.no_grad():
#     X11=torch.Tensor(X11).float().cpu()
#     Fea_X1, _ = autoencoder(X11)
#     Fea_X1 = Fea_X1.cpu().detach().numpy()
    
#     X12=torch.Tensor(X12).float().cpu()
#     Fea_X2, _ = autoencoder(X12)
#     Fea_X2 = Fea_X2.cpu().detach().numpy()
    
#     X13=torch.Tensor(X13).float().cpu()
#     Fea_X3, _ = autoencoder(X13)
#     Fea_X3 = Fea_X3.cpu().detach().numpy()
    
#     X14=torch.Tensor(X14).float().cpu()
#     Fea_X4, _ = autoencoder(X14)
#     Fea_X4 = Fea_X4.cpu().detach().numpy()
    
#     X15=torch.Tensor(X15).float().cpu()
#     Fea_X5, _ = autoencoder(X15)
#     Fea_X5 = Fea_X5.cpu().detach().numpy()
    
#     # sio.savemat('AE_tucker_XJTU1.mat',{'Fea_1':Fea_X1.cpu().detach().numpy()})
#     sio.savemat('AE_tucker_XJTU1.mat',{'Fea_1':Fea_X1,'Fea_2':Fea_X2,'Fea_3':Fea_X3,'Fea_4':Fea_X4,'Fea_5':Fea_X5})
