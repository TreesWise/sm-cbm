import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler



#load data and preprocess
def data_load_preprocess(data_path):
    df_test = pd.read_csv(data_path, index_col='signaldate', parse_dates=True)
    df_test.columns = df_test.columns.str.replace('(\.\d+)$','', regex=True)

    col_dup_test = df_test.columns[df_test.columns.duplicated()]
    df_dup_test = df_test[col_dup_test]
    df_test = df_test.drop(col_dup_test, axis=1)
    # df_test = df_test.drop('Signal Date', axis=1)
    df_dup_test = df_dup_test.iloc[:,~df_dup_test.columns.duplicated()]
    df_test = pd.concat([df_test,df_dup_test], axis=1)
    return df_test


class Transform_data:
    def __init__(self, data) -> None:
        self.model = MinMaxScaler()
        self.data = data
        self.norm = None
        self.dnorm = None
    def normalize(self,):
        self.d = self.model.fit_transform(self.data)
        self.norm = pd.DataFrame(self.d, index=self.data.index, columns=self.data.columns)
        return self.norm
    def denormalize(self,):
        self.dn = self.model.inverse_transform(self.norm)
        self.dnorm = pd.DataFrame(self.dn, index=self.data.index, columns=self.data.columns)
        return self.dnorm



# Creating a PyTorch class architecture
# 883 ==> 3 ==> 883
class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()

		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 883 ==> 9
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(846, 512),
			torch.nn.Sigmoid(),
			torch.nn.Linear(512, 256),
			torch.nn.Sigmoid(),
			torch.nn.Linear(256, 128),
			torch.nn.Sigmoid(),
			torch.nn.Linear(128, 64),
			torch.nn.Sigmoid(),
			torch.nn.Linear(64, 3)
		)

		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 883
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(3, 64),
			torch.nn.Sigmoid(),
			torch.nn.Linear(64, 128),
			torch.nn.Sigmoid(),
			torch.nn.Linear(128, 256),
			torch.nn.Sigmoid(),
			torch.nn.Linear(256, 512),
			torch.nn.Sigmoid(),
			torch.nn.Linear(512, 846),
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

#model weight loading
# model = torch.load('utils\\AC\\model\\model_0.0175.pt', map_location=torch.device('cpu'))
# model.eval()

def recreation_loss(base_data_test, recreated_df_test):
    fault_id = []
    anom = []
    for idx,val in enumerate(range(base_data_test.shape[0])):
        v = mean_squared_error(base_data_test[idx,:], recreated_df_test[idx,:], multioutput='raw_values')[0]
        if v > 0.07:
            anom.append(v)
            fault_id.append(idx)
    print(len(fault_id))
    return fault_id


# recreated_df_test = model(df_tensor_test).cpu().detach().numpy()
# base_data_test = df_tensor_test.cpu().detach().numpy()
# faulty_date = recreation_loss(base_data_test, recreated_df_test)