import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import r2_score
import scipy.ndimage
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo

    
app_ui = ui.page_fluid(
    ui.input_file("file1", "Choose an MRI dataframe in CSV, wavenumber 600 to 4000 cm-1", accept=[".csv"], multiple=False),
    ui.input_file("file2", "Choose a CSV file with organic matter content for validation", accept=[".csv"], multiple=False),
    ui.output_text_verbatim("verb", placeholder=True),
    ui.download_button("down", "Download the predictions") 
)   


def server(input: Inputs, output: Outputs, session: Session):  
    @reactive.calc
    def parsed_file():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        
        scaler = joblib.load('scaler.save')

        class Model(nn.Module):
            # Input layer (1701 features) --> Hidden layer1 (p neurons) --> Hidden layer2 (m neurons) --> Output
            def __init__(self, in_features=1701, h1=100, h2=100, h3=100, out_features=1):
                super().__init__()
                self.fc1 = nn.Linear(in_features, h1)
                self.fc2 = nn.Linear(h1,h2)
                self.fc3 = nn.Linear(h2,h3)       
                self.out = nn.Linear(h3,out_features)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.out(x))

                return x
            
        model = Model()
        model.load_state_dict(torch.load('state_dict.model',map_location=torch.device('cpu')))

        X = pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
            file[0]["datapath"],header=None)
        
        X = scipy.ndimage.zoom(X, (1,1701/X.shape[1]),order=5)

        X = scaler.transform(X)

        with torch.no_grad():
            y_eval = model(torch.FloatTensor(X))

        return y_eval
    #pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
            #file[0]["datapath"]



        #)
    @render.text
    def verb():
        file: list[FileInfo] | None = input.file2()
        if file is None:
            return ' '
        y_test = pd.read_csv(file[0]["datapath"],header=None)
        y_test = y_test.to_numpy()
        y_eval = parsed_file()
        val = []
        val.append(r2_score(y_test,y_eval.cpu().detach().numpy()))
        val.append(np.sqrt(mean_squared_error(y_test,y_eval.cpu().detach().numpy())))
        return str("R^2 : %5.2f, RMSE : %5.2f" % (val[0], val[1])) 


            
 
    @render.download(filename='file.csv')      
    async def down():
        X = parsed_file()
        X = X.detach().numpy()
        X = X.reshape(-1)
        X = np.char.mod('%f', X)
        yield str(",".join(X))
        
            
app = App(app_ui, server)

