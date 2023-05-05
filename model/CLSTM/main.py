from dbrnna_sy import Run_Model
from preprocess import Run_Preprocess
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

FileName = 'lifelog' #MyData.csv 라면 MyData 를 입력

Config_W2V_MinCNT = 5
Config_W2V_Window = 5

Config_Min_Smpl_Per_Class = 1
Config_Num_CV = 10
Config_Model_Type = "LSTM" #CNN, LSTM, CLSTM
Config_Model_Epochs = 5
Config_W2V_Size = 200
Config_W2V_SentLen = 50
Config_Model_BatchSize = 4
Config_Model_TopK = 10 # 가장 높은 것 하나를 추천 (1,5,10,20의 평균)
Config_Model_LR = 0.8

os.system('clear')


Run_Preprocess(FileName, Config_W2V_MinCNT, Config_W2V_Size, Config_W2V_Window)

Result = Run_Model(
    _Dataset_Name = FileName,
    _Min_Samples_Per_Class = Config_Min_Smpl_Per_Class,
    _Num_CV = Config_Num_CV, 
    _Model_Type = Config_Model_Type,
    _Epochs = Config_Model_Epochs,
    _W2V_Size = Config_W2V_Size,
    _W2V_Sent_Len = Config_W2V_SentLen,
    _Batch_Size = Config_Model_BatchSize,
    _TopK = Config_Model_TopK,
    _LR = Config_Model_LR
)

print("Result: ", Result)
