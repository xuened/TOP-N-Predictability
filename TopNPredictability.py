import pickle
import random
import numpy as np  
import matplotlib.pyplot as plt  
import collections
import math
from sklearn import linear_model

def readSessiontxt(in_file_path,in_data_name):
    lines = []
    Session_row = 100
    ItemId_row = 100
    UserId = 'UserId'
    if in_data_name == 'rsc15' or in_data_name == 'yoochoose' or in_data_name == 'instacart':
        UserId = 'SessionId'
    with open(in_file_path, 'r') as file_to_read:
        first_line_flag = 1
        row_count = 0
        while True:
            row_count += 1
            if row_count >= 10000000:
                break
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n').split("\t")
            if first_line_flag == 1:
                first_line_flag = 0
                print(line)
                for i in range(len(line)):
                    if line[i] == UserId:    
                    # if line[i] == 'UserId':
                        Session_row = i
                        # print(Session_row)
                    elif line[i] == 'ItemId':
                        ItemId_row = i
                        # print(ItemId_row)
                continue
            # print(Session_row,ItemId_row)
            lines.append(line)
    two_raw_data = []
    # print(len(lines))
    for i in range(len(lines)):
        two_raw_data.append([lines[i][Session_row],lines[i][ItemId_row]])
    return two_raw_data

def ReadData(in_data_name,in_data_name_houzui):
    train_full_path = ''
    if in_data_name == 'dunnhumby':
        train_full_path = ''
    elif in_data_name == 'tmall':
        train_full_path = ''
    elif in_data_name == 'rsc15':
        train_full_path = ''
    elif in_data_name == 'instacart':
        train_full_path = ''

    two_raw_data = readSessiontxt(train_full_path,in_data_name)
    print('len(two_raw_data):', len(two_raw_data))
    SeIt = {}  # session item字典
    for cell in two_raw_data:
        if cell[0] in SeIt:
            temp_UP = SeIt[cell[0]]
            temp_UP += [cell[1]]
            SeIt[cell[0]] = temp_UP
        else:
            SeIt[cell[0]] = [cell[1]]
    return SeIt

def GetN(l):
    one_user_position_set = set(l)       
    one_user_neighbor = []  
    for i in range(len(l)-1):
        one_user_neighbor.append([l[i],l[i+1]])
    one_position_to_onthers_number = {}     
    for neighbor in one_user_neighbor:
        if neighbor[0] in one_position_to_onthers_number:
            temp_position = one_position_to_onthers_number[neighbor[0]]
            temp_position += [neighbor[1]]
            one_position_to_onthers_number[neighbor[0]] = temp_position
        else:
            one_position_to_onthers_number[neighbor[0]] = [neighbor[1]]
    Nr = 0      
    for posi in one_position_to_onthers_number:
        temp_length = len(set(one_position_to_onthers_number[posi]))
        if temp_length > Nr:
            Nr = temp_length
    return len(l),len(one_user_position_set), Nr

def EasyGetPredictability(In_C,In_X_scale, In_N, In_S):
    Pi = [x/10000 for x in range(1,10000)]
    gap = 1000000000
    result = 10000
    for x in Pi:
        formula = 1
        for i in range(In_X_scale):
            formula *= (In_C[i] * x) ** (In_C[i] * x)
        formula = formula * ((1 - sum(In_C[:In_X_scale]) * x) / (In_N - In_X_scale)) ** (1 - sum(In_C[:In_X_scale]) * x) \
                  - 2 ** (-In_S)
        formula = complex(formula).real
        if abs(formula) < gap:
            gap = abs(formula)
            result = x
    return result

def contains(small, big):
    for i in range(len(big)-len(small)+1):
        if big[i:i+len(small)] == small:
            return True
    return False

def actual_entropy(l):
    n = len(l)
    sequence = [l[0]]
    sum_gamma = 0
    for i in range(1, n):
        for j in range(i+1, n+1):
            s = l[i:j]
            if not contains(list(s), sequence): # s is not contained in previous sequence
                sum_gamma += len(s)
                sequence.append(l[i])
                break
    ae = 1 / (sum_gamma / n ) * math.log(n)
    return ae

def GetC(in_SeIt,in_mini_one_session_time):
    All_Data = []
    for se in in_SeIt:
        All_Data += in_SeIt[se]
    N = len(list(set(All_Data)))
    list_counter = collections.Counter(All_Data)
    temp_C = []
    for key in list_counter:
        temp_C.append(list_counter[key])
    temp_C.sort(reverse=True)
    guiyi_C = []
    if len(temp_C) > 100:
        print('temp_C:',temp_C[:100])
    else:
        print('temp_C:', temp_C)
    for tup in temp_C:
        if tup < in_mini_one_session_time:  
            continue
        guiyi_C.append(tup / float(temp_C[0]))
    C_All = guiyi_C
    x,y = [],[]
    for c in range(len(C_All)):
        x.append(c+1)
        y.append(C_All[c])

    print('return x:', x[:10])
    print('return y:', y[:10])
    # X=np.log10(x)  
    # # X = np.array(x_mean)
    # Y=np.log10(y)
    X = x
    Y = y
    return N,Y

def DataFitAndVisualization(in_data_name,in_X,in_Y):
    X_parameter=[]
    Y_parameter=[]
    in_X = np.log10(in_X)
    in_Y = np.log10(in_Y)
    for single_square_feet, single_price_value in zip(in_X, in_Y):
        X_parameter.append([float(single_square_feet)])
        Y_parameter.append(float(single_price_value))
        # Xlog.append([np.log10(float(single_square_feet))])
        # Ylog.append(np.log10(float(single_price_value)))

    if in_data_name == 'rsc15':
        regr = linear_model.LinearRegression()
        # regr.fit(X_parameter, Y_parameter)
        regr.fit(X_parameter[:10], Y_parameter[:10])
    else:
        regr = linear_model.LinearRegression()
        # regr.fit(X_parameter, Y_parameter)
        regr.fit(X_parameter, Y_parameter)
    print('Coefficients: \n', regr.coef_,)
    return 0

def CalPre(In_SeIt,in_data_name,in_PiNumber):
    X_scale = 10
    N, Y = GetC(In_SeIt, 100)
    C = Y[:10]
    if in_data_name == 'rsc15':
        C = [1.0, 0.7335505066351673, 0.5592920027315849, 0.512855059891844, 0.41844004355770476, 0.35541979660766687, 0.3390948856610251, 0.33520053155165097, 0.32286225798711726, 0.31038555951348257]
    elif in_data_name == 'tmall':
        C = []
    print('N:',N)
    print('C:', C)
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    DataFitAndVisualization(data_name, X, C)
    NewC = []
    for i in C:
        NewC.append(i/C[in_PiNumber-1])
    print(NewC)
    WholeData = []
    for user in In_SeIt:
        randomNumber = random.random()
        # randomNumber = 1
        if randomNumber < 0.5:
            continue
        WholeData += In_SeIt[user]
        if len(WholeData) > 10000:      
            break
    S = actual_entropy(WholeData)
    print('S:',S)
    # L, Nc, Nr = GetN(WholeData)
    # print("length of data:", L, "N:", N, "Nc:", Nc, "Nr:", Nr)
    WholePre = EasyGetPredictability(NewC, X_scale, N, S)
    print('Whole Pre:',WholePre)

if __name__=="__main__":
    ##################################
    # dunnhumby   instacart     rsc15       tmall       
    data_name_houzui = ''
    data_name = 'rsc15'           
    mini_length = 100
    PiNumber = 1   
    ##################################
    SeIt = ReadData(data_name,data_name_houzui)
    CalPre(SeIt,data_name,PiNumber)


