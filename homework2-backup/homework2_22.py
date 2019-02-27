







import xlrd
import matplotlib.pyplot as plt 
import numpy as np






def loadData(filename):
    workbook = xlrd.open_workbook(filename)
    boyinfo = workbook.sheet_by_index(0)
    col_num = boyinfo.ncols
    row_num = boyinfo.nrows
    col0 = boyinfo.col_values(0)[1:]
    data = np.array(col0)
    if col_num == 1:
        return data
    else:
        for i in range(col_num-1):
            coltemp = boyinfo.col_values(i+1)[1:]
            data = np.c_[data, coltemp]
    return data




def plotData(X, y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    p1 = plt.plot(X[pos,0],X[pos,1],marker='s',markersize=7, color='red')[0]
    p2 = plt.plot(X[neg,0],X[neg,1],marker='o',markersize=7,color='green')[0]
    return p1 ,p2




def normalization(X):
    Xmin=np.min(X,axis=0)
    Xmax=np.max(X,axis=0)
    Xmu=np.mean(X,axis=0)
    X_norm=(X-Xmu)/(Xmax-Xmin)
    return X_norm






def plotDecisionBoundary(trainX,trainY,w,iter_num=0):
    xcord1=[];ycord1=[];xcord2=[];ycord2 = []
    m,n=np.shape(trainX)
    for i in range(m):
        if trainY[i] == 1:
            xcord1.append(trainX[i,1])
            ycord1.append(trainX[i,2])
        else:
            xcord2.append(trainX[i,1])
            ycord2.append(trainX[i,2])
    x_min=min(trainX[:,2])
    y_min=min(trainX[:,1])
    x_max=max(trainX[:,1])
    y_max=max(trainX[:,2])
    fig =  plt.figure(1)
    plt.scatter(xcord1,ycord1,s=30,c='red',marker='s',label='I like you')
    plt.scatter(xcord2,ycord2,s=30,c='green',marker='o',label='I dont like you')
    plt.legend(loc='upper right')
    
    delta_x=x_max-x_min
    delta_y=y_max-y_min
    my_x_ticks =np.arange(x_min - delta_x/10,x_max + delta_x/10,1)
    my_y_ticks =np.arange(y_min - delta_y/10,y_max + delta_y/10,1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.axis([x_min-delta_x/10,x_max+delta_x/10,y_min-delta_y/10,y_max+delta_y/10])
    x=np.arange(x_min-delta_x/10,x_max+delta_x/10,0.01)
    y=(-w[0]-w[1]*x)/w[2]
    plt.plot(x,y.T)
    fig_name='Traning'+str(iter_num)+'times.png'
    plt.title(fig_name)
    fig.savefig(fig_name)
    plt.show(fig)
    
    






def sigmoid(wx):
    sigmoidV=1.0/(1.0+np.exp(-wx))
    return sigmoidV




def loss(X,Y,w):
    m,n=np.shape(X)
    trainMat=np.mat(X)
    Y_=[]
    for i in np.arange(m):
        Y_.append(sigmoid(trainMat[i]*w))
    m=np.shape(Y_)[0]
    sum_err = 0.0
    
    for i in range(m):
        sum_err -= Y[i]*np.log(Y_[i])+(1-Y[i])*np.log(1-Y_[i])
        
        return sum_err/m




def BGD(X,y,iter_num,alpha):
    trainMat=np.mat(X)
    trainY=np.mat(y).T
    m,n=np.shape(X)
    w=np.ones((n,1))
    for i in range(iter_num):
        error =sigmoid(trainMat*w)-trainY
        w=w - (1.0/m)*alpha*trainMat.T*error
    return w




def classify(wx):
    prob = sigmoid(wx)
    if prob > 0.5:
        return 1
    else:
        return 0




def predict(testX,w):
    m,n=np.shape(testX)
    testMat=np.mat(testX)
    result = []
    for i in np.arange(m):
        result.append(classify(float(testMat[i]*w)))
    return result




def precision(X,Y,w):
    result = predict(X,w)
    right_sum=0
    for i in range(len(result)):
        if result[i]-int(Y[i])==0:
            right_sum +=1
    return 1.0*right_sum/len(Y)




if __name__ == '__main__':
    data = loadData('data.xls')                     
    X = data[:,:2]                                  
    y = data[:,2]                                   
    
    
    plt_data = plt.figure(1)
    p1, p2 = plotData(X, y)                    

    
    plt.xlabel('tall')                              
    plt.ylabel('salary')                            
    
    
    plt.legend((p1, p2), ('I like you', "I don't like you"), numpoints=1, handlelength=0)
    plt_data.savefig('visualization_org.png')
    plt.show(plt_data)
    plt.close(plt_data)
    
    X_norm = normalization(X)
    
    plt_norm = plt.figure(1)
    
    p1_norm, p2_norm = plotData(X_norm, y)

    
    plt.xlabel('tall')                         
    plt.ylabel('salary')                       
    
    
    plt.legend((p1_norm, p2_norm), ('I like you', "I don't like you"), numpoints=1, handlelength=0)
    plt.show(plt_norm)
    plt_norm.savefig('visualization_norm.png')
    plt.close(plt_norm) 
    
        
    iter_num=200                                    
    lr=0.05                                        
    m,n = np.shape(data)                            
    offset = np.ones((m, 1))                        
    trainMat = np.c_[offset, X_norm]                
    theta=BGD(trainMat,y,iter_num,lr)               

    
    
    plotDecisionBoundary(trainMat, y, theta, iter_num)
    cost = loss(trainMat, y, theta)                 
    print('Cost theta: {0}'.format(cost))           

    
    p = precision(trainMat, y, theta)               
    print('Train Accuracy: {0}'.format(p))          
    print('finished!')                              
    
    








