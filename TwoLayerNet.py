import numpy as np
from PIL import Image

class TowLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        print("初期化中")
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
    
    def step(self,x):
        '''ステップ関数'''
        y=x>0
        return y.astype(np.int)

    def sigmoid(self,x):
        '''シグモイド関数'''
        return 1/(1+np.exp(-x))
    
    def ReLU(self,x):
        '''ReLU関数'''
        return np.maximum(0,x)

    def softmax(selx,x):
        '''ソフトマックス関数'''
        c=np.max(x)
        exp_x=np.exp(x-c)
        sum_exp_x=np.sum(exp_x)
        y=exp_x/sum_exp_x
        return y
    
    def meanSquaredError(self,y,t):
        '''
        2乗和誤差
        y:予想ラベル(ソフトマックス関数の出力)
        t:正解ラベル
        return:小さいほど正確
        '''
        return 0.5*np.sum((y-t)**2)

    def crossEntropyError(self,y,t):
        '''
        交差エントロピー誤差
        y:予想ラベル(ソフトマックス関数の出力)
        t:正解ラベル
        return:0に近いほど正確
        '''
        delta=1e-7
        if y.ndim==1:
            t=t.reshape(1,t.size)
            y=y.reshape(1,y.size)
        batch_size=y.shape[0]
        return -np.sum(t*np.log(y+delta))/batch_size

    def numericalDiff(self,f,x):
        '''
        数値微分
        f:数値微分したい関数
        x:x
        '''
        h=1e-4
        return (f(x+h)-f(x-h))/(2*h)

    def numericalGradientFunc(self,f,x):
        '''
        数値偏微分(勾配)
        f:勾配を求めたい関数
        x:x
        '''
        h=1e-4
        grad=np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx=it.multi_index
            print(str(idx)+"の要素の勾配の計算を開始")
            tmp_val=x[idx]
            x[idx]=float(tmp_val)+h
            fxh1=f(x)
            x[idx]=float(tmp_val)-h
            fxh2=f(x)
            grad[idx]=(fxh1-fxh2)/(2*h)
            print("勾配 : "+str(grad[idx]))
            x[idx]=tmp_val
            it.iternext()
        return grad

    def numericalGradient(self,x,t):
        print("勾配の計算を開始")
        loss_W=lambda W:self.loss(x,t)
        grads={}
        print("W1の勾配の計算を開始")
        grads['W1']=self.numericalGradientFunc(loss_W,self.params['W1'])
        print("b1の勾配の計算を開始")
        grads['b1']=self.numericalGradientFunc(loss_W,self.params['b1'])        
        print("W2の勾配の計算を開始")
        grads['W2']=self.numericalGradientFunc(loss_W,self.params['W2'])
        print("b2の勾配の計算を開始")
        grads['b2']=self.numericalGradientFunc(loss_W,self.params['b2'])
        return grads

    def numericalDescent(self,f,init_x,lr=0.1,step_num=100):
        '''
        勾配法
        f:関数
        lr:学習率
        '''
        x=init_x
        for i in range(step_num):
            grad=self.numericalGradientFunc(f,x)
            x-=lr*grad
        return x

    def predict(self,x):
        '''
        推論を行う
        '''
        W1=self.params['W1']
        W2=self.params['W2']
        b1=self.params['b1']
        b2=self.params['b2']
        a1=np.dot(x,W1)+b1
        z1=self.sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=self.softmax(a2)
        return y

    def loss(self,x,t):
        y=self.predict(x)
        return self.crossEntropyError(y,t)

    def load_train(self,path,data_max,teach):
        '''
        教師データの読み込み
        path:画像までのパス
        data_max:読みこむ画像数
        teach:正解ラベル(np.array)
        '''
        x_train=[]
        t_train=[]
        for i in range(data_max):
            img_pixels = []
            img = Image.open(path+str(i)+'.jpg')
            gray_img = img.convert('L')
            width, height = gray_img.size
            for y in range(height):
                for x in range(width):
                    img_pixels.append(gray_img.getpixel((x,y))/255.0)
            img_pixels = np.array(img_pixels)
            x_train.append(img_pixels)
            t_train.append(teach)
        return x_train,t_train
