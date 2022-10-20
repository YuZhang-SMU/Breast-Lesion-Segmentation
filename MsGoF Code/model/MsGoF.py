
from keras.layers import Average, Concatenate,GlobalMaxPooling2D,Dense
from keras.layers import Input
from keras.models import Model
from module.SCA import SCA1, SCA2, SCA3
from module.FFM import FFM
from module.SFM import SFM
from module.TFM import TFM


def fusion1(x1, x2, x3):
    f1 = Average()([x1, x2])
    f2 = Average()([x2, x3])
    f3 = Average()([x3, x1])
    return f1, f2, f3

def MsGoF(input_size):
    Input_ = Input(shape=(input_size, input_size, 1))
    # SCA
    SG1 = SCA1(Input_)
    SG2 = SCA2(Input_)
    SG3 = SCA3(Input_)
    # FFM
    X1w1 = FFM(SG1, 16)
    X2w1 = FFM(SG2, 16)
    X3w1 = FFM(SG3, 16)
    f11, f12, f13 = fusion1(X1w1, X2w1, X3w1)  # 两两交互fusion1
    # SFM
    X1w2 = SFM(f11, 32)
    X2w2 = SFM(f12, 32)
    X3w2 = SFM(f13, 32)
    f21, f22, f23 = fusion1(X1w2, X2w2, X3w2)
    # TFM
    X1w3 = TFM(f21, 32)
    X2w3 = TFM(f22, 32)
    X3w3 = TFM(f23, 32)
    # fusion
    Scale1 = GlobalMaxPooling2D()(X1w3)
    Scale2 = GlobalMaxPooling2D()(X2w3)
    Scale3 = GlobalMaxPooling2D()(X3w3)
    Feature_All = Concatenate(name='feature_all')([Scale1, Scale2, Scale3])
    Output_1 = Dense(1, activation='sigmoid', name='Output_1')(Feature_All)
    model = Model(inputs=[Input_], outputs=[Output_1])
    return model

