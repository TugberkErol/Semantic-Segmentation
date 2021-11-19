from tensorflow import keras


def conv_blockx2(x, filters):
    for _ in range(2):
        x = keras.layers.Conv2D(filters, (3,3), padding = 'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        return x
    
class unet:
          
    def build_model(self,num_filters,input_size):

        inputs = keras.Input(input_size)
        encoder1 = conv_blockx2(inputs, num_filters[0])
        encoder2 = keras.layers.MaxPooling2D((2,2))(encoder1)
        encoder2 = conv_blockx2(encoder2, num_filters[1])
        encoder3 = keras.layers.MaxPooling2D((2,2))(encoder2)
        encoder3 = conv_blockx2(encoder3, num_filters[2])
        encoder4 = keras.layers.MaxPooling2D((2,2))(encoder3)
        encoder4 = conv_blockx2(encoder4, num_filters[3])
        encoder5 = keras.layers.MaxPooling2D((2,2))(encoder4)
        encoder5 = conv_blockx2(encoder5, num_filters[4])
        up4 = keras.layers.UpSampling2D((2,2))(encoder5)
        decoder4 = keras.layers.Concatenate()([encoder4, up4])
        decoder4 = conv_blockx2(decoder4, num_filters[3])
        up3 = keras.layers.UpSampling2D((2,2))(decoder4)
        decoder3 = keras.layers.Concatenate()([encoder3, up3])
        decoder3 = conv_blockx2(decoder3, num_filters[2])
        up2 = keras.layers.UpSampling2D((2,2))(decoder3)
        decoder2 = keras.layers.Concatenate()([encoder2, up2])        
        decoder2 = conv_blockx2(decoder2, num_filters[1])
        up1 = keras.layers.UpSampling2D((2,2))(decoder2)
        decoder1 = keras.layers.Concatenate()([encoder1, up1])
        decoder1 = conv_blockx2(decoder1, num_filters[0])
        outputs = keras.layers.Conv2D(1, 1, activation= 'sigmoid')(decoder1)
        
        model = keras.models.Model(inputs, outputs)
        return model

u = unet()
model = u.build_model(num_filters = [32,64,128,256,512], input_size = (224,224,3))
model.summary()
#Total parameter size = 3.925.057