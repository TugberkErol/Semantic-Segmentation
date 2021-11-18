from tensorflow import keras
    
def skip_connection(x, pooling = False, upscale = False, scale = 1):
    if pooling:
        x = keras.layers.MaxPooling2D(pool_size = (scale,scale))(x)
    if upscale:
        x = keras.layers.UpSampling2D(size = (scale, scale))(x)
    x = conv_blockx1(x, 16)
    return x
            
def conv_blockx1(x, filters):
    x = keras.layers.Conv2D(filters, kernel_size=(3,3), padding = 'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x
    
def conv_blockx2(x, filters):
    x = keras.layers.Conv2D(filters, kernel_size = (3,3), padding = 'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters, kernel_size = (3,3), padding = 'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    return x


class unet3p():

    
    def build_model(self, num_filters, input_shape):
        inputs = keras.Input(input_shape)
        encoder1 = conv_blockx2(inputs, num_filters[0])
        encoder2 = keras.layers.MaxPooling2D((2,2))(encoder1)
        encoder2 = conv_blockx2(encoder2, num_filters[1])
        encoder3 = keras.layers.MaxPooling2D((2,2))(encoder2)
        encoder3 = conv_blockx2(encoder3, num_filters[2])
        encoder4 = keras.layers.MaxPooling2D((2,2))(encoder3)
        encoder4 = conv_blockx2(encoder4, num_filters[3])
        encoder5 = keras.layers.MaxPooling2D((2,2))(encoder4)
        encoder5 = conv_blockx2(encoder5, num_filters[4])
        
        #Concatenate decoder4
        #Concatenate()([encoder1,encoder2, encoder3, encoder4, encoder5])
        decoder4_encoder1 = skip_connection(encoder1, pooling = True, upscale = False, scale = 8)
        decoder4_encoder2 = skip_connection(encoder2, pooling = True, upscale = False, scale = 4)
        decoder4_encoder3 = skip_connection(encoder3, pooling = True, upscale = False, scale = 2)
        decoder4_encoder4 = skip_connection(encoder4, pooling = False, upscale = False, scale = 1)
        decoder4_encoder5 = skip_connection(encoder5, pooling = False, upscale = True, scale = 2)
        
        decoder4 = keras.layers.Concatenate()([decoder4_encoder1, decoder4_encoder2, decoder4_encoder3, decoder4_encoder4, decoder4_encoder5])
        
        #Concatenate decoder3 
        #Concatenate()([encoder1, encoder2, encoder3, encoder5, decoder4])
        decoder3_encoder1 = skip_connection(encoder1, pooling = True, upscale = False, scale = 4)
        decoder3_encoder2 = skip_connection(encoder2, pooling = True, upscale = False, scale = 2)
        decoder3_encoder3 = skip_connection(encoder3, pooling = False, upscale = False, scale =1)
        decoder3_encoder5 = skip_connection(encoder5, pooling = False, upscale = True, scale = 4)
        decoder3_decoder4 = skip_connection(decoder4, pooling = False, upscale = True, scale = 2)
        
        decoder3 = keras.layers.Concatenate()([decoder3_encoder1, decoder3_encoder2, decoder3_encoder3, decoder3_encoder5, decoder3_decoder4])
        
        
        #Concatenate decoder2
        #Concatenate()([encoder1, encoder2, encoder5, decoder4, decoder3])
        decoder2_encoder1 = skip_connection(encoder1, pooling = True, upscale = False, scale = 2)
        decoder2_encoder2 = skip_connection(encoder2, pooling = False, upscale = False, scale = 1)
        decoder2_encoder5 = skip_connection(encoder5, pooling = False, upscale = True, scale = 8)
        decoder2_decoder4 = skip_connection(decoder4, pooling = False, upscale = True, scale = 4)
        decoder2_decoder3 = skip_connection(decoder3, pooling = False, upscale = True, scale = 2)
        
        decoder2 = keras.layers.Concatenate()([decoder2_encoder1, decoder2_encoder2, decoder2_encoder5, decoder2_decoder4, decoder2_decoder3])
        

        #Concatenate decoder1
        #Concatenate()([encoder1, encoder5, decoder4, decoder3, decoder2])
        decoder1_encoder1 = skip_connection(encoder1, pooling = False, upscale = False, scale = 1)
        decoder1_encoder5 = skip_connection(encoder5, pooling = False, upscale = True, scale = 16)
        decoder1_decoder4 = skip_connection(decoder4, pooling = False, upscale = True, scale = 8)
        decoder1_decoder3 = skip_connection(decoder3, pooling = False, upscale = True, scale = 4)
        decoder1_decoder2 = skip_connection(decoder2, pooling = False, upscale = True, scale = 2)
        
        decoder1 = keras.layers.Concatenate()([decoder1_encoder1, decoder1_encoder5, decoder1_decoder4, decoder1_decoder3, decoder1_decoder2])
        
        outputs = keras.layers.Conv2D(1, (3,3), padding = 'same')(decoder1)
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.Activation('sigmoid')(outputs)
        model = keras.models.Model(inputs, outputs)
        return model        
