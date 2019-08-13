
 def build_discriminator(self):
        input_layer = Input(shape=(None,None,2))
        condition_layer = Input(shape=(None,None,1))
        hid = Concatenate()([input_layer,condition_layer])
        #hid = Multiply()([input_layer,condition_layer])
        hid = SeparableConv2D(128, kernel_size=5, strides=1, padding='same')(input_layer)
        hid = BatchNormalization(momentum=0.2)(hid)
        hid = Dropout(0.4)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)

        hid = SeparableConv2D(128, kernel_size=5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.2)(hid)
        hid = Dropout(0.4)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)

        hid = SeparableConv2D(128, kernel_size=5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.2)(hid)
        hid = Dropout(0.4)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)

        hid = GlobalAveragePooling2D()(hid)
        hid = Dense(256, activation='relu')(hid)
        hid = Dropout(0.4)(hid)
        out = Dense(1, activation='sigmoid')(hid)
        model = Model(inputs=[input_layer, condition_layer], outputs=out)
        print("--Discriminator--")
        model.summary()
        return model
