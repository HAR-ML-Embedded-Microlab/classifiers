import hls4ml
import tensorflow as tf
model = tf.keras.models.load_model("work/model_trained")
config = hls4ml.utils.config_from_keras_model(model, granularity='model')
config['Model']['Strategy'] = 'Resource'
config['Model']['ReuseFactor'] = 1024
hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir='work/hls4ml_model_trained', part='xczu7cg-fbvb900-2-i', io_type='io_stream')
hls_model.build(csim=False)
