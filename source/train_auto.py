import time
from galaxyencode import GalaxyEncoder
import os
import tensorflow as tf
from zoobot.data_utils import tfrecord_datasets
from zoobot.estimators import auto_preprocess as preprocess
from zoobot import schemas, label_metadata

question_answer_pairs = label_metadata.decals_pairs
dependencies = label_metadata.get_gz2_and_decals_dependencies(question_answer_pairs)
schema = schemas.Schema(question_answer_pairs, dependencies)

batch_size = 256
train_records_dir = "/scratch3/users/ezraf/lengau_auto_shards/train_shards"
eval_records_dir = "/scratch3/users/ezraf/lengau_auto_shards/eval_shards"

train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

raw_train_dataset = tfrecord_datasets.get_dataset(train_records, schema.label_cols, batch_size, shuffle=True)
raw_test_dataset = tfrecord_datasets.get_dataset(eval_records, schema.label_cols, batch_size, shuffle=False)


preprocess_config = preprocess.PreprocessingConfig(
    label_cols=schema.label_cols,
    input_size=224,
    make_greyscale=False,
    normalise_from_uint8=True
)

train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
test_dataset = preprocess.preprocess_dataset(raw_test_dataset, preprocess_config)

model = GalaxyEncoder()
model.compile(optimizer='sgd', loss='mse')
start = time.time()
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=100,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="/idia/projects/hippo/gzd/autoencoder/logs")]
)
end = time.time()
diff = end - start
print('End Time: ', end)
print('Time Diff: ', diff)
model.save("/idia/projects/hippo/gzd/autoencoder/model_save")
