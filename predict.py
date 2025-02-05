import tensorflow as tf
import data
import sys
import model as ml

from configs import DEFINES
	
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    arg_length = len(sys.argv)
    
    if(arg_length < 2):
        raise Exception("Don't call us. We'll call you")
  
    
    train_q, train_a, test_q, test_a = data.load_data()

    token_train_q = data.tokenizing_data(train_q)
    token_train_a = data.tokenizing_data(train_a)

    token_test_q = data.tokenizing_data(test_q)
    token_test_a = data.tokenizing_data(test_a)

    char2idx, idx2char, vocabulary_length = data.load_voc(token_train_q, token_train_a)

    input = ""
    for i in sys.argv[1:]:
        input += i 
        input += " "
        
    print(input)
    # 테스트셋 인코딩 / 디코딩 입력 / 디코딩 출력
    predic_input_enc = data.enc_processing([input], char2idx)
    predic_input_dec = data.dec_input_processing([""], char2idx)
    predic_target_dec = data.dec_target_processing([""], char2idx)

    # 에스티메이터 구성
    classifier = tf.estimator.Estimator(
            model_fn=ml.model,
            model_dir=DEFINES.check_point_path, 
            params={ 
                'hidden_size': DEFINES.hidden_size, 
                'layer_size': DEFINES.layer_size, 
                'learning_rate': DEFINES.learning_rate, 
                'vocabulary_length': vocabulary_length, 
                'embedding_size': DEFINES.embedding_size, 
                'embedding': DEFINES.embedding, 
                'multilayer': DEFINES.multilayer, 
            })

    predictions = classifier.predict(
        input_fn=lambda:data.eval_input_fn(predic_input_enc, predic_input_dec, predic_target_dec, 1))
    
    answer, finished = data.pred_next_string(predictions, idx2char)

    # 예측한 값을 인지 할 수 있도록
    # 텍스트로 변경
    print("answer: ", answer)
