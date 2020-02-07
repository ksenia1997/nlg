1model.pt parameters: 
* SEED = 5 
* CLIP = 10 
* N_EPOCHS = 20 
* DATA_TYPE = "PERSONA"  
* WITHOUT ATTENTION 
* BATCH_SIZE = 5 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 256 
* DROPOUT_RATE = 0.1 
* NUM_LAYERS = 4  
* PREPROCESS = False

Trained on Persona Chat with description and all communications. 

2model.pt parameters:

* SEED = 5 
* CLIP = 10 
* N_EPOCHS = 10
* DATA_TYPE = "PERSONA"  
* WITH ATTENTION 
* BATCH_SIZE = 5 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 256 
* DROPOUT_RATE = 0.1 
* NUM_LAYERS = 1 
* PREPROCESS = False

3model.pt parameters: 

* SEED = 5
* CLIP = 10
* N_EPOCHS = 10
* DATA_TYPE = "PERSONA"  
* WITHOUT ATTENTION 
* BATCH_SIZE = 5 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 512 
* DROPOUT_RATE = 0.5 
* NUM_LAYERS = 1
* PREPROCESS = False

with changed preprocessing  


4model.pt parameters: 

* SEED = 5
* CLIP = 10
* N_EPOCHS = 10
* DATA_TYPE = "PERSONA"  
* WITHOUT ATTENTION 
* BATCH_SIZE = 5 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 512 
* DROPOUT_RATE = 0.5 
* NUM_LAYERS = 1
* PREPROCESS = False

changed preprocess: 
1. person1: persona # persona # CC SS
2. person2: start conversation
3. person1: persona # persona # CC SS start conversation #
4. person2: answer to person1



5model.pt parameters:
* SEED = 5  # set seed value for deterministic results
* CLIP = 10
* N_EPOCHS = 15
* DATA_TYPE = "PERSONA_BOTH"  
* WITH_DESCRIPTION = True
* WITH_ATTENTION = False
* IS_BEAM_SEARCH = False
* PREPROCESS = False
* BATCH_SIZE = 10 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 512 
* DROPOUT_RATE = 0.1
* NUM_LAYERS = 1

Without teacher forcing

tensorboard --logdir ./tensorboard-logs/train_3model/
