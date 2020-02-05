1model.pt parameters: 
* SEED = 5 
* CLIP = 10 
* N_EPOCHS = 20 
* WITHOUT ATTENTION 
* BATCH_SIZE = 5 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 256 
* DROPOUT_RATE = 0.1 
* NUM_LAYERS = 4  

Trained on Persona Chat with description and all communications. 

2model.pt parameters:

* SEED = 5 
* CLIP = 10 
* N_EPOCHS = 10
* WITH ATTENTION 
* BATCH_SIZE = 5 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 256 
* DROPOUT_RATE = 0.1 
* NUM_LAYERS = 1 


3model.pt parameters: 

* SEED = 5
* CLIP = 10
* N_EPOCHS = 10
* WITHOUT ATTENTION 
* BATCH_SIZE = 5 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 512 
* DROPOUT_RATE = 0.5 
* NUM_LAYERS = 1

with changed preprocessing  


4model.pt parameters: 

* SEED = 5
* CLIP = 10
* N_EPOCHS = 10
* WITHOUT ATTENTION 
* BATCH_SIZE = 5 
* EMBEDDING_DIM = 100 
* HIDDEN_DIM = 512 
* DROPOUT_RATE = 0.5 
* NUM_LAYERS = 1

changed preprocess: 
1. person1: persona # persona # CC SS
2. person2: start conversation
3. person1: persona # persona # CC SS start conversation #
4. person2: answer to person1
