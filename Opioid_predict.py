import pandas as pd
import pickle
from keras.layers import Tokenizer




class ProcessData:

    def dffiles(self, directory):
        txtfilesList = []
        patientL = []
        count = 0
        for file in os.listdir(directory):
            try:
                if file.endswith(".txt"):
                    txtfilesList.append(directory + str(file))
                    patientL.append(str(file))
                    count = count + 1
                else:
                    print("Not a text file")

            except Exception as e:
                raise e
                print("No Files found here!")

         print("Total Files found", count)
         txtfilesTrain_df = pd.DataFrame({'Filename':patientL, 'FileList':txtfilesList})
         return txtfilesTrain_df

    def openData(self, text_df):
        print(text_df)
        text_df['FileList'] = text_df.FileList.apply(lambda x: open(x, "r").read())
        text_df['FileList'] = text_df.FileList.apply(lambda x: ", ".join(x.split( )))
        return text_df


    def predict(self, tokenizer_loc, model, text_df, outputDir):
        model = load_model(model)
        tokenizer_pkl = open(tokenizer_loc, 'rb')
        tokenizer = pickle.load(tokenizer_pkl)
        x_test = text_df.FileList
        sequences_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(sequences_val, maxlen = 35000)
        predict_prob = model.predict(np.asarray(x_test), batch_size = 1)
        prediction = []
        predictProb = []
        for x in predict_prob:
            predictProb.append(x[0])
            if x[0] < 0.5:
                prediction.append(0)
            else:
                prediction.append(1)
        
        documents = text_df.Filename.tolist()
        prob_df = pd.DataFrame({'Filename': documents, 'Predictions': prediction, 'Prediction_Probability': predictProb})
        data_df.to_csv(outputDr + 'OpioidResult.csv', sep = '|', index = False)
        return data_df
        
        
if __name__ == "__main__":
    PD = ProcessData()
    
    inputDir = "/Archive/"
    outputDir = "/home/"

    txtfilesTrain_df = PD.dfffiles(inputDir)
    text_df = PD.openData(txtfilesTrain_df)
    data_df = PD.predict(tokenizer_loc, model_loc, text_df, outputDir)
