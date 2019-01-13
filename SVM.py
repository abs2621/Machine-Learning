import numpy as np
import sklearn.svm
from sklearn.svm import SVC
import random
import pandas as pd
from scipy import stats

def fold(data, currenti, kfold):
	data=np.array(data)
	foldSize = len(data)//kfold
	start = currenti*foldSize
	end = start + foldSize
	testI  = list(range(start, end))
	trainI = list(range(0, start)) + list(range(end, len(data)))
	#print(trainI)
	#print(testI) 
	training = data[trainI]
	testing = data[testI]
	#print(training.reshape(-1))
	#print(testing.reshape(-1))
	#print("TRAINING", training)
	#print("TESTING", testing)
	training=pd.DataFrame(training)
	testing=pd.DataFrame(testing)
	return (training, testing)



class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)
    def load_data(self, csv_fpath):
        training_data = pd.read_csv(training_csv,skipinitialspace=True)
   
        training_data=np.array(training_data)
        training_data=pd.DataFrame(training_data)


        other_list = [1,3,5,6,7,8,9,13]
        int_list = [0,2,4,10,11,12]
        #print("shape: ",np.shape(training_data)[1])
        modes_training =(training_data.mode().iloc[0])
        #print("MEANS: ",training_data.mean())
        #print("Mode: ", training_data.mode())
        mean_training=(training_data.mean())


        for x in other_list:
            training_data[x]=training_data[x].str.strip()
            training_data[x]=training_data[x].replace({"?":modes_training[x]})
        for x in int_list:
            training_data[x]=training_data[x].replace({"?":mean_training[x]})    
        #training set labels
        training_data[14]=training_data[14].str.strip()
        training_data[14]=training_data[14].map({'<=50K':1,'>50K':-1})

        #testing set labels

        for t in other_list:
            #training (changing categorical data to ints)
            training_data[t]=pd.Categorical(training_data[t])
            training_data[t]=training_data[t].cat.codes





        for x in int_list:
            training_data[x]=(training_data[x]-training_data[x].min())/(training_data[x].max()-training_data[x].min())
        #print("SHAPExxx :", np.shape(training_data.loc[:,0:13]),"SHAPExxx :",np.shape(training_data.loc[:,14]),"\n")


        return training_data 
		

		
    def train_and_select_model(self, training_csv):
        param_set = [
                     {'kernel': 'poly', 'C': 1, 'degree': 1},
                    
        ]
        
       
        training_data = self.load_data(training_csv)

        score_table=[]
        best_option=param_set[0]
        best_option_score =0
        for param in param_set:
            current_score=0
            
            #print("=============================================================================")
            #print("NEW PARAM", param)
            #kfold
            for i in range(0,3):
                #print("fold: ",i)
                training_list=[]
                score=0
                #print("i", i)
                #get current data
                current_train, current_test = fold(training_data,i ,3)
                #print("size of traitnitng list: ",np.shape(current_train))
                current_SVM = SVC(kernel=param['kernel'], C=param['C'],degree=param['degree'])
                #run fit on training data
                x=np.array(current_train.loc[:,0:13])
                y=np.array(current_train.loc[:,14])
                y=y.astype('float')
                x=x.astype('float')
                test_vals=current_test.loc[:,14]
               # print("stuff: ",current_test.loc[2,0:13])
               # print("yyyyyy: ",y)
               # print("xxxxxx",x)
                current_SVM.fit(x,y)
                #run training through each testing point
                #print("AFTERFIT")
                for t in range(len(current_test)):
                    training_list.append(current_SVM.predict([current_test.loc[t,0:13]]))
                training_list=pd.DataFrame(training_list)
                training_list=np.array(training_list)
                for t in range(len(training_list)):
                    #print("SHAPE TRAINING_LIST:",np.shape(training_list)," testinglistshape:",np.shape(test_vals))
                    #print("TRAINING_LIST",pd.DataFrame(training_list))
                    #print("Testing_list",test_vals)
                    if(training_list[t]==test_vals[t]):
                        score+=1
                current_score+=(score/len(training_list))
                #print("Score: ",score, " weighted: ",score/len(training_list))
                #print("training_list:",training_list)
                #print("testing_list:",training_list)
            current_score=current_score/3
            if(current_score>best_option_score):
                #print("NEW BEST")
                best_option=param
                best_option_score=current_score
                best_model=current_SVM
            score_table.append([current_score,param])
            #print("Best Option:",best_option, " current_score= ",current_score)           
                #run svm, 
                #check output agains test[14]
                #current_score=current_score/3





        print(score_table)
    

        return best_model, best_option_score

    def predict(self, test_csv, trained_model):
        x_test = self.load_data(test_csv)
        predictions = trained_model.predict(x_test.loc[:,0:13])
        print("model: ", trained_model)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    #training, testing = readData(training_csv, testing_csv)
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    print("The best model was scored", cv_score)
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)
    #clf.output_results(predictions)

#data is dirty -> some are ? categorical data(replace with mode)
#floating point values (numerical mean)
#iterate over parameter sets  kernal types RBF/linear 
