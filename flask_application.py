#import Flask 
from flask import Flask, render_template, request
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import csv 
from scipy.spatial import distance

import nltk
from newspaper import Article

import numpy as np
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

def documentvec(word2vec_model,summarywords):
    "This function Creates a document vector by taking the mean of word vectors of the words in the document"
    k=[]
    for i in range(len(summarywords)): 
        if summarywords[i] in word2vec_model.wv:#model.wv.vocab gives the entire word vocabulary 
            k.append(word2vec_model.wv[summarywords[i]])#of the generated model upon the given dataset
    return np.mean(k,axis=0)


@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    print(request)
    if request.method == "POST":
    
        test_summary = request.form.get('otherField1')
        if test_summary == "":
            url = request.form.get('otherField11')
            article = Article(url)
            article.download()
            article.parse()
            nltk.download('punkt')
            article.nlp()
            test_summary = article.summary
            print(test_summary)
            # pca_df = pd.read_csv(r"Vector_list.csv")
            # pca_df.head()


            # x = pca_df.iloc[:, 0:100].values 
            # y = pca_df.iloc[:, 100].values 


            # x = StandardScaler().fit_transform(x) # normalizing the features

            # x.shape

            # np.mean(x),np.std(x)

            # feat_cols = ['feature'+str(i) for i in range(x.shape[1])]

            # normalised_breast = pd.DataFrame(x,columns=feat_cols)

            # normalised_breast.tail()

            # from sklearn.decomposition import PCA
            # pca_breast = PCA(n_components=2)
            # principalComponents_breast = pca_breast.fit_transform(x)

            # principal_breast_Df = pd.DataFrame(data = principalComponents_breast
            #             , columns = ['principal component 1', 'principal component 2'])

            # principal_breast_Df

            # print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

            # plt.figure()
            # plt.figure(figsize=(10,10))
            # plt.xticks(fontsize=12)
            # plt.yticks(fontsize=14)
            # plt.xlabel('Principal Component - 1',fontsize=20)
            # plt.ylabel('Principal Component - 2',fontsize=20)
            # plt.title("Principal Component Analysis of Fake News Dataset",fontsize=20)
            # targets = [1,0]
            # Classes = ["True","False"]
            # colors = ['g','r']
            # for target, color in zip(targets,colors):
            #     indicesToKeep = pca_df['Class'] == target
            #     plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
            #             , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 100)

            # plt.legend(Classes,prop={'size': 15})
                
        try:
            prediction = preprocessDataAndPredict(test_summary)    #pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
        pass
    pass

def preprocessDataAndPredict(test_summary):
    
    model = Word2Vec.load("word2vec.model")
    with open('Final_Vectors.csv', newline='') as f:
        reader = csv.reader(f)
        True_vector = next(reader)  # gets the first line
        False_vector = next(reader)
    true_vector_floats = []
    false_vector_floats = []
    for item in True_vector:
        true_vector_floats.append(float(item))
    for item in False_vector:
        false_vector_floats.append(float(item))

    test_summary_words = test_summary.split(' ')
    corpus2 = []
    corpus3 = []
    category = {}
    corpus2.append(test_summary_words)
    model.build_vocab(corpus2, update = True)
    model.train(corpus2, total_examples=2, epochs = 1)
    test_vector = documentvec(model,test_summary_words)
    corpus3.append(test_vector)
    print(test_vector)
    true_class  = 1 - distance.cosine(test_vector,true_vector_floats)
    false_class  = 1 - distance.cosine(test_vector,false_vector_floats)
    if true_class > false_class:
        category[0] = 1
        category[1] = true_class
        category[2] = false_class
        category[3] = 0
    else:
        category[0] = 0
        category[1] = false_class
        category[2] = true_class
        category[3] =1

    # #open file
    # file = open("logisticregression_model.pkl","rb")
    
    # #load trained model
    # trained_model = joblib.load(file)
    
    # #predict
    # category = trained_model.predict(corpus3)
    # print(category)
    return category
    pass

if __name__ == '__main__':
    app.run(host="localhost", port=5050, debug=True)