#open dataset files
file1 = open('/Users/dev/anshul/sentiment labelled sentences/imdb_labelled.txt', 'r')
file2 = open('/Users/dev/anshul/sentiment labelled sentences/amazon_cells_labelled.txt', 'r')
file3 = open('/Users/dev/anshul/sentiment labelled sentences/yelp_labelled.txt', 'r')

import pandas as pd
   # Create DataFrame
df1 = pd.DataFrame(columns=['tweet'])
df2 = pd.DataFrame(columns=['tweet'])
df3 = pd.DataFrame(columns=['tweet'])
label=[]
for Lines in file1.readlines():
    Lines=Lines.strip()
    df1.loc[len(df1.index), 'tweet'] = Lines[:-1]
    label.append(Lines[-1])
    
df1=df1.assign(label=label)

label=[]
for Lines in file2.readlines():
    Lines=Lines.strip()
    df2.loc[len(df2.index), 'tweet'] = Lines[:-1]
    label.append(Lines[-1])
    
df2=df2.assign(label=label)

label=[]
for Lines in file3.readlines():
    Lines=Lines.strip()
    df3.loc[len(df3.index), 'tweet'] = Lines[:-1]
    label.append(Lines[-1])
    
df3=df3.assign(label=label)

#merge dataframes
df=pd.concat([df1,df2,df3])


from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#transform textual tweets into numerical vectors
tv = TfidfVectorizer(input='content',decode_error='ignore',ngram_range=(1, 1),min_df=0.01)#,stop_words='english')
t=tv.fit(df["tweet"])
t=tv.transform(df["tweet"])
print(t.shape)


#apply first step of dimensionality reduction
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, chi2
sf=chi2
mi = SelectPercentile(chi2, percentile=50)
t=mi.fit_transform(t,df['tweet'])
print(t.shape)

#apply second step of dimensionality reduction
from sklearn.decomposition import PCA
# Create a PCA that will retain 99% of variance
pca = PCA(n_components=0.99, whiten=True)
# apply PCA
t = pca.fit_transform(t.toarray())
print(t.shape)


scoring='accuracy'
#creating models
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(C=10)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C = 1)

for model in [svm,gnb,logreg,clf]:
    print(model)


    #applying ten fold cross validation
    scores=cross_val_score(model,t,df['label'],cv=f,scoring=scoring)
    print("\n"+scoring+"\n")
    print("CV =:"+str(f))
    print("\nCV Scores: \n{}".format(scores.mean()*100))








    
