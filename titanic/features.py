import re
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def processCabin():
    global df
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'U0'
    
    df['CabinLetter'] = df['Cabin'].map( lambda x : getCabinLetter(x))
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]

    cletters = pd.get_dummies(df['CabinLetter']).rename(columns=lambda x: 'CabinLetter_' + str(x))
    df = pd.concat([df, cletters], axis=1)

    df['CabinNumber'] = df['Cabin'].map( lambda x : getCabinNumber(x)).astype(int) + 1
    scaler = preprocessing.StandardScaler()
    df['CabinNumber_scaled'] = scaler.fit_transform(df['CabinNumber'])


def getCabinLetter(cabin):
    match = re.compile("([a-zA-Z]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 'U'


def getCabinNumber(cabin):
    match = re.compile("([0-9]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 0


def processTicket():    
    global df
    df['TicketPrefix'] = df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('[\.?\/?]', '', x) )
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )    

    df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]

    prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
    df = pd.concat([df, prefixes], axis=1)

    df.drop(['TicketPrefix'], axis=1, inplace=True)

    df['TicketNumber'] = df['Ticket'].map( lambda x: getTicketNumber(x) )
    df['TicketNumberDigits'] = df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)
    df['TicketNumberStart'] = df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)

    df['TicketNumber'] = df.TicketNumber.astype(np.int)
    
    scaler = preprocessing.StandardScaler()
    df['TicketNumber_scaled'] = scaler.fit_transform(df['TicketNumber'])


def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'U'

def getTicketNumber(ticket):
    match = re.compile("([\d]+$)").search(ticket)
    if match:
        return match.group()
    else:
        return '0'


def processFare():    
    global df
    df.loc[np.isnan(df['Fare']), 'Fare'] = df['Fare'].median()
    df.loc[np.where(df['Fare']==0)[0], 'Fare'] = df['Fare'][ df['Fare'].nonzero()[0] ].min() / 10
    df['Fare_bin'] = pd.qcut(df['Fare'], 4)
    
    df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)        
    df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]+1
    
    scaler = preprocessing.StandardScaler()
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'])
    
    df['perFare'] =  df['Fare']*1.0/(df['Family'])
    scaler = preprocessing.StandardScaler()
    df['perFare_scaled'] = scaler.fit_transform(df['perFare'])
        
    scaler = preprocessing.StandardScaler()
    df['Fare_bin_id_scaled'] = scaler.fit_transform(df['Fare_bin_id'])        
    df.drop('Fare_bin', axis=1, inplace=True)


def processEmbarked():    
    global df
    df.loc[ df.Embarked.isnull(), 'Embarked'] = df.Embarked.dropna().mode().values
    df['Embarked'] = pd.factorize(df['Embarked'])[0]
    df = pd.concat([df, pd.get_dummies(df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)



def processPClass():    
    global df
    df.loc[ df.Pclass.isnull(), 'Pclass'] = df.Pclass.dropna().mode().values
    df = pd.concat([df, pd.get_dummies(df['Pclass']).rename(columns=lambda x: 'Pclass_' + str(x))], axis=1)
    scaler = preprocessing.StandardScaler()
    df['Pclass_scaled'] = scaler.fit_transform(df['Pclass'])


def processFamily():    
    global df
    df['SibSp'] = df['SibSp'] + 1
    df['Parch'] = df['Parch'] + 1
    df['Family'] = df['SibSp'] + df['Parch'] + 1
    
    scaler = preprocessing.StandardScaler()
    df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'])
    df['Parch_scaled'] = scaler.fit_transform(df['Parch'])
    df['Family_scaled'] = scaler.fit_transform(df['Family'])

    sibsps = pd.get_dummies(df['SibSp']).rename(columns=lambda x: 'SibSp_' + str(x))
    parchs = pd.get_dummies(df['Parch']).rename(columns=lambda x: 'Parch_' + str(x))
    families = pd.get_dummies((df['Family']), prefix = 'Family')
    df = pd.concat([df, sibsps, parchs, families], axis=1)



def processSex():    
    global df
    df['Gender'] = np.where(df['Sex'] == 'male', 1, 0)



def processName():    
    global df
    df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    
    df.loc[df.Title == 'Jonkheer', 'Title'] = 'Master'
    df.loc[df.Title.isin(['Ms','Mlle']), 'Title'] = 'Miss'
    df.loc[df.Title == 'Mme', 'Title'] = 'Mrs'
    df.loc[df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir']), 'Title'] = 'Sir'
    df.loc[df.Title.isin(['Dona', 'Lady', 'the Countess']), 'Title'] = 'Lady'

    df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
    scaler = preprocessing.StandardScaler()
    df['Names_scaled'] = scaler.fit_transform(df['Names'])
    df['Title_id'] = pd.factorize(df['Title'])[0]+1

    scaler = preprocessing.StandardScaler()
    df['Title_id_scaled'] = scaler.fit_transform(df['Title_id'])


def processAge():    
    global df
    
    setMissingAges()    
    
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df['Age'])
    
    df['isChild'] = np.where(df.Age < 13, 1, 0)
    
    df['Age_bin'] = pd.qcut(df['Age'], 4)

    df = pd.concat([df, pd.get_dummies(df['Age_bin']).rename(columns=lambda x: 'Age_' + str(x))], axis=1)
    
    df['Age_bin_id'] = pd.factorize(df['Age_bin'])[0]+1

    scaler = preprocessing.StandardScaler()
    df['Age_bin_id_scaled'] = scaler.fit_transform(df['Age_bin_id'])

    df.drop('Age_bin', axis=1, inplace=True)


def setMissingAges():
    global df
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title_id','Pclass','Names','CabinLetter']]
    X = age_df.loc[ (df.Age.notnull()) ].values[:, 1::]
    y = age_df.loc[ (df.Age.notnull()) ].values[:, 0]

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)

    predictedAges = rtr.predict(age_df.loc[ (df.Age.isnull()) ].values[:, 1::])
    
    title_ages = df[['Age','Title_id']].groupby(['Title_id']).median()['Age']        
    df.loc[ (df.Age.isnull()), 'Age' ] = np.add(df[df.Age.isnull()].apply(lambda r: title_ages[r.Title_id] ,1).values  , predictedAges)/2.0
    

def genPolyFeatures():
    global df
    #numerics = df.loc[:, ['Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', 'SibSp_scaled', 
    #                      'Names_scaled', 'CabinNumber_scaled', 'Age_bin_id_scaled', 'Fare_bin_id_scaled', 'perFare_scaled', 'Family_scaled']]    
    numerics = df.loc[:, ['Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', 'SibSp_scaled', 
                          'perFare_scaled', 'Family_scaled']]    

    new_fields_count = 0
    for i in range(0, numerics.columns.size-1):
        for j in range(0, numerics.columns.size-1):
            if i <= j:
                name = str(numerics.columns.values[i]) + "*" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
                new_fields_count += 1
            if i < j:
                name = str(numerics.columns.values[i]) + "+" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)
                new_fields_count += 1
            if not i == j:
                name = str(numerics.columns.values[i]) + "/" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name=name)], axis=1)
                name = str(numerics.columns.values[i]) + "-" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)
                new_fields_count += 2

    print "\n", new_fields_count, "new poly features generated"
    
def dropCorrFeatures(corr_threshold):
    global df
    df_corr = df.drop(['Survived', 'PassengerId'],axis=1).corr(method='spearman')
    
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr

    drops = []
    for col in df_corr.columns.values:
        if np.in1d([col],drops):
            continue

        corr = df_corr[abs(df_corr[col]) > corr_threshold].index            
        drops = np.union1d(drops, corr)

    print "\nDropping", drops.shape[0], "highly correlated features.\n"
    print 'Dropped columns: ', drops
    df.drop(drops, axis=1, inplace=True)


def genDF(data, train_len, corr_threshold=0.98,  usePolyFeatures=False):    
    global df
    df = data
    processCabin()
    processTicket()
    processName()
    processFamily()
    processFare()    
    processEmbarked()        
    processSex()
    processPClass()
    processAge()
            
    rawDropList = ['Name', 'Names', 'Title', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked', \
                   'Cabin', 'CabinLetter', 'CabinNumber', 'Age', 'Fare', 'Ticket', 'perFare', 'Family', 'Title_id', 'TicketNumber'] 
    df.drop(rawDropList, axis=1, inplace=True)    
    
    # Move the survived column back to the first position
    columns_list = list(df.columns.values)
    columns_list.remove('Survived')
    new_col_list = list(['Survived'])
    new_col_list.extend(columns_list)
    df = df.reindex(columns=new_col_list)

    print "\nStarting with", df.columns.size, "manually generated features.\n"
    
    if usePolyFeatures:
        genPolyFeatures()
    
    dropCorrFeatures(corr_threshold)

    input_df = df[:train_len]
        
    submit_df  = df[train_len:]
    submit_df = submit_df.drop('Survived', 1)
    
    submit_ids = submit_df['PassengerId']    
    input_df = input_df.drop('PassengerId', 1) 
    submit_df = submit_df.drop('PassengerId', 1)
    
    X = input_df.iloc[:, 1::]
    y = input_df.iloc[:, 0]
    
    print "\n", X.columns.size, " features generated.\n"
    
    #survived_weight = .75
    #y_weights = np.array([survived_weight if s == 1 else 1 for s in y])
    y_weights = np.sum(y ==1)/float(np.sum(y == 0))    
        
    #X, submit_df = selectImportance(X, y, submit_df, submit_ids, fi_threshold, y_weights)
    
    return  X, y, submit_df, submit_ids, y_weights
    


def selectImportance(X, y, submit_df, y_weights, fi_threshold = 18):    
          
    # use RF to select import features
    forest = RandomForestClassifier(oob_score=True, n_estimators=10000, class_weight={0:1 , 1:y_weights }, criterion='entropy')
    forest.fit(X.values, y.values)   #, sample_weight=y_weights)
    feature_importance = forest.feature_importances_
    
    feature_importance = 100.0 * (feature_importance / feature_importance.max())    

    important_idx = np.where(feature_importance > fi_threshold)[0]    

    features_list = X.columns.values
    drop_idx = [  i for i in range(0, len(features_list)) if i not in important_idx ]
    print 'Drop less important features: ', features_list[drop_idx]
    
    important_features = features_list[important_idx]
    print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance).\n"
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]

    # Plot feature importance
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    plt.show()
    
    X = X.iloc[:, important_idx].iloc[:, sorted_idx]        
    submit_df = submit_df.iloc[:,important_idx].iloc[:,sorted_idx]        
        
    return X, submit_df

