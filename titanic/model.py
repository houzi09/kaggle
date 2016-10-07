import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC

from itertools import combinations

import xgboost as xgb

class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        y_prob = self.stacker.predict_proba(S_test)[:,1]
        return y_pred,y_prob
    

def find_RF_clf(x_data, y_data,y_weights):
    sqrtfeat = int(np.sqrt(x_data.shape[1]))
    best_score = 0
    
    grids = {
        'max_features': np.rint(np.linspace(sqrtfeat, sqrtfeat+2, 3)).astype(int),
        'min_samples_split': np.rint(np.linspace(X.shape[0]*.01, X.shape[0]*.05, 3)).astype(int)
        }.items()

    test_grids =list(combinations(grids, 2))
    print 'need iter times: ', len(test_grids)

    parameters = {'n_estimators': 100,
            'oob_score': True,
            'random_state': 7,
            'class_weight': {0:1 , 1:  y_weights},
            'n_jobs': -1}
    
    count = 0
    best_score = 0
    for comb_grids in test_grids:    
        comb_dict = dict(comb_grids)
        rf_clf = RandomForestClassifier(**parameters)
        grid_search = GridSearchCV(estimator=rf_clf, param_grid=comb_dict)
        grid_search.fit(x_data, y_data)            
        if best_score< grid_search.best_score_:    
            best_parameters=grid_search.best_estimator_.get_params()
            for name in comb_dict.keys():
                parameters[name]=best_parameters[name]
            best_score = grid_search.best_score_
            count += 1
        print 'best %r , current %r ' %(best_score, grid_search.best_score_)   
    print 'Set times: ',count
    print 'Best parameters:', parameters
    rf_clf = RandomForestClassifier(**parameters)
    return rf_clf


def find_XGB_clf(x_data, y_data, y_weights, n_iter=100):
    sqrtfeat = int(np.sqrt(x_data.shape[1]))
    best_score = 0
    param_dist = {            
        'learning_rate': np.arange(0.001, 0.2, 0.001),
        'max_depth': np.rint(np.linspace(3, np.log(x_data.shape[0]), 3)).astype(int),
        'min_child_weight': np.rint(np.linspace(x_data.shape[0]*.01, x_data.shape[0]*.05, 3)).astype(int),
        #'gamma':  np.arange(0.0, 0.5, 0.1),
        #'subsample': np.arange(0.6, 1, 0.1),
        'colsample_bytree': np.linspace(sqrtfeat, x_data.shape[1], 5, endpoint=False)/(x_data.shape[1]),  
        'reg_lambda': np.arange(0.0, 2, 0.2)
    }
    
    parameters = {
        'subsample': 0.7,
        'gamma': 0.3,
        'n_estimators': 500,
        'objective' : 'binary:logistic',
        'nthread' : -1,
        'scale_pos_weight' : y_weights,        
        'seed' : 7
    }
    
    
    clf = xgb.XGBClassifier(**parameters)
    rand_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter)
    
    rand_search.fit(x_data, y_data)
    best_parameters=rand_search.best_estimator_.get_params()
    
    for name in param_dist:
        parameters[name]=best_parameters[name]
    
    print 'best score:',rand_search.best_score_    
    '''    
    test_grids =list(combinations(param_dist.items(), 2))
    print 'need iter times: ', len(test_grids)
    count = 0
    best_score = 0
    for comb_grids in test_grids:    
        comb_dict = dict(comb_grids)
        clf = xgb.XGBClassifier(**parameters)        
        grid_search = GridSearchCV(estimator=clf,param_grid=comb_dict)
        grid_search.fit(x_data, y_data)
        if best_score< grid_search.best_score_:
            best_parameters=grid_search.best_estimator_.get_params()
            for name in comb_dict.keys():
                parameters[name]=best_parameters[name]
            best_score = grid_search.best_score_
            count += 1
        print 'best %r , current %r' %(best_score, grid_search.best_score_)

    print 'hit: ',count
    '''
    print "Generating RandomForestClassifier model with parameters: ", parameters
    return xgb.XGBClassifier(**parameters)




def find_SVC(x_data, y_data):
    best_score = 0
    grids = {
            'C': np.arange(0.6,3,0.1),
            'degree': np.arange(2,10,1),
            'kernel': ['linear', 'poly', 'rbf','sigmoid']
        }.items()

    test_grids =list(combinations(grids, 2))
    print 'need iter times: ', len(test_grids)
    parameters = {
        'class_weight':{0:1 , 1:  y_weights},
        'random_state' : 7
    }

    count = 0
    best_score = 0
    for comb_grids in test_grids:    
        comb_dict = dict(comb_grids)          
        svc_clf = SVC(**parameters)
        grid_search = GridSearchCV(estimator=svc_clf,param_grid=comb_dict)
        grid_search.fit(x_data, y_data)            
        if best_score< grid_search.best_score_:
            best_parameters=grid_search.best_estimator_.get_params()
            for name in comb_dict.keys():
                parameters[name]=best_parameters[name]
            best_score = grid_search.best_score_
            count += 1
        print 'best %r , current %r' %(best_score, grid_search.best_score_)
        
    print 'hit: ',count
    
    return SVC(**parameters)    

def find_dt_cols(parameters, X, y, col_start, col_end):
    dt_clf = tree.DecisionTreeClassifier(**parameters)
    best_i = 0
    best_cv = 0
    for i in range(col_start, col_end):
        cv = cross_val_score(dt_clf, X[:,:i], y, cv=10)
        if cv.mean() > best_cv:
            best_cv = cv.mean()        
            best_i = i
            print best_i, cv, best_cv
    return best_i

def valid_y(y_test, y_pred):
    return 1-sum([ 1 if y_test[i]!=y_pred[i] else 0 for i in range(len(y_test))])*1.0/len(y_test)

def diff(a, b):
    count =  0
    for i, j in zip(a, b):
        if i!=j:
            count += 1
    return count

def cross_val_test(m, X, y, n_folds=10):
    skf = StratifiedKFold(y, n_folds)
    trains = []
    tests = []
    values = []
    for train_index, test_index in skf:
        trains.append(train_index)
        tests.append(test_index)      
        x_train = [ X[i] for i in train_index ]        
        y_train = [ y[i] for i in train_index ]
        x_test = [ X[i] for i in test_index ]        
        y_test = [ y[i] for i in test_index ]  
        m.fit(np.array(x_train), np.array(y_train))
        y_pred = m.predict(x_test)
        v =  1-sum([ 1 if y_test[i]!=y_pred[i] else 0 for i in range(len(y_test))])*1.0/len(y_test)
        values.append(v)
    print 'cv: ', values, sum(values)*1.0/len(values)
    
    return trains,tests



def try_CNN(nn_data):
    
    # small less than RT, XGB here.
    
    import tensorflow as tf

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


    ## conv1 layer ##
    W_conv1 = weight_variable([3,3, 1, 8])
    b_conv1 = bias_variable([8])

    xs = tf.placeholder(tf.float32, [None, 7*7])
    x_image = tf.reshape(xs, [-1, 7, 7, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)


    ## conv2 layer ##
    W_conv2 = weight_variable([3,3, 8, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)


    ## func1 layer ##  full connection
    W_fc1 = weight_variable([7*7*16, 32])
    b_fc1 = bias_variable([32])

    # h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*8])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 7*7*16])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    ## func2 layer ## output layer: softmax
    W_fc2 = weight_variable([32, 2])
    b_fc2 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # the error between prediction and real data
    ys = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = -tf.reduce_sum(ys * tf.log(prediction)) # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #sess = tf.Session()
    sess = tf.InteractiveSession()
    # important step
    sess.run(tf.initialize_all_variables())


    for i in range(5000):
        batch_indices = np.random.choice(nn_data.index, 100)
        batch_xy = nn_data.loc[batch_indices].values
        batch_xs = batch_xy[:,2:]    
        batch_ys = batch_xy[:,:2]    
        if i % 100 == 0:        
            train_accuacy = accuracy.eval(feed_dict={xs: batch_xs, 
                                    ys: batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuacy))

        train_step.run(feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    
    print("train all accuracy %g"%(accuracy.eval(feed_dict={xs: nn_data.values[:,2:],
                                ys: nn_data.values[:,:2], keep_prob: 1.0})))
    
    #p_nn_y = prediction.eval(feed_dict={xs: submit_df, keep_prob: 1.0})
    #p_nn_survival = map(lambda x: 1 if x[0]>x[1] else 0,  p_nn_y)    