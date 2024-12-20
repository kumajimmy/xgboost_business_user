"""
Error Distribution:
Rating diff.     Count
>=0 and <1      n = 102411
>=1 and <2      n = 32697
>=2 and <3      n = 6111
>=3 and <4      n = 823
>=4             n = 2

RMSE: 
RMSE: 0.9776803904377737

Execution Time:
260s

"""


from pyspark import SparkConf, SparkContext
from xgboost import XGBRegressor
import csv
import json
import sys
import time
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans


def calculate_global_stats(train_data):
    ratings = train_data.map(lambda x: float(x[2]))
    global_mean = ratings.mean()
    return global_mean

def prepare_data(data, split="train"):
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(",")).map(lambda row: (row[0], row[1], float(row[2])))
    return data

def create_dictionaries(train_data, global_mean):
    business_user_dic = (train_data.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set).collectAsMap())
    user_business_dic = (train_data.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set).collectAsMap())
    
    business_stats = (train_data.map(lambda row: (row[1], float(row[2]))).groupByKey().mapValues(lambda x: (sum(x), len(x))).collectAsMap())
    
    min_reviews = 5  
    business_avg_dic = {}
    for business, (rating_sum, count) in business_stats.items():
        weighted_rating = (rating_sum + global_mean * min_reviews)/(count + min_reviews)
        business_avg_dic[business] = weighted_rating
    
    user_stats = (train_data.map(lambda row: (row[0], float(row[2]))).groupByKey().mapValues(lambda x: (sum(x), len(x))).collectAsMap())
    
    user_avg_dic = {}
    for user, (rating_sum, count) in user_stats.items():
        weighted_rating = (rating_sum + global_mean * min_reviews)/(count + min_reviews)
        user_avg_dic[user] = weighted_rating
    
    business_user_r_dic = {}
    business_user_ratings = (train_data.map(lambda row: (row[1], (row[0], float(row[2])))).groupByKey().mapValues(list).collect())
    
    for business, user_r_list in business_user_ratings:
        temp = {}
        for user_r in user_r_list:
            temp[user_r[0]] = user_r[1]
        business_user_r_dic[business] = temp

    return (business_user_dic, user_business_dic, business_avg_dic, 
            user_avg_dic, business_user_r_dic)
            
def process_photos(photo_rdd):
    photo_features = (photo_rdd
        .map(lambda row: (
            row["business_id"],
            (
                1,
                1 if row.get("label", "") == "food" else 0,
                1 if row.get("label", "") == "interior" else 0,
                1 if row.get("label", "") == "exterior" else 0,
                1 if row.get("label", "") == "drink" else 0,
                1 if row.get("label", "") == "menu" else 0
            )
        ))
        .reduceByKey(lambda a, b: (
            a[0] + b[0],  
            a[1] + b[1],  
            a[2] + b[2],  
            a[3] + b[3],  
            a[4] + b[4],  
            a[5] + b[5]   
        ))
        .mapValues(lambda x: (
            x[0],  
            x[1]/x[0] if x[0] > 0 else 0,  
            x[2]/x[0] if x[0] > 0 else 0,  
            x[3]/x[0] if x[0] > 0 else 0,  
            x[4]/x[0] if x[0] > 0 else 0,  
            x[5]/x[0] if x[0] > 0 else 0   
        )))
    
    return photo_features.collectAsMap()

def process_reviews(review_rdd):
    review_rdd = (review_rdd.map(lambda row: (row["business_id"],
                (float(row.get("useful", 0)), float(row.get("funny", 0)),
                 float(row.get("cool", 0)), 1)))
                .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1],
                                           a[2]+b[2], a[3]+b[3]))
                .mapValues(lambda x: (x[0]/x[3] if x[3] != 0 else 0,
                                      x[1]/x[3] if x[3] != 0 else 0,
                                      x[2]/x[3] if x[3] != 0 else 0)).cache())
    return review_rdd.collectAsMap()

def process_user(usr_rdd):
    def transform_user(row):
        yelping_str = row.get("yelping_since", "")
        membership_years = 0
        if yelping_str:
            try:
                yelp_date = datetime.strptime(yelping_str, "%Y-%m-%d")
                now = datetime.now()
                membership_years = (now-yelp_date).days/365.25
            except:
                membership_years = 0

        compliment_sum = sum([int(row.get(k, 0)) for k in ["compliment_hot", "compliment_more", "compliment_profile",
                                              "compliment_cute", "compliment_list", "compliment_note",
                                              "compliment_plain", "compliment_cool", "compliment_funny",
                                              "compliment_writer", "compliment_photos"]])
        num_elite = len(row.get("elite", "").split(',')) if row.get("elite", "") else 0
        num_friends = len(row.get("friends", "").split(',')) if row.get("friends", "") else 0

        return (
            float(row.get("average_stars", 0)),
            float(row.get("review_count", 0)),
            float(row.get("fans", 0)),
            float(row.get("useful", 0)),
            float(row.get("funny", 0)),
            float(row.get("cool", 0)),
            num_elite,
            num_friends,
            compliment_sum,
            membership_years
        )

    return usr_rdd.map(lambda row: (row["user_id"], transform_user(row))).cache().collectAsMap()

def process_bus(bus_rdd):
    bus_rdd = bus_rdd.map(lambda row: (
        row["business_id"],
            (float(row.get("stars", 0)),
            float(row.get("review_count", 0)),
            int(row.get("is_open", 0)),
            len(row.get("categories", "").split(', ')) if row.get("categories", "") else 0))).cache()
    return bus_rdd.collectAsMap()

def process_checkins(checkin_rdd):
    checkin_counts = checkin_rdd.map(lambda row: (row['business_id'], sum(row['time'].values()))).collectAsMap()
    return checkin_counts

def process_tips_business(tip_rdd):
    business_tips = tip_rdd.map(lambda row: (row["business_id"],(row.get("likes", 0), len(row.get("text", "")), 1)))
    business_tip_stats = (business_tips.reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])).mapValues(lambda x: (x[0]/x[2] if x[2] != 0 else 0,x[1]/x[2] if x[2] != 0 else 0, x[2])))
    return business_tip_stats.collectAsMap()

def process_tips_user(tip_rdd):
    user_tips = tip_rdd.map(lambda row: (row["user_id"], (len(row.get("text", "")), 1)))
    user_tip_stats = user_tips.reduceByKey(lambda a, b: (a[0]+b[0], a[1] + b[1])).mapValues(lambda x: (x[0]/x[1] if x[1] != 0 else 0, x[1]))
    return user_tip_stats.collectAsMap()

def process_categories(business_rdd):
    def extract_category_features(categories_str):
        if not categories_str:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        
        categories = set(cat.strip() for cat in categories_str.split(','))
        return [
            len(categories),  #category count
            1 if 'Restaurants' in categories else 0,
            1 if 'Bars' in categories else 0,
            1 if 'Fast Food' in categories else 0,
            1 if 'Coffee & Tea' in categories else 0,
            1 if 'Food' in categories else 0,
            1 if 'Nightlife' in categories else 0,
            1 if 'Shopping' in categories else 0
        ]
    
    return business_rdd.map(lambda x: (x['business_id'], extract_category_features(x.get('categories', '')))).collectAsMap()

def process_geographic_features(business_rdd):
    def extract_geo_features(row):
        try:
            lat = row.get('latitude')
            lng = row.get('longitude')
            
            if lat is not None and lng is not None:
                lat = float(lat)
                lng = float(lng)
            else:
                return (row['business_id'], None)
            
            if lat == 0 and lng == 0:
                return (row['business_id'], None)
            
            state = row.get('state', '')
            postal_code = row.get('postal_code', '')
            
            return (row['business_id'], (lat, lng, state, postal_code))
        except (ValueError, TypeError):
            return (row['business_id'], None)
    
    geo_data = business_rdd.map(extract_geo_features).filter(lambda x: x[1] is not None)
    coords_with_ids = geo_data.map(lambda x: (x[0], x[1][:2])).collect()
    
    if not coords_with_ids:
        return {
            'clusters': {},
            'state_postal': {},
            'postal_density': {},
            'state_ratings': {}
        }
    
    business_ids = [x[0] for x in coords_with_ids]
    coordinates = np.array([x[1] for x in coords_with_ids])
    
    coord_scaler = StandardScaler()
    scaled_coordinates = coord_scaler.fit_transform(coordinates)
    
    n_clusters = min(20, len(coordinates))  #ensure not having more clusters than points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_coordinates)
    cluster_dict = dict(zip(business_ids, clusters))
    state_postal_dict = geo_data.map(lambda x: (x[0], (x[1][2], x[1][3]))).collectAsMap()#state/postal code dictionaries
    
    postal_density = (geo_data
        .map(lambda x: (x[1][3], 1))
        .reduceByKey(lambda x, y: x + y)
        .collectAsMap())
        
    def safe_float(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return 0.0
            
    state_ratings = (business_rdd
        .map(lambda x: (x.get('state', ''), (safe_float(x.get('stars', 0)), 1)))
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        .mapValues(lambda x: x[0]/x[1] if x[1] > 0 else 0)
        .collectAsMap())
    
    return {
        'clusters': cluster_dict,
        'state_postal': state_postal_dict,
        'postal_density': postal_density,
        'state_ratings': state_ratings
    }

def process_train_data(row, review_dic, usr_dic, bus_dic, checkin_dic, tip_bus_dic, 
                      tip_usr_dic, photo_dic, category_dic, geo_features):
    usr, bus = row[:2]
    
    useful, funny, cool = review_dic.get(bus, (0, 0, 0))
    usr_features = usr_dic.get(usr, (0,) * 10)
    bus_features = bus_dic.get(bus, (0,) * 4)
    total_checkins = checkin_dic.get(bus, 0)
    bus_tip_features = tip_bus_dic.get(bus, (0, 0, 0))
    usr_tip_features = tip_usr_dic.get(usr, (0, 0))
    photo_features = photo_dic.get(bus, (0, 0, 0, 0, 0, 0))
    category_features = category_dic.get(bus, [0] * 8)
    cluster = geo_features['clusters'].get(bus, -1)
    state_postal = geo_features['state_postal'].get(bus, ('', ''))
    postal_density = geo_features['postal_density'].get(state_postal[1], 0)
    state_avg_rating = geo_features['state_ratings'].get(state_postal[0], 0)
    
    features = [
        useful, funny, cool,
        *usr_features,
        *bus_features,
        total_checkins,
        *bus_tip_features,
        *usr_tip_features,
        *photo_features,
        *category_features,
        cluster,
        postal_density,
        state_avg_rating
    ]

    user_avg_stars = usr_features[0]
    business_stars = bus_features[0]
    user_review_count = usr_features[1]
    business_review_count = bus_features[1]

    interaction_feature = user_avg_stars * (business_stars / max(1, business_review_count))
    geo_interaction = business_stars * state_avg_rating / max(1, abs(business_stars - state_avg_rating))
    
    features.extend([interaction_feature, geo_interaction])
    
    features = [0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in features]
    return features

def save_data(data, output_file_name):
    header = ["user_id", " business_id", " prediction"]
    with open(output_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def read_json_spark(path, sc):
    return sc.textFile(path).map(lambda row: json.loads(row))

def calculate_rating_diffs(predictions_and_actuals_local):
    diffs = [abs(pred - actual) for pred, actual in predictions_and_actuals_local]
    counts = {
        '0-1': sum(1 for d in diffs if 0 <= d < 1),
        '1-2': sum(1 for d in diffs if 1 <= d < 2),
        '2-3': sum(1 for d in diffs if 2 <= d < 3),
        '3-4': sum(1 for d in diffs if 3 <= d < 4),
        '4+': sum(1 for d in diffs if d >= 4)
    }
    print("\nRating Difference Distribution:")
    print("Rating diff.     Count")
    print(f">=0 and <1      n = {counts['0-1']}")
    print(f">=1 and <2      n = {counts['1-2']}")
    print(f">=2 and <3      n = {counts['2-3']}")
    print(f">=3 and <4      n = {counts['3-4']}")
    print(f">=4             n = {counts['4+']}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit competition.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    conf = SparkConf().setAppName("Competition")
    spark = SparkContext(conf=conf)
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        train_data = prepare_data(spark.textFile(folder_path + '/yelp_train.csv'), "train").cache()
        global_mean = calculate_global_stats(train_data)
        dictionaries = create_dictionaries(train_data, global_mean)

        review_rdd = read_json_spark(folder_path + "/review_train.json", spark)
        review_dict = process_reviews(review_rdd)

        usr_rdd = read_json_spark(folder_path + "/user.json", spark)
        user_dict = process_user(usr_rdd)

        bus_rdd = read_json_spark(folder_path + "/business.json", spark)
        bus_dict = process_bus(bus_rdd)
        category_dict = process_categories(bus_rdd)
        geo_features = process_geographic_features(bus_rdd)

        checkin_rdd = read_json_spark(folder_path + "/checkin.json", spark)
        checkin_dict = process_checkins(checkin_rdd)

        tip_rdd = read_json_spark(folder_path + "/tip.json", spark)
        tip_bus_dict = process_tips_business(tip_rdd)
        tip_usr_dict = process_tips_user(tip_rdd)
        
        photo_rdd = read_json_spark(folder_path + "/photo.json", spark)
        photo_dict = process_photos(photo_rdd)

        review_dict_broadcast = spark.broadcast(review_dict)
        user_dict_broadcast = spark.broadcast(user_dict)
        bus_dict_broadcast = spark.broadcast(bus_dict)
        checkin_dict_broadcast = spark.broadcast(checkin_dict)
        tip_bus_dict_broadcast = spark.broadcast(tip_bus_dict)
        tip_usr_dict_broadcast = spark.broadcast(tip_usr_dict)
        photo_dict_broadcast = spark.broadcast(photo_dict)
        category_dict_broadcast = spark.broadcast(category_dict)
        geo_features_broadcast = spark.broadcast(geo_features)

        test_data = prepare_data(spark.textFile(test_file), "test").cache()

        train_features = train_data.map(lambda x: (
            (x[0], x[1]),
            process_train_data(
                x,
                review_dict_broadcast.value,
                user_dict_broadcast.value,
                bus_dict_broadcast.value,
                checkin_dict_broadcast.value,
                tip_bus_dict_broadcast.value,
                tip_usr_dict_broadcast.value,
                photo_dict_broadcast.value,
                category_dict_broadcast.value,
                geo_features_broadcast.value
            ),x[2]))
        
        X_train_full = np.array(train_features.map(lambda x: x[1]).collect(), dtype='float32')
        y_train = np.array(train_features.map(lambda x: x[2]).collect(), dtype='float32')

        test_features = test_data.map(lambda x: (
            (x[0], x[1]),
            process_train_data(
                x,
                review_dict_broadcast.value,
                user_dict_broadcast.value,
                bus_dict_broadcast.value,
                checkin_dict_broadcast.value,
                tip_bus_dict_broadcast.value,
                tip_usr_dict_broadcast.value,
                photo_dict_broadcast.value,
                category_dict_broadcast.value,
                geo_features_broadcast.value
            ))).collect()
        
        X_test = np.array([x[1] for x in test_features], dtype='float32')
        test_keys = [x[0] for x in test_features]

        scaler = MinMaxScaler()
        X_train_full = scaler.fit_transform(X_train_full)
        X_test = scaler.transform(X_test)
        
#        param = {
#            'lambda': 4,
#            'alpha': 0.05,
#            'colsample_bytree': 0.8,
#            'subsample': 0.9,
#            'learning_rate': 0.015,
#            'max_depth': 8,
#            'random_state': 16,
#            'min_child_weight': 10,
#            'n_estimators': 1500,
#            'gamma': 1,
#            'tree_method':'hist',
#            'eval_metric': 'rmse'
#        } #RMSE: 0.9782931137645079
#
        param = {
            'lambda': 6,
            'alpha': 0.08,
            'colsample_bytree': 0.8,
            'subsample': 0.85,
            'learning_rate': 0.012,
            'max_depth': 8,
            'random_state': 16,
            'min_child_weight': 12,
            'n_estimators': 1800,
            'gamma': 1.4,
            'tree_method':'hist',
            'eval_metric': 'rmse'
        }#best RMSE: RMSE: 0.9776803904377737

#        param = {
#            'lambda': 5.712561131806751,
#            'alpha': 0.05213197353376989,
#            'colsample_bytree': 0.8267190165714701,
#            'subsample': 0.8699015865811938,
#            'learning_rate': 0.01036110689687315,
#            'max_depth': 8,
#            'random_state': 16,
#            'min_child_weight': 11,
#            'n_estimators': 1774,
#            'gamma': 1.3952243831078934,
#            'tree_method':'hist',
#            'eval_metric': 'rmse'
#        }#RMSE: 0.978213631023836

#        # Parameter optimization
#        from sklearn.model_selection import RandomizedSearchCV
#        from scipy.stats import uniform, randint
#
#        print("\nStarting parameter optimization...")
#        
#        focused_param_distributions = {
#            'reg_lambda': uniform(4.0, 2.5),        
#            'reg_alpha': uniform(0.05, 0.04),       
#            'colsample_bytree': uniform(0.75, 0.1), 
#            'subsample': uniform(0.83, 0.09),       
#            'learning_rate': uniform(0.01, 0.005),  
#            'max_depth': [8],                       
#            'min_child_weight': randint(10, 14),    
#            'n_estimators': randint(1500, 2000),    
#            'gamma': uniform(1.0, 0.5)              
#        }
#
#        base_model = XGBRegressor(
#            tree_method='hist',
#            eval_metric='rmse',
#            random_state=16,
#            verbosity=0
#        )
#
#        random_search = RandomizedSearchCV(
#            estimator=base_model,
#            param_distributions=focused_param_distributions,
#            n_iter=50,
#            cv=3,
#            random_state=16,
#            n_jobs=-1,
#            verbose=1,
#            scoring='neg_root_mean_squared_error'
#        )
#
#        print("\nFitting RandomizedSearchCV...")
#        random_search.fit(X_train_full, y_train)
#
#        print("\nBest parameters found:")
#        for param, value in random_search.best_params_.items():
#            print(f"{param}: {value}")
#        print(f"\nBest CV RMSE: {-random_search.best_score_}")
#
#        best_params = random_search.best_params_
#        best_params.update({
#            'tree_method': 'hist',
#            'eval_metric': 'rmse',
#            'random_state': 16,
#            'verbosity': 0
#        })
#
#        # Train final model with best parameters
#        print("\nTraining final model with best parameters...")
#        final_model = XGBRegressor(**best_params)
#        final_model.fit(
#            X_train_full, 
#            y_train,
#            eval_set=[(X_train_full, y_train)],
#            early_stopping_rounds=50,
#            verbose=False
#        )
#
#        y_pred_model = final_model.predict(X_test)

        xgb = XGBRegressor(**param)
        xgb.fit(X_train_full, y_train, verbose=False)

        y_pred_model = xgb.predict(X_test)
        final_predictions = spark.parallelize(zip(test_keys, y_pred_model))
        pred_data = final_predictions.map(lambda x: [x[0][0], x[0][1], x[1]]).collect()
        save_data(pred_data, output_file)

        end_time = time.time()
        print(f'Duration: {end_time - start_time:.2f}s')

        actual_rdd = test_data.map(lambda x: ((x[0], x[1]), x[2]))
        predictions_rdd = final_predictions.map(lambda x: (x[0], x[1]))
        predictions_and_actuals = predictions_rdd.join(actual_rdd)
        predictions_and_actuals_local = predictions_and_actuals.map(lambda x: x[1]).collect()
        rmse = math.sqrt(np.mean([(pred - actual)**2 for pred, actual in predictions_and_actuals_local]))
        print(f"RMSE: {rmse}")
#        calculate_rating_diffs(predictions_and_actuals_local)

    finally:
        spark.stop()