import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List,Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVC
import seaborn as sns

SEED = 1
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 1000)

with open('data.json') as f:
    list_ = json.load(f)
len_ = len(list_)
test_list = list_[:int(len_ * 0.3)]
learn_list = list_[int(len_ * 0.3) + 1:]


def key_by_value(dict_: dict, value) -> Any:
    return [i for i in dict_ if dict_[i] == value]


def data_parser(list_: list) -> Dict:
    result = {}
    for dict_ in list_:
        if dict_['cuisine'] not in result:
            result.update({dict_['cuisine']: {}})
        for ingr in dict_['ingredients']:
            if ingr not in result[dict_['cuisine']]:
                result[dict_['cuisine']].update({ingr: 1})
            else:
                result[dict_['cuisine']][ingr] += 1
    return result


def encode_ingredients(list_: list) -> List[Dict]:
    result = {}
    result_fin = {}
    toptop = {}
    tmp = {}
    tmp1 = {}
    for dict_ in list_:
        for ingr in dict_['ingredients']:
            if ingr not in result:
                result.update({ingr:1})
            else:
                result[ingr] += 1
    list_ = sorted(list(result.values()), reverse=True)
    # print(list_)
    count = len(list_)
    for _ in list_:
        max_val = max(list_)
        d_new = {key: count for key in key_by_value(result, max_val)}
        d_new_1 = {key: max_val for key in key_by_value(result, max_val)}
        tmp.update(d_new)
        tmp1.update(d_new_1)
        count -= 1
        for __ in d_new:
            list_.pop(0)
        # print(list_)
    result_fin.update(tmp)
    toptop.update(tmp1)
    # print('Encoded:',result_fin)
    return [result_fin,toptop]
    # result = {}
    # count = 0.01
    # for dict_ in list_:
    #     for ingredient in dict_['ingredients']:
    #         if ingredient not in result:
    #             result.update({str(ingredient): count})
    #             count += 0.01
    # return result


def good_max_ingredient(dict_: Dict):
    result = {}
    for cuisine in dict_:
        tmp = {}
        d = dict_[cuisine]
        list_ = sorted(list(d.values()), reverse=True)
        for _ in list_:
            max_val = max(list_)
            d_new = {key: max_val for key in key_by_value(d, max_val)}
            tmp.update(d_new)
            for __ in d_new:
                list_.pop(0)
            # print(list_)
        result.update({cuisine: tmp})
    return result


def max_ingredient(dict_: Dict, a: int | None, flag=False):
    result = {}
    for cuisine in dict_:
        max_value = 0
        max_key = ''
        top = []

        for ing in dict_[cuisine]:
            top.append({ing: dict_[cuisine][ing]})
        N = len(top)
        for i in range(N):
            for j in range(N - i - 1):
                if list(top[j].values())[0] > list(top[j + 1].values())[0]:
                    top[j], top[j + 1] = top[j + 1], top[j]
        if flag:
            top3 = {}
            for i in range(N - 1, N - a - 1, -1):
                top3.update(top[i])
            result.update({cuisine: top3})
        else:
            result.update({cuisine: top})
    return result


def data_parser_frame(list_: list, top_dict: dict, enc_dict: dict,toptop: dict) -> Dict:
    result = {}
    # print(toptop['white almond bark'])
    for i, dict_ in enumerate(list_):
        top = 0.0
        tmp_ing = 0
        tmp_top_ing = 0
        tmp_toptop = 0
        kol_ingr_in_recipe = 0

        result.update({i: {'cuisine': dict_['cuisine'],'average': 0.0, 'quotient': 1.0}})
        N = len(dict_['ingredients'])
        top_ingredients = list(top_dict[dict_['cuisine']].keys())[0:N]
        coef = 0

        for j,ing in enumerate(dict_['ingredients']):
            # значение ingredients - сумма уникальных значений в рецептк (данных каждому ингредиенту)
            verh = top_dict[dict_['cuisine']][ing]
            niz = toptop[ing]
            coef += verh / niz
            tmp_ing += (enc_dict[ing])
            tmp_top_ing += (enc_dict[top_ingredients[j]])
            kol_ingr_in_recipe += 1
        result[i]['average'] = tmp_ing/kol_ingr_in_recipe
        result[i]['quotient'] = tmp_ing/tmp_top_ing * coef

    return result


# def f7(seq):
#     seen = set()
#     seen_add = seen.add
#     return [x for x in seq if not (x in seen or seen_add(x))]


def get_dataframe(dict_: Dict):
    dataframe = pd.DataFrame(dict_.values())
    dataframe = dataframe.fillna(0)
    dataframe['target'] = dataframe['cuisine'].map(
        {'greek': 1, 'southern_us':2, 'filipino': 3, 'indian': 4, 'jamaican': 5, 'spanish': 6, 'italian': 7,
         'mexican': 8, 'chinese': 9, 'british': 10, 'thai': 11, 'vietnamese': 12, 'cajun_creole': 13, 'brazilian': 14,
         'french': 15, 'japanese': 16, 'irish': 17, 'korean': 18, 'moroccan': 19, 'russian': 20}).values
    return dataframe


def make_dict_for_dataframe(list_: list,all_ingr: list) -> Dict:
    result = {}
    for i,dict_ in enumerate(list_):
        result.update({i:{"cuisine": dict_["cuisine"]}})
        for ing in all_ingr:
            if(ing in dict_["ingredients"]):
                result[i].update({ing: 1})
            else:
                result[i].update({ing: 0})
    return result


def get_all_ingredients(list_: list) -> List:
    result = []
    for dict_ in list_:
        for ing in dict_["ingredients"]:
            if(ing not in result):
                result.append(ing)
    return result




def plot_features_scores(model, data, target, column_names, model_type):
    '''Функция для визуализации важности признаков'''

    model.fit(data, target)

    if model_type == 'rf':
        (pd.DataFrame(data={'score': model['rf'].feature_importances_},
                      index=column_names).sort_values(by='score')
         .plot(kind='barh', grid=True,
               figsize=(6, 6), legend=False));
        plt.show()
    elif model_type == 'lr':
        (pd.DataFrame(data={'score': model['lr'].coef_[0]},
                      index=column_names).sort_values(by='score')
         .plot(kind='barh', grid=True,
               figsize=(6, 6), legend=False));
        plt.show()
    else:
        raise KeyError('Unknown model_type')


def sign_importance(X, y, model_type):
    if model_type == 'rf':
        model = Pipeline([('rf', RandomForestClassifier(n_jobs=-1,
                                                        class_weight='balanced',
                                                        random_state=SEED))])
    elif model_type == 'lr':
        model = Pipeline([('p_trans', PowerTransformer(method='yeo-johnson', standardize=True)),
                          ('lr', LogisticRegression(solver='liblinear',
                                                    penalty='l1',
                                                    max_iter=500,
                                                    class_weight='balanced',
                                                    random_state=SEED)
                           )])

    # # параметры кросс-валидации (стратифицированная 5-фолдовая с перемешиванием)
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    #
    # scores = cross_validate(estimator=rf, X=X, y= y,
    #                          cv=skf, scoring='roc_auc', n_jobs=-1)
    # print('scores = {} \nmean score = {:.5f} +/- {:.5f}'.format(scores, scores.mean(), scores.std()))

    # важность признаков
    plot_features_scores(model=model, data=X, target=y, column_names=X.columns, model_type=model_type)


def diagram_ingridients(dict_: Dict):
    width = 4
    multiplier = 0
    fig, ax = plt.subplots(10, 2, sharey='row')
    num = np.arange(len(dict_))
    j = 0
    i = 0
    for d in dict_:
        tmp = np.arange(0, 3 * width, width)
        rect = ax[i, j].bar(tmp, list(dict_[d].values()), width, label=list(dict_[d].keys()))
        ax[i, j].bar_label(rect, padding=3)
        ax[i, j].set_ylabel('N')
        ax[i, j].set_title(d)
        ax[i, j].set_xticks(tmp, list(dict_[d].keys()))

        if ((j + 1) // 2 != 0 and (j + 1) % 2 == 0):
            i = i + 1
            j = 0
        else:
            j = j + 1
        # ax.bar_label(rect3, padding=3

    # tmp = np.arange(0,10000,500)
    # print(len(tmp))

    # ax.set_ylim(0, 1100)
    fig.tight_layout()
    plt.subplots_adjust(hspace=1, top=0.976, bottom=0.03)
    plt.show()


all_ingr = get_all_ingredients(list_)
# print(make_dict_for_dataframe(learn_list,all_ingr))
# print(learn_list)
# print(max_ingredient(data_parser(test_list)))
# diagram_ingridients(max_ingredient(data_parser(learn_list)),3)
# print(data_parser_frame(learn_list))
# print(encode_ingredients(learn_list))
# [enc,toptop] = encode_ingredients(learn_list)


dataframe_learn = get_dataframe(make_dict_for_dataframe(learn_list, all_ingr))

# dataframe = pd.read_json('learn_list.json')
# ВЫБРАННЫЕ ПРИЗНАКИ: СУММА КОДОВ ИНГРЕДИЕНТОВ, СУММА ПОВТОРЕНИЙ ВО ВСЕХ РЕЦЕПТАХ КУХНИ 5 САМЫХ ПОПУЛЯРНЫХ ИНГРЕДИЕНТОВ, СРЕДНЕЕ АРИФМЕТИЧЕСКОЕ РЕЦЕПТА
# #оставим только численные признаки
# [enc,toptop] = encode_ingredients(test_list)
dataframe_test = get_dataframe(make_dict_for_dataframe(test_list, all_ingr))

X_test = dataframe_test.select_dtypes(exclude=['object']).copy().drop(columns=['target'])
# X_test = X_test[['ingredients','sum_index_top']]
X_learn = dataframe_learn.select_dtypes(exclude=['object']).copy().drop(columns=['target'])
# X = X[['ingredients','sum_index_top']]
print(X_learn)
y_learn = dataframe_learn['target']
y_test = dataframe_test['target']



# rf = RandomForestClassifier()
# grid_space={'max_depth':[3,5,10,None],
#               'n_estimators':[10,100,200],
#               'max_features':[1,3,5,7],
#               'min_samples_leaf':[1,2,3],
#               'min_samples_split':[1,2,3]
#            }
#
# grid= GridSearchCV(rf, param_grid=grid_space,cv = 3,scoring='accuracy')
# model_grid = grid.fit(X,y)
#
# print('Best hyperparameters are: '+str(model_grid.best_params_))
# print('Best score is: '+str(model_grid.best_score_))


# print(dataframe.index,y)
# sign_importance(X, y, 'rf')
# # sign_importance(X, y, 'lr')
#
# correlation_matrix = dataframe_learn.drop(columns=['cuisine']).corr()
# # # Выводим признаки на тепловую карту
# plt.figure(figsize=(10, 6))
# sns.heatmap(correlation_matrix, annot=True)
# plt.show()



reg = RandomForestClassifier(max_depth= 500).fit(X_learn, y_learn.tolist())
print(reg.score(X_learn,y_learn), reg.score(X_test,y_test))
a = reg.predict(X_test)
print(a)
print(y_test)
b = y_test.tolist()
c = [a[i] == b[i] for i in range(len(b))]
print(c)



# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = np.dot(X, np.array([1, 2])) + 3
# print(y)
# reg = LinearRegression().fit(X, y)
# print(reg.score(X, y),reg.coef_,reg.intercept_)
