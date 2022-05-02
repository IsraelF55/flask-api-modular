import pymongo
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response
from bson.json_util import dumps
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

# variables de entorno
app = Flask(__name__)
client = pymongo.MongoClient(
    "mongodb+srv://modular:modular@cluster0.970tq.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.myFirstDatabase

# Funcion ejecutada en el proceso de clasificación


def find_n_neighbours(df, n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                      .iloc[:n].index,
                                      index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df

# rutas

# Devovler 6 juegos como maximo
@app.route('/recomendations/<string:user>/<int:games>', methods=['GET'])
def recomendations(user, games):
    print("\n\n estoy en recomendations 25", user, games, "\n\n")
    # Inicializacion y promediación
    data = db.ratings.find({})
    Ratings = pd.DataFrame(list(data))
    # print('\nRatings \n', Ratings)
    Mean = Ratings.groupby(by="user", as_index=False)['rate'].mean()
    # print('\nMean\n', Mean)
    Rating_avg = pd.merge(Ratings, Mean, on='user')
    # print('\nRating_avg\n', Rating_avg)
    Rating_avg['adg_rating'] = Rating_avg['rate_x'] - Rating_avg['rate_y']
    # print('\nRating_avg con adg\n', Rating_avg)

    # Me parece que solo hace falta que el Game_seen_by_user sea un parametro
    # Tabla final con la que se trabaja
    check = pd.pivot_table(Rating_avg, values='rate_x', index='user', columns='game')
    print('\n\nCheck ---\n', check)
    Game_seen_by_user = check.columns[check[check.index == user].notna().any()].tolist()
    print('\n\nGame_seen_by_user  ---\n', Game_seen_by_user)

    # Limpieza y normalizacion de valores nulos
    final = pd.pivot_table(Rating_avg, values='adg_rating', index='user', columns='game')
    # print('\nFinal\n', final)
    # Remplazamos los NaN por el promedio total de la columna de juego
    final_game = final.fillna(final.mean(axis=0))
    # Remplazamos los NaN por el promedio total de la fila del usuario
    final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)
    # print('\nFinal con limpieza\n', final_user)

    # similitud del usuario actual despues de la sustitución de NaN por la media del usuario
    b = cosine_similarity(final_user)
    np.fill_diagonal(b, 0)
    similarity_with_user = pd.DataFrame(b, index=final_user.index)
    similarity_with_user.columns = final_user.index

    # similitud del usuario actual despues de la sustitución de NaN por la media de cada juego
    cosine = cosine_similarity(final_game)
    np.fill_diagonal(cosine, 0)
    similarity_with_game = pd.DataFrame(cosine, index=final_game.index)
    similarity_with_game.columns = final_user.index
    # print('\nsimilarity_with_user\n', similarity_with_user)

    # Top 3 vecinos para cada usuario dado por el argumento
    sim_user_3_u = find_n_neighbours(similarity_with_game, 2)
    # print('\nsim_user_3_u\n', sim_user_3_u)
    # Top 3 vecinos para cada juego dado por el argumento
    # sim_user_3_g = find_n_neighbours(similarity_with_game, 3)
    # print('\nsim_user_3_g\n', sim_user_3_g)

    # Checar que hace - creo que se puede borrar, no
    Rating_avg = Rating_avg.astype({"game": str})
    Game_user = Rating_avg.groupby(by='user')['game'].apply(lambda x: ','.join(x))
    # print('\n\n\n Game_user* \n', Game_user)

    # Funcion controladora
    def recomendation_kubnn(user):
        print('\n\nUser en recomendation_kubnn:', user, '\n\n')
        Game_seen_by_user = check.columns[check[check.index == user].notna(
        ).any()].tolist()
        print('\n\Game_seen_by_user:', Game_seen_by_user, '\n\n')
        a = sim_user_3_u[sim_user_3_u.index == user].values
        b = a.squeeze().tolist()
        d = Game_user[Game_user.index.isin(b)]
        l = ','.join(d.values)
        Game_seen_by_similar_users = l.split(',')
        Games_under_consideration = list(set(Game_seen_by_similar_users)-set(list(map(str, Game_seen_by_user))))
        print('\n\n\n Games_under_consideration. Debe ser numerico \n', Games_under_consideration)
        Games_under_consideration = list(map(int, Games_under_consideration))
        score = []
        # print('\n\n\n **Games_under_consideration** \n', Games_under_consideration)

        # llenamos la prediccion del score
        for item in Games_under_consideration:
            c = final_game.loc[:, str(item)]
            d = c[c.index.isin(b)]
            f = d[d.notnull()]
            avg_user = Mean.loc[Mean['user'] == user, 'rate'].values[0]
            index = f.index.values.squeeze().tolist()
            corr = similarity_with_game.loc[user, index]
            fin = pd.concat([f, corr], axis=1)
            fin.columns = ['adg_score', 'correlation']
            fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
            nume = fin['score'].sum()
            deno = fin['correlation'].sum()
            final_score = avg_user + (nume/deno)
            score.append(final_score)

        data = pd.DataFrame(
            {'game': Games_under_consideration, 'score': score})
        # Aqui decidimos cuantos juegos recomendar
        print('\n\n\n **data** \n', data)
        print('\n\n\n **games** \n', games)
        top_recomendations = data.sort_values(
            by='score', ascending=False).head(games)
        print('\n\n\n **top_recomendations** \n', top_recomendations)
        # print('\n\n\n **games** \n', games)
        # print('\n\n\n top 5 pd- \n\n', top_recomendations)
        recomendations_n_6 = top_recomendations['game'].values.tolist()
        return recomendations_n_6

    # esta es la ultima parte de la respuesta de la API: Llama a la funcion y entrega el resultado    
    print('\n\n\n Hola, estoy en 121--- \n')
    recomendacionFinal = recomendation_kubnn(user)
    print('\n\n\n Hola, estoy en 123--- \n')
    data = {'recomendations': recomendacionFinal}
    print('\n\n\n Resultado--- \n', data)
    return jsonify(data), 200


@app.route('/live')
def live():
    return 'Hey, hola ahi. Estoy vivo.'


# Ejecucion de la aplicacion
if __name__ == 'main':
    app.run(debug = True)
