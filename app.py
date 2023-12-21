# from turtle import width
# from matplotlib.ft2font import HORIZONTAL
# from matplotlib.style import use
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import date
from datetime import datetime
import difflib
from streamlit_option_menu import option_menu
# from PIL import Image
# import collections
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import altair as alt
import re
import time

st.set_page_config(page_title='Movie Recoms', page_icon='clapper')

st.markdown("""
<style>
MainMenu{visibility:hidden;}
footer{visibility:visible;}
footer:after{content:'Created by Gunny';
display:block;
position:relative;
color:tomato;
padding:5px;
top:3px;}</style>""", unsafe_allow_html=True)

with st.sidebar:
    sel = option_menu(menu_title='Main Menu',
                      options=['Home', 'Multi-Select Movies', 'Actors', 'Director', 'Year', 'About'], \
                      menu_icon=['list'],
                      icons=['house', 'check-all', 'person-bounding-box', 'person-workspace', 'calendar-date',
                             'info-circle'],
                      default_index=0)


# @st.cache_data()
# def data_load():
#     # sim_1 = pickle.load(open('sim_1.pkl', 'rb'))
#     # sim_2 = pickle.load(open('sim_2.pkl', 'rb'))
#     # sim_3 = pickle.load(open('sim_3.pkl', 'rb'))
#     # sim_4 = pickle.load(open('sim_4.pkl', 'rb'))
#     # sim_5 = pickle.load(open('sim_5.pkl', 'rb'))
#     # sim_6 = pickle.load(open('sim_6.pkl', 'rb'))
#     # sim_7 = pickle.load(open('sim_7.pkl', 'rb'))
#     # sim_8 = pickle.load(open('sim_8.pkl', 'rb'))
#     # sim_9 = pickle.load(open('sim_9.pkl', 'rb'))
#     # sim_10 = pickle.load(open('sim_10.pkl', 'rb'))
#     # sim_11 = pickle.load(open('sim_11.pkl', 'rb'))
#     # sim_12 = pickle.load(open('sim_12.pkl', 'rb'))
#     # final = pickle.load(open('df_final.pkl', 'rb'))
#     # actor = pickle.load(open('df_actor.pkl', 'rb'))
#     # eda_df = pickle.load(open('eda_df.pkl', 'rb'))
#     sim_1 = pd.read_pickle('sim_1.pkl')
#     sim_2 = pd.read_pickle('sim_2.pkl')
#     sim_3 = pd.read_pickle('sim_3.pkl')
#     sim_4 = pd.read_pickle('sim_4.pkl')
#     sim_5 = pd.read_pickle('sim_5.pkl')
#     sim_6 = pd.read_pickle('sim_6.pkl')
#     sim_7 = pd.read_pickle('sim_7.pkl')
#     sim_8 = pd.read_pickle('sim_8.pkl')
#     sim_9 = pd.read_pickle('sim_9.pkl')
#     sim_10 = pd.read_pickle('sim_10.pkl')
#     # sim_11 = pd.read_pickle('sim_11.pkl')
#     # sim_12 = pd.read_pickle('sim_12.pkl')
#     final = pd.read_pickle('df_final.pkl')
#     actor = pd.read_pickle('df_actor.pkl')
#     eda_df = pd.read_pickle('eda_df.pkl')

#     sim_mat = np.concatenate([sim_1, sim_2, sim_3, sim_4, sim_5, sim_6, sim_7, sim_8, sim_9, sim_10])

#     return  sim_mat, final, actor, eda_df

# sim_mat, final, actor, eda_df = data_load()

def data_load1():
    sim_1 = pickle.load(open('new_sim1.pkl', 'rb'))
    sim_2 = pickle.load(open('new_sim2.pkl', 'rb'))
    sim_3 = pickle.load(open('new_sim3.pkl', 'rb'))
    sim_4 = pickle.load(open('new_sim4.pkl', 'rb'))
    # sim_5 = pickle.load(open('sim_5.pkl', 'rb'))
    # sim_6 = pickle.load(open('sim_6.pkl', 'rb'))
    # sim_7 = pickle.load(open('sim_7.pkl', 'rb'))
    # sim_8 = pickle.load(open('sim_8.pkl', 'rb'))
    # sim_9 = pickle.load(open('sim_9.pkl', 'rb'))
    # sim_10 = pickle.load(open('sim_10.pkl', 'rb'))
    # sim_11 = pickle.load(open('sim_11.pkl', 'rb'))
    # sim_12 = pickle.load(open('sim_12.pkl', 'rb'))

    sim_mat = np.concatenate([sim_1, sim_2, sim_3, sim_4])
    return sim_mat

def data_load2():
    final = pd.read_pickle('final_new2.pkl')
    actor = pd.read_pickle('df_actor.pkl')
    eda_df = pd.read_pickle('eda_df.pkl')
    return final,actor,eda_df


sim_mat = data_load1()
final,actor,eda_df = data_load2()













def recommend(movie):
    index = final[final.key == movie].index[0]
    results = sorted(list(enumerate(sim_mat[index])),key=lambda x :x[1])

    idx = []
    for i in results[:13]:
        idx.append(i[0])


    return final.iloc[idx]['title'].values[1:],'https://image.tmdb.org/t/p/w500'+final.iloc[idx]['poster'].values[1:],final.iloc[idx]['key'].values[1:]


if sel == 'Home':

    # st.title('MOVIE RECOMMENDATION SYSTEM')
    st.markdown("""
    <h3 style = font-size:270%;text-align:center;color:darkslategray;>
    MOVIE RECOMMENDATION SYSTEM
    </h3>
    """, unsafe_allow_html=True)
    name = st.selectbox(label='Please select the Movie', options=final['key'], index=2500,
                        help='Select Movie from below to get recommendations')

    st.subheader(name)
    st.markdown("""
    <h2 style=color:teal;text-align:center;> {}</h1>""".format(name), unsafe_allow_html=True)

    index = final[final.key == name].index[0]
    poster_path = 'https://image.tmdb.org/t/p/w500' + final.iloc[index]['poster']
    year = final.iloc[index]['year']
    runtime = final.iloc[index]['runtime']
    rating = final.iloc[index]['rating']
    genre = final.iloc[index]['genre']
    summary = final.iloc[index]['overview']
    director = final.iloc[index]['director']
    cast = final.iloc[index]['cast']
    num_votes = final.iloc[index]['num_votes']
    budget = final.iloc[index]['budget']
    revenue = final.iloc[index]['revenue']

    col1, col2 = st.columns(2)

    with col1:
        st.image(poster_path)

    with col2:
        st.markdown('**`Name`** : {}'.format(final.iloc[index]['title']))
        st.markdown('**`Year`** : {}'.format(year))
        st.markdown('**`Runtime`** : {}'.format(runtime))
        st.markdown('**`Genre`** : {}'.format(genre))
        st.markdown('**`Rating`** : {} ({} Votes)'.format(rating, num_votes))
        st.markdown('**`Budget`** : {}'.format(budget))
        st.markdown('**`Revenue`** : {}'.format(revenue))
        st.markdown('**`Summary`** : {}'.format(summary))
        st.markdown('**`Director`** : {}'.format(director))

    len_cast = len(cast.split(','))
    cast_names = cast.split(',')


    def age_cast(birthdate):
        today = date.today()
        d = datetime.strptime(birthdate, "%Y-%m-%d")
        age = ((today - d.date()) / 365).days
        return str(age) + ' Years'


    api_key = 'ef9ce1abb955e162c424955afe1df5a7'
    st.markdown('#### CAST :')

    for i, k in enumerate(st.columns(len_cast)):
        name_cast = cast_names[i]
        name_cast = difflib.get_close_matches(name_cast, actor.name)[0]
        cast_idx = actor[actor.name == name_cast].index[0]
        poster_path_cast = actor.iloc[cast_idx]['poster']
        k.image(poster_path_cast)
        if k.button(cast_names[i], key=i):

            cast_id = actor.iloc[cast_idx]['id']
            res = requests.get(
                'https://api.themoviedb.org/3/person/{}?api_key={}&language=en-US'.format(cast_id, api_key))
            data = res.json()
            age = age_cast(data['birthday'])
            k.markdown('*Age* : {}'.format(age))
            k.markdown('*Born* : {}'.format(data['place_of_birth']))
            # k.markdown('*Biography* : {}'.format(data['biography']))
            st.text_area('Bio', data['biography'], height=200)
            st.markdown('*Best Known For:*')

            known_for_idx = []

            for i, k in enumerate(final.cast):
                for j in k.split(','):
                    if re.search(name_cast + '$', j):
                        known_for_idx.append(i)

            known_posters = final.iloc[known_for_idx].sort_values(by='num_votes_num', ascending=False)[:4][
                'poster'].values
            known_title = final.iloc[known_for_idx].sort_values(by='num_votes_num', ascending=False)[:4]['title'].values
            for i, cols in enumerate(st.columns(len(known_posters))):
                cols.image('https://image.tmdb.org/t/p/w500' + known_posters[i], caption=known_title[i], width=170)

    with st.sidebar:
        if st.button('Clear All Dropdowns'):
            st.write(' ')

    st.subheader('Recommendations For You :')

    titles, posters, title_key = recommend(name)

    c = 0
    for i in range(0, 12, 2):
        for col in st.columns(2):
            col.image(posters[c], width=250, caption=titles[c])

            # col.markdown('**{}**'.format(titles[c]))

            # if col.button('Find Similar Movies', key=c):
            #     t, p, k = recommend(title_key[c])
            #     with st.expander('Recoms :', expanded=True):
            #         w = 0
            #         for k in st.columns(5):
            #             k.image(p[w], caption=t[w], width=80)
            #             w += 1
            #
            #         # st.image(p[x],width=100,caption=t[x])
            #
            c += 1

if sel == 'Year':
    st.markdown("""
    <h1 style ='color:DarkSlateGray;font-size:250%;text-align:center;'>
    SEARCH BY YEAR
    </h1>""", unsafe_allow_html=True)
    year_range = st.slider(label='Select Year', min_value=1950, max_value=2021, step=1, value=[2005, 2021])

    start = year_range[0]
    end = year_range[1]
    eda = eda_df[eda_df.year.between(start, end, inclusive=True)]

    radio_sel = st.radio(label='', options=['No. Of Movies', 'Revenue By Year'], index=0, horizontal=True)
    if radio_sel == 'No. Of Movies':
        a = px.bar(data_frame=eda, x='values', y='year', labels={'values': 'No. Of Movies', 'year': 'Year'},
                   color='values', \
                   height=1000, width=750, orientation='h', title='NO. OF MOVIES BY YEAR', color_continuous_scale='ice')
        a.update_yaxes(nticks=len(eda) + 1)
        st.plotly_chart(a, use_container_width=False)

        if st.button(label='Raw Data'):
            st.dataframe(eda[['year', 'values']])

    if radio_sel == 'Revenue By Year':
        b = px.scatter(data_frame=eda, x='year', y='revenue', size='revenue', size_max=25, color='revenue',
                       color_continuous_scale='picnic', \
                       labels={'year': 'YEAR', 'revenue': 'REVENUE'}, title='REVENUE BY YEAR', hover_name='title',
                       template='seaborn')
        st.plotly_chart(b, use_container_width=True)

        if st.button(label='Raw Data'):
            st.dataframe(eda[['year', 'revenue', 'title']])

if sel == 'Multi-Select Movies':

    st.markdown("""
    <h3 style='color:darkslategray;font-size:250%;text-align:center;'>
    MULTI-MOVIE RECOMMENDATION
    </h3>""", unsafe_allow_html=True)

    st.caption('You can get more accurate recommendations by selecting more than one movie.')
    st.write(' ')
    col1, col2 = st.columns([1, 4])
    no_movies = col1.selectbox(label='No. of movies', options=range(2, 9), index=2)
    multi_movies = col2.multiselect(label='Select your favoutive movies below :', options=final.key,
                                    default=['The Pursuit of Happyness (2006)', 'The Social Network (2010)', \
                                             'The Big Short (2015)', 'A Beautiful Mind (2001)'])

    multi_movies_idx = []
    for i in multi_movies:
        multi_movies_idx.append(final[final.key == i].index[0])

    multi_movies_titles = final.iloc[multi_movies_idx]['title'].values
    multi_movies_ratings = final.iloc[multi_movies_idx]['rating'].values
    multi_movies_votes = final.iloc[multi_movies_idx]['num_votes'].values
    if len(multi_movies) == no_movies:

        if no_movies <= 4:
            for i, col in enumerate(st.columns(no_movies)):
                col.image('https://image.tmdb.org/t/p/w500' + final[final.key == multi_movies[i]]['poster'].values[0],
                          width=150,
                          caption=multi_movies_titles[i] + ' ' + '\n' + str(multi_movies_ratings[i]) + '(' + str(
                              multi_movies_votes[i]) + ' Votes)')

        if no_movies > 4:
            for i, col in enumerate(st.columns(4)):
                col.image('https://image.tmdb.org/t/p/w500' + final[final.key == multi_movies[i]]['poster'].values[0],
                          width=150,
                          caption=multi_movies_titles[i] + ' ' + '\n' + str(multi_movies_ratings[i]) + '(' + str(
                              multi_movies_votes[i]) + ' Votes)')

            for i, col in enumerate(st.columns(4)):
                try:
                    col.image(
                        'https://image.tmdb.org/t/p/w500' + final[final.key == multi_movies[4:][i]]['poster'].values[0],
                        width=150,
                        caption=multi_movies_titles[4:][i] + ' ' + '\n' + str(multi_movies_ratings[4:][i]) + '(' + str(
                            multi_movies_votes[4:][i]) + ' Votes)')

                except:
                    st.write('')

    # with st.expander(label='Get Reommendations:'):
    radio_sel = st.radio(label=' ', options=('Show Recommendations', "Don't Show Recommendations"), horizontal=True,
                         index=1)
    if radio_sel == 'Show Recommendations':
        s = np.zeros((11321,))
        for i in multi_movies_idx:
            s += sim_mat[i]

        multi_results = sorted(list(enumerate(s)), key=lambda x: x[1])
        multi_results_idx = []
        for j in multi_results:
            multi_results_idx.append(j[0])

        for i in multi_movies_idx:
            multi_results_idx.remove(i)
        multi_results_idx = multi_results_idx[:12]
        multi_results_posters = final.iloc[multi_results_idx]['poster'].values
        multi_results_title = final.iloc[multi_results_idx]['title'].values
        multi_results_rating = final.iloc[multi_results_idx]['rating'].values
        multi_results_votes = final.iloc[multi_results_idx]['num_votes'].values
        multi_results_budget = final.iloc[multi_results_idx]['budget'].values
        multi_results_year = final.iloc[multi_results_idx]['year'].values
        multi_results_runtime = final.iloc[multi_results_idx]['runtime'].values
        multi_results_genre = final.iloc[multi_results_idx]['genre'].values
        multi_results_overview = final.iloc[multi_results_idx]['overview'].values
        multi_results_director = final.iloc[multi_results_idx]['director'].values
        multi_results_cast = final.iloc[multi_results_idx]['cast'].values
        multi_results_revenue = final.iloc[multi_results_idx]['revenue'].values

        q = 0
        for i in range(0, 12, 3):
            for col in st.columns(3):
                col.image('https://image.tmdb.org/t/p/w500' + multi_results_posters[q], caption=multi_results_title[q])

                if col.button(label='Get Movie Info', key=q + 100):
                    col.markdown(' *Year:* {}'.format(multi_results_year[q]), unsafe_allow_html=True)

                    col.markdown('_Runtime:_ {}'.format(multi_results_runtime[q]), unsafe_allow_html=True)
                    col.markdown('_Rating:_ {}({} Votes)'.format(multi_results_rating[q], multi_results_votes[q]),
                                 unsafe_allow_html=True)
                    col.markdown('_Genre:_ {}'.format(multi_results_genre[q]), unsafe_allow_html=True)

                    col.markdown('_Director:_ {}'.format(multi_results_director[q]), unsafe_allow_html=True)
                    col.markdown('_Cast:_ {}'.format(multi_results_cast[q]), unsafe_allow_html=True)
                    col.markdown('_Budget:_ {}'.format(multi_results_budget[q]), unsafe_allow_html=True)
                    col.markdown('_Revenue:_ {}'.format(multi_results_revenue[q]), unsafe_allow_html=True)
                    st.text_area(label='Summary:', value=multi_results_overview[q])

                q += 1

if sel == 'Actors':
    st.markdown("""
    <h1 style ='color:DarkSlateGray;font-size:250%;text-align:center;'>
    SEARCH BY ACTOR
    </h1>""", unsafe_allow_html=True)

    graph_sel = st.sidebar.selectbox(label='SEARCH BY :', options=['Year', 'Genre', 'Other Actors'])

    # st.cache()


    def age_actor(birthdate):
        today = date.today()
        d = datetime.strptime(birthdate, "%Y-%m-%d")
        age = ((today - d.date()) / 365).days
        return str(age) + ' Years'


    api_key = 'ef9ce1abb955e162c424955afe1df5a7'

    sel_actor = st.selectbox(label='', options=actor.name_imdb.values, index=14236)

    # st.cache()


    def actor_details(sel_actor):
        actor_idx = actor[actor.name_imdb == sel_actor].index[0]
        actor_path = actor.iloc[actor_idx]['poster']
        actor_id = actor.iloc[actor_idx]['id']

        col1, col2 = st.columns(2)
        with col1:

            st.image(actor_path, caption=sel_actor, width=300)

        res = requests.get('https://api.themoviedb.org/3/person/{}?api_key={}&language=en-US'.format(actor_id, api_key))
        data = res.json()
        age = age_actor(data['birthday'])

        with col2:
            for i in range(4):
                st.write('')
            st.markdown('*Age* : {}'.format(age))
            st.markdown('*Born* : {}'.format(data['place_of_birth']))
            # st.markdown('*Bio* : ')
            st.text_area('Bio', data['biography'], height=270)

        st.markdown('*Best Known For:*')

        known_for_idx = []

        for i, k in enumerate(final.cast):
            for j in k.split(','):
                if re.search(sel_actor + '$', j):
                    known_for_idx.append(i)

        known_posters = final.iloc[known_for_idx].sort_values(by='num_votes_num', ascending=False)[:4]['poster'].values
        known_title = final.iloc[known_for_idx].sort_values(by='num_votes_num', ascending=False)[:4]['title'].values
        for i, cols in enumerate(st.columns(len(known_posters))):
            cols.image('https://image.tmdb.org/t/p/w500' + known_posters[i], caption=known_title[i], width=170)

        return known_for_idx


    known_for_idx = actor_details(sel_actor)

    # st.cache()


    def actor_movies(idx):
        d_a = final.iloc[idx].reset_index(drop=True)
        return d_a


    d_a = actor_movies(known_for_idx)

    if graph_sel == 'Year':
        top_movies = st.sidebar.radio(label='Show :', options=['All Movies', 'Top Movies'], key=11)
        graph_sort = st.radio(label='Show values by :', options=['REVENUE', 'POPULARITY', 'RATING'], horizontal=True)

        # st.cache()


        def actor_year_graph(top_movies, graph_sort):
            if graph_sort == 'REVENUE':

                val = 'revenue_num'
            elif graph_sort == 'POPULARITY':
                val = 'num_votes_num'
            else:
                val = 'rating'

            if top_movies == 'All Movies':
                fig = px.bar(data_frame=d_a, x='year', y=val, hover_name='title',
                             labels={'year': 'Year', val: graph_sort, 'rating': 'Rating'}, hover_data=['rating'],
                             color=val, color_continuous_scale='ice', title='Movies By ' + graph_sort)
                fig.update_xaxes(type='category')
                fig.update_layout(title_font_family='cambria', title_font_size=14)
                st.plotly_chart(fig)

                with st.expander(label='Raw Data'):
                    st.write(d_a[['title', 'year', 'runtime', 'revenue', 'budget', 'num_votes', 'overview', 'cast',
                                  'director', 'genre']])

            elif top_movies == 'Top Movies':
                d_a2 = d_a.sort_values(by=['year', val], ascending=[True, False]).drop_duplicates('year').reset_index(
                    drop=True)

                fig = px.bar(data_frame=d_a2, x='year', y=val, hover_name='title',
                             labels={'year': 'Year', val: graph_sort, 'rating': 'Rating'}, hover_data=['rating'],
                             color=val, color_continuous_scale='ice', title='Movies By ' + graph_sort)
                fig.update_xaxes(type='category')
                fig.update_layout(title_font_family='cambria', title_font_size=14)
                st.plotly_chart(fig)

                with st.expander(label='Raw Data'):
                    st.write(d_a2[['title', 'year', 'runtime', 'revenue', 'budget', 'num_votes', 'overview', 'cast',
                                   'director', 'genre']])


        actor_year_graph(top_movies, graph_sort)

    if graph_sel == 'Genre':
        # st.cache()


        def genre_list(df):
            genre_list = []
            for i in df.genre:
                for j in i.split(' '):
                    genre_list.append(j.strip())
            return set(genre_list)


        genre_l = genre_list(d_a)

        c1, c2 = st.columns([5, 1])

        with c1:

            genre_sel = st.multiselect(label='Select Genre', options=genre_l)
        with c2:
            st.write(' ')
            st.write(' ')
            search = st.button('Search')

        # st.cache()


        def genre_sel_idx(genre_sel):
            g_idx = []
            for g in genre_sel:
                for i, k in enumerate(d_a['genre']):
                    for j in k.split(' '):
                        if re.search(g + '$', j):
                            g_idx.append(i)
            return g_idx


        g_idx = genre_sel_idx(genre_sel)
        if search == True:
            colors = px.colors.named_colorscales()
            fig3 = px.scatter(data_frame=d_a.iloc[g_idx], x='title', y='revenue_num', size='num_votes_num',
                              hover_name='title', \
                              color_continuous_scale=colors[np.random.randint(0, 95)], color='rating',
                              labels={'title': 'Movie', 'revenue_num': 'Revenue', 'num_votes_num': 'Popularity',
                                      'rating': 'Rating', 'year': 'Year'}, \
                              hover_data=['year'], size_max=50, title='Movies By Revenue VS Popularity VS Rating')
            fig3.update_xaxes(type='category')
            fig3.update_layout(title_font_family='cambria', title_font_size=14)

            st.plotly_chart(fig3)
        with st.expander('raw data'):
            st.write(d_a.iloc[g_idx][
                         ['title', 'year', 'runtime', 'revenue', 'budget', 'num_votes', 'overview', 'cast', 'director',
                          'genre']].drop_duplicates().reset_index(drop=True))

    if graph_sel == 'Other Actors':
        other_actor_sel = st.selectbox(label='Select Actor:', options=actor.name)


        # st.title('Sort by 4 options')
        # @st.cache_data
        def other_actor_index(name):
            actor2_idx = []
            for k, i in enumerate(d_a.cast):
                for j in i.split(','):
                    if re.search(name + '$', j):
                        actor2_idx.append(k)
            return actor2_idx


        actor2_idx = other_actor_index(other_actor_sel)
        if actor2_idx == []:
            st.write('No Movies Found')
        else:
            sortby = st.radio(label='Sort By:', options=['Year', 'Rating', 'Revenue', 'Popularity'], horizontal=True)
            if sortby == 'Year':
                d_b = d_a.iloc[actor2_idx].sort_values(by='year').reset_index(drop=True)
            elif sortby == 'Rating':
                d_b = d_a.iloc[actor2_idx].sort_values(by='rating', ascending=False).reset_index(drop=True)
            elif sortby == 'Revenue':
                d_b = d_a.iloc[actor2_idx].sort_values(by='revenue_num', ascending=False).reset_index(drop=True)
            elif sortby == 'Popularity':
                d_b = d_a.iloc[actor2_idx].sort_values(by='num_votes_num', ascending=False).reset_index(drop=True)
            # st.write(d_a.iloc[actor2_idx])
            # d_b = d_a.iloc[actor2_idx]
            d_b_name = d_b['key'].values
            d_b_year = d_b['year'].values
            d_b_runtime = d_b['runtime'].values
            d_b_rating = d_b['rating'].values
            d_b_votes = d_b['num_votes'].values
            d_b_director = d_b['director'].values
            d_b_cast = d_b['cast'].values
            d_b_genre = d_b['genre'].values
            d_b_overview = d_b['overview'].values
            for i in range(d_b.shape[0]):
                colu1, colu2 = st.columns([1, 3])
                with colu1:
                    colu1.image('https://image.tmdb.org/t/p/w500' + d_b['poster'].values[i])
                with colu2:
                    st.markdown('_Name_ : {}'.format(d_b_name[i]))
                    st.markdown('_Runtime_ : {}'.format(d_b_runtime[i]))
                    st.markdown('_Rating_ : {} ({} Votes)'.format(d_b_rating[i], d_b_votes[i]))
                    st.markdown('_Cast_ : {}'.format(d_b_cast[i]))
                    st.markdown('_Director_ : {}'.format(d_b_director[i]))
                    st.markdown('_Genre_ : {}'.format(d_b_genre[i]))
                    # st.markdown('_Summary_ :'.format(d_b_overview[i]))
                    # st.text_area(label='Summary',value=d_b_overview[i])

            with st.expander(label='Raw Data '):
                st.write(d_b.reset_index(drop=True))

if sel == 'Director':
    st.markdown("""
    <h1 style ='color:DarkSlateGray;font-size:250%;text-align:center;'>
    SEARCH BY DIRECTOR
    </h1>""", unsafe_allow_html=True)
    graph_sel = st.sidebar.selectbox(label='Search By :', options=['Year', 'Genre', 'Other Actors'])


    # @st.cache_data
    def director_list():
        direc = pickle.load(open('df_director.pkl', 'rb'))

        return direc['name'], direc['path'], direc['bio'], direc['birthday'], direc['death'], direc['place']


    def age_direc(birthdate):
        today = date.today()
        d = datetime.strptime(birthdate, "%Y-%m-%d")
        age = ((today - d.date()) / 365).days
        return str(age) + ' Years'


    direc_names, direc_path, direc_bio, direc_birthday, direc_death, direc_place = director_list()

    sel_direc = st.selectbox(label='', options=direc_names, index=921)
    # st.cache()


    def director_details():
        direc_idx = np.where(direc_names == sel_direc)[0][0]
        col1, col2 = st.columns(2)
        with col1:
            st.image(direc_path[direc_idx], caption=direc_names[direc_idx], width=300)
        with col2:
            try:
                for i in range(4):
                    st.write('')
                st.markdown('_Age_ : {}'.format(age_direc(direc_birthday[direc_idx])))
                st.markdown('_Born_ :{}'.format(direc_place[direc_idx]))
                st.text_area('Bio :', direc_bio[direc_idx], height=270)
            # st.markdown(direc_bio[direc_idx])

            except:
                st.write('No Info Available')
        try:
            st.markdown('##### **Best Know For:**')
            known_for_idx = []

            for i, k in enumerate(final.director):
                for j in k.split(','):
                    if re.search(sel_direc + '$', j):
                        known_for_idx.append(i)

            known_posters = final.iloc[known_for_idx].sort_values(by='num_votes_num', ascending=False)[:4][
                'poster'].values
            known_title = final.iloc[known_for_idx].sort_values(by='num_votes_num', ascending=False)[:4]['title'].values
            for i, cols in enumerate(st.columns(len(known_posters))):
                cols.image('https://image.tmdb.org/t/p/w500' + known_posters[i], caption=known_title[i], width=170)

        except:
            st.write('No movies found.')
        return known_for_idx


    known_for_idx = director_details()

    # st.cache()


    def director_movies(idx):
        q_a = final.iloc[idx].reset_index(drop=True)
        return q_a


    q_a = director_movies(known_for_idx)
    if graph_sel == 'Year':

        top_movies = st.sidebar.radio(label='Show :', options=['All Movies', 'Top Movies'], key=11)
        graph_sort = st.radio(label='Show values by :', options=['REVENUE', 'POPULARITY', 'RATING'], horizontal=True)

        # st.cache()


        def director_year_graph(top_movies, graph_sort):
            if graph_sort == 'REVENUE':

                val = 'revenue_num'
            elif graph_sort == 'POPULARITY':
                val = 'num_votes_num'
            else:
                val = 'rating'

            if top_movies == 'All Movies':
                fig = px.bar(data_frame=q_a, x='year', y=val, hover_name='title',
                             labels={'year': 'Year', val: graph_sort, 'rating': 'Rating'}, hover_data=['rating'],
                             color=val, color_continuous_scale='ice', title='Movies By ' + graph_sort)
                fig.update_xaxes(type='category')
                fig.update_layout(title_font_family='cambria', title_font_size=14)
                st.plotly_chart(fig)

                with st.expander(label='Raw Data'):
                    st.write(q_a[['title', 'year', 'runtime', 'revenue', 'budget', 'num_votes', 'overview', 'cast',
                                  'director', 'genre']])

            elif top_movies == 'Top Movies':
                q_a2 = q_a.sort_values(by=['year', val], ascending=[True, False]).drop_duplicates('year').reset_index(
                    drop=True)

                fig = px.bar(data_frame=q_a2, x='year', y=val, hover_name='title',
                             labels={'year': 'Year', val: graph_sort, 'rating': 'Rating'}, hover_data=['rating'],
                             color=val, color_continuous_scale='ice', title='Movies By ' + graph_sort)
                fig.update_xaxes(type='category')
                fig.update_layout(title_font_family='cambria', title_font_size=14)
                st.plotly_chart(fig)

                with st.expander(label='Raw Data'):
                    st.write(q_a2[['title', 'year', 'runtime', 'revenue', 'budget', 'num_votes', 'overview', 'cast',
                                   'director', 'genre']])


        director_year_graph(top_movies, graph_sort)

    if graph_sel == 'Genre':
        # st.cache()


        def genre_list(df):
            genre_list = []
            for i in df.genre:
                for j in i.split(' '):
                    genre_list.append(j.strip())
            return set(genre_list)


        genre_l = genre_list(q_a)

        c1, c2 = st.columns([5, 1])

        with c1:

            genre_sel = st.multiselect(label='Select Genre', options=genre_l)
        with c2:
            st.write(' ')
            st.write(' ')
            search = st.button('Search')

        # st.cache()


        def genre_sel_idx(genre_sel):
            g_idx = []
            for g in genre_sel:
                for i, k in enumerate(q_a['genre']):
                    for j in k.split(' '):
                        if re.search(g + '$', j):
                            g_idx.append(i)
            return g_idx


        g_idx = genre_sel_idx(genre_sel)
        if search == True:
            colors = px.colors.named_colorscales()
            fig3 = px.scatter(data_frame=q_a.iloc[g_idx], x='title', y='revenue_num', size='num_votes_num',
                              hover_name='title', \
                              color_continuous_scale=colors[np.random.randint(0, 95)], color='rating',
                              labels={'title': 'Movie', 'revenue_num': 'Revenue', 'num_votes_num': 'Popularity',
                                      'rating': 'Rating', 'year': 'Year'}, \
                              hover_data=['year'], size_max=50, title='Movies By Revenue VS Popularity VS Rating')
            fig3.update_xaxes(type='category')
            fig3.update_layout(title_font_family='cambria', title_font_size=14)

            st.plotly_chart(fig3)
        with st.expander('raw data'):
            st.write(q_a.iloc[g_idx][
                         ['title', 'year', 'runtime', 'revenue', 'budget', 'num_votes', 'overview', 'cast', 'director',
                          'genre']].drop_duplicates().reset_index(drop=True))

    if graph_sel == 'Other Actors':
        other_actor_sel = st.selectbox(label='Select Actor:', options=actor.name)


        # st.title('Sort by 4 options')
        # @st.cache_data
        def other_actor_index(name):
            actor2_idx = []
            for k, i in enumerate(q_a.cast):
                for j in i.split(','):
                    if re.search(name + '$', j):
                        actor2_idx.append(k)
            return actor2_idx


        actor2_idx = other_actor_index(other_actor_sel)
        if actor2_idx == []:
            st.write('No Movies Found')
        else:
            sortby = st.radio(label='Sort By:', options=['Year', 'Rating', 'Revenue', 'Popularity'], horizontal=True)
            if sortby == 'Year':
                q_b = q_a.iloc[actor2_idx].sort_values(by='year').reset_index(drop=True)
            elif sortby == 'Rating':
                q_b = q_a.iloc[actor2_idx].sort_values(by='rating', ascending=False).reset_index(drop=True)
            elif sortby == 'Revenue':
                q_b = q_a.iloc[actor2_idx].sort_values(by='revenue_num', ascending=False).reset_index(drop=True)
            elif sortby == 'Popularity':
                q_b = q_a.iloc[actor2_idx].sort_values(by='num_votes_num', ascending=False).reset_index(drop=True)
            # st.write(d_a.iloc[actor2_idx])
            # d_b = d_a.iloc[actor2_idx]
            q_b_name = q_b['key'].values
            q_b_year = q_b['year'].values
            q_b_runtime = q_b['runtime'].values
            q_b_rating = q_b['rating'].values
            q_b_votes = q_b['num_votes'].values
            q_b_director = q_b['director'].values
            q_b_cast = q_b['cast'].values
            q_b_genre = q_b['genre'].values
            q_b_overview = q_b['overview'].values
            for i in range(q_b.shape[0]):
                colu1, colu2 = st.columns([1, 3])
                with colu1:
                    colu1.image('https://image.tmdb.org/t/p/w500' + q_b['poster'].values[i])
                with colu2:
                    st.markdown('_Name_ : {}'.format(q_b_name[i]))
                    st.markdown('_Runtime_ : {}'.format(q_b_runtime[i]))
                    st.markdown('_Rating_ : {} ({} Votes)'.format(q_b_rating[i], q_b_votes[i]))
                    st.markdown('_Cast_ : {}'.format(q_b_cast[i]))
                    st.markdown('_Director_ : {}'.format(q_b_director[i]))
                    st.markdown('_Genre_ : {}'.format(q_b_genre[i]))
                    # st.markdown('_Summary_ :'.format(d_b_overview[i]))
                    # st.text_area(label='Summary',value=d_b_overview[i])

            with st.expander(label='Raw Data '):
                st.write(q_b.reset_index(drop=True))

if sel == 'About':

    st.markdown("""
        <h3 style = font-size:270%;text-align:center;color:darkslategray;>
        ABOUT
        </h3>
        """, unsafe_allow_html=True)
    st.write("This recommendation system was created Gunpreet Singh")
    st.write("Email: gunpreetsingh72@gmail.com")

