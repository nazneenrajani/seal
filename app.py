## LIBRARIES ###
## Data
import numpy as np
from numpy.core.numeric import outer
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from math import floor
from collections import defaultdict
from transformers import AutoTokenizer
#pd.set_option('precision', 2)
#pd.options.display.float_format = '${:,.2f}'.format

# Analysis
# from gensim.models.doc2vec import Doc2Vec
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import nltk
from nltk.cluster import KMeansClusterer
import scipy.spatial.distance as sdist
from scipy.spatial import distance_matrix
# nltk.download('punkt') #make sure that punkt is downloaded

# App & Visualization
import streamlit as st
import altair as alt
import plotly.graph_objects as go
from streamlit_vega_lite import altair_component


# utils
from random import sample
from seal import utils as ut


def down_samp(embedding):
    """Down sample a data frame for altiar visualization """
    # total number of positive and negative sentiments in the class
    #embedding = embedding.groupby('slice').apply(lambda x: x.sample(frac=0.3))
    total_size = embedding.groupby(['slice', 'label'], as_index=False).count()

    user_data = 0
    # if 'Your Sentences' in str(total_size['slice']):
    #     tmp = embedding.groupby(['slice'], as_index=False).count()
    #     val = int(tmp[tmp['slice'] == "Your Sentences"]['source'])
    #     user_data = val

    max_sample = total_size.groupby('slice').max()['content']

    # # down sample to meeting altair's max values
    # # but keep the proportional representation of groups
    down_samp = 1/(sum(max_sample.astype(float))/(1000-user_data))

    max_samp = max_sample.apply(lambda x: floor(
        x*down_samp)).astype(int).to_dict()
    max_samp['Your Sentences'] = user_data

    # # sample down for each group in the data frame
    embedding = embedding.groupby('slice').apply(
        lambda x: x.sample(n=max_samp.get(x.name))).reset_index(drop=True)

    # # order the embedding
    return(embedding)

#down sample low loss points only so misclassified examples are not down sampled in viz


def down_samp_ll(embedding):
    df_ll = embedding[embedding['slice'] == 'low-loss']
    #if(len(df_ll)<5000):
    #    return embedding
    #else:
    df_hl = embedding[embedding['slice'] == 'high-loss']
    down_samp = len(df_ll) - (1000-len(df_hl))
    df_ll.sample(n=down_samp)
    embedding.drop(df_ll.index)
    return embedding


def data_comparison(df):
    selection = alt.selection_multi(fields=['cluster', 'label'])
    color = alt.condition(alt.datum.slice == 'high-loss', alt.Color('cluster:N', scale=alt.Scale(
        domain=df.cluster.unique().tolist()), legend=None), alt.value("lightgray"))
    opacity = alt.condition(selection, alt.value(0.7), alt.value(0.25))

    # basic chart
    scatter = alt.Chart(df).mark_point(size=100, filled=True).encode(
        x=alt.X('x:Q', axis=None),
        y=alt.Y('y:Q', axis=None),
        color=color,
        shape=alt.Shape('label:N', scale=alt.Scale(
            range=['circle', 'diamond'])),
        tooltip=['cluster:N', 'slice:N', 'content:N', 'label:N', 'pred:N'],
        opacity=opacity
    ).properties(
        width=1000,
        height=800
    ).interactive()

    legend = alt.Chart(df).mark_point(size=100, filled=True).encode(
        x=alt.X("label:N"),
        y=alt.Y('cluster:N', axis=alt.Axis(
            orient='right'), sort='ascending', title=''),
        shape=alt.Shape('label:N', scale=alt.Scale(
            range=['circle', 'diamond']), legend=None),
        color=color,
    ).add_selection(
        selection
    )
    layered = scatter | legend
    layered = layered.configure_axis(
        grid=False
    ).configure_view(
        strokeOpacity=0
    )

    content = legend.encode(text='content:N')

    return layered


def viz_panel(embedding_df):
    """ Visualization Panel Layout"""
    all_metrics = {}
    st.warning("**Error group visualization**")
    with st.expander("How to read this chart:"):
        st.markdown("* Each **point** is an input example.")
        st.markdown("* Gray points have low-loss and the colored have high-loss. High-loss instances are clustered using **kmeans** and each color represents a cluster.")
        st.markdown(
            "* The **shape** of each point reflects the label category --  positive (diamond) or negative sentiment (circle).")
    #st.altair_chart(data_comparison(down_samp(embedding_df)), use_container_width=True)
    viz = data_comparison(embedding_df)
    st.altair_chart(viz, use_container_width=True)

@st.cache()
def frequent_tokens(data, tokenizer, loss_quantile=0.95, top_k=200, smoothing=0.005):
    unique_tokens = []
    tokens = []
    for row in tqdm(data['content']):
        tokenized = tokenizer(row, padding=True, truncation=True, return_tensors='pt')
        tokens.append(tokenized['input_ids'].flatten())
        unique_tokens.append(torch.unique(tokenized['input_ids']))
    losses = data['loss'].astype(float)
    high_loss = losses.quantile(loss_quantile)
    loss_weights = np.where(losses > high_loss,losses,0.0)
    loss_weights = loss_weights / loss_weights.sum()

    token_frequencies = defaultdict(float)
    token_frequencies_error = defaultdict(float)
    weights_uniform = np.full_like(loss_weights, 1 / len(loss_weights))

    for i in tqdm(range(len(data))):
        for token in unique_tokens[i]:
            token_frequencies[token.item()] += weights_uniform[i]
            token_frequencies_error[token.item()] += loss_weights[i]

    token_lrs = {k: (smoothing+token_frequencies_error[k]) / (
        smoothing+token_frequencies[k]) for k in token_frequencies}
    tokens_sorted = list(map(lambda x: x[0], sorted(
        token_lrs.items(), key=lambda x: x[1])[::-1]))

    top_tokens = []
    for i, (token) in enumerate(tokens_sorted[:top_k]):
        top_tokens.append(['%10s' % (tokenizer.decode(token)), '%.4f' % (token_frequencies[token]), '%.4f' % (
            token_frequencies_error[token]), '%4.2f' % (token_lrs[token])])
    return pd.DataFrame(top_tokens, columns=['token', 'freq', 'error-freq', 'ratio'])


def load_precached_groups(data_ll, df_list, num_clusters, group_dict_path, group_idx_path, num_points=1000):
    merged = dynamic_groups(df_list, num_clusters)
    down_samp = len(data_ll) - (num_points-len(merged))
    sample_idx = data_ll.sample(n=down_samp)
    data_ll = data_ll.drop(sample_idx.index)
    # put all the low loss data in one bigger cluster
    data_ll['cluster'] = merged.loc[merged['cluster'].idxmax()].cluster + 1
    merged = pd.concat([merged, data_ll])
    # merged['cluster'] = merged['cluster'].astype('str')
    # with open(group_dict_path, 'rb') as f:
    #     group_dict = pickle.load(f)
    # with open(group_idx_path, 'rb') as f:
    #     group_idx_dict = pickle.load(f)
    # for k,v in group_idx_dict.items():
    #     label = group_dict.get(k)
    #     merged.loc[merged.index.isin(v), ['cluster']] = label
    return merged


def dynamic_groups(df_list, num_clusters):
    merged = pd.DataFrame()
    ind = 0
    for df in df_list:
        kmeans_df, assigned_clusters = kmeans(df, num_clusters=num_clusters)
        kmeans_df['cluster'] = kmeans_df['cluster'] + ind*num_clusters
        ind = ind+1
        merged = pd.concat([merged, kmeans_df])
    return merged


@st.cache(ttl=600)
def get_data(inference, emb):
    preds = inference.outputs.numpy()
    losses = inference.losses.numpy()
    embeddings = pd.DataFrame(emb, columns=['x', 'y'])
    num_examples = len(losses)
    # dataset_labels = [dataset[i]['label'] for i in range(num_examples)]
    return pd.concat([pd.DataFrame(np.transpose(np.vstack([dataset[:num_examples]['content'],
                                                           dataset[:num_examples]['label'], preds, losses])), columns=['content', 'label', 'pred', 'loss']), embeddings], axis=1)


def kmeans(data, num_clusters=3):
    X = np.array(data['embedding'].to_list())
    kclusterer = KMeansClusterer(
        num_clusters, distance=nltk.cluster.util.cosine_distance,
        repeats=25, avoid_empty_clusters=True)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    data['cluster'] = pd.Series(
        assigned_clusters, index=data.index).astype('int')
    data['centroid'] = data['cluster'].apply(lambda x: kclusterer.means()[x])
    return data, assigned_clusters


def distance_from_centroid(row):
    return sdist.norm(row['embedding'] - row['centroid'].tolist())


@st.cache(ttl=600)
def craft_prompt(cluster_df):
    instruction = "In this task, we'll assign a short and precise label to a cluster of documents based on the topics or concepts most relevant to these documents. The documents are all subsets of a sentiment classification dataset.\n"
    if len(cluster_df) > 10:
        content = cluster_df['content'].str[:600].tolist()
    else:
        content = cluster_df['content'].str[:1000].tolist()
    examples = '\n - '.join(content)
    text = instruction + '- ' + examples + '\n Cluster label:'
    return text.strip()


@st.cache(ttl=600)
def topic_distribution(weights, smoothing=0.01):
    topic_frequencies = defaultdict(float)
    topic_frequencies_error = defaultdict(float)
    weights_uniform = np.full_like(weights, 1 / len(weights))
    num_examples = len(weights)
    for i in range(num_examples):
        example = dataset[i]
        category = example['title']
        topic_frequencies[category] += weights_uniform[i]
        topic_frequencies_error[category] += weights[i]

    topic_ratios = {c: (smoothing + topic_frequencies_error[c]) / (
        smoothing + topic_frequencies[c]) for c in topic_frequencies}

    categories_sorted = map(lambda x: x[0], sorted(
        topic_ratios.items(), key=lambda x: x[1], reverse=True))

    topic_distr = []
    for category in categories_sorted:
        topic_distr.append(['%.3f' % topic_frequencies[category], '%.3f' %
                           topic_frequencies_error[category], '%.2f' % topic_ratios[category], '%s' % category])

    return pd.DataFrame(topic_distr, columns=['Overall frequency', 'Error frequency', 'Ratio', 'Category'])


def populate_session(dataset, model):
    data_df = read_file_to_df(
        './assets/data/'+dataset + '_' + model+'.parquet')
    if model == 'albert-base-v2-yelp-polarity':
        tokenizer = AutoTokenizer.from_pretrained('textattack/'+model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
    # if "user_data" not in st.session_state:
    #     st.session_state["user_data"] = data_df
    # if "selected_slice" not in st.session_state:
    #     st.session_state["selected_slice"] = None
    return tokenizer


@st.cache(allow_output_mutation=True)
def read_file_to_df(file):
   return pd.read_parquet(file)


if __name__ == "__main__":
    ### STREAMLIT APP CONGFIG ###
    st.set_page_config(layout="wide", page_title="Interactive Error Analysis")

    ut.init_style()

    lcol, rcol = st.columns([5, 2])
    # ******* loading the mode and the data
    #st.sidebar.mardown("<h4>Interactive Error Analysis</h4>", unsafe_allow_html=True)

    dataset = st.sidebar.selectbox(
        "Dataset",
        ["amazon_polarity", "yelp_polarity", "imdb"],
        index=1
    )

    model = st.sidebar.selectbox(
        "Model",
        ["distilbert-base-uncased-finetuned-sst-2-english",
            "albert-base-v2-yelp-polarity", "distilbert-imdb"],
    )

    ### LOAD DATA AND TOKENIZER VARIABLES ###
    ##uncomment the next next line to run dynamically and not from file
    #tokenizer = populate_session(dataset, model)
    if dataset == 'imdb':
        data_df = read_file_to_df('./assets/data/imdb_distilbert.parquet')
    else:
        data_df = read_file_to_df(
            './assets/data/'+dataset + '_' + model+'.parquet')
        data_df = data_df[:20000]

    loss_quantile = st.sidebar.slider(
        "Loss Quantile", min_value=0.9, max_value=1.0, step=0.01, value=0.98
    )

    data_df['loss'] = data_df['loss'].astype(float)
    data_df['pred'] = data_df['pred'].astype(int)
    losses = data_df['loss']
    high_loss = losses.quantile(loss_quantile)
    data_df['slice'] = np.where(data_df['loss'] >= high_loss, 'high-loss', 'low-loss')
    # drop rows that are not hl
    data_hl = pd.DataFrame(data_df[data_df['slice'] == 'high-loss'])
    #data_hl = data_hl.drop(data_hl[data_hl.pred==data_hl.label].index)
    data_ll = pd.DataFrame(data_df[data_df['slice'] == 'low-loss'])
    # this is to allow clustering over each error type. fp, fn for binary classification
    df_list = [d for _, d in data_hl.groupby(['label'])]

    run_kmeans = st.sidebar.radio(
        "Cluster error group?", ('True', 'False'), index=0)

    num_clusters = st.sidebar.slider(
        "# clusters", min_value=1, max_value=60, step=1, value=3)

    num_points = st.sidebar.slider(
        "# data points to visualize", min_value=1000, max_value=5000, step=100, value=1000)

    selected_cluster = st.sidebar.number_input(
        label='Cluster #:', max_value=num_clusters-1, min_value=0)

    if run_kmeans == 'True':
        with st.spinner(text='running kmeans...'):
            group_dict_path = './assets/data/cluster-labels/'+dataset+'.pkl'
            group_idx_path = './assets/data/cluster-labels/'+dataset+'_idx.pkl'
            #data_hl_path = './assets/data/high-loss/'+dataset+'.parquet'
            merged = load_precached_groups(data_ll, df_list, int(
                (num_clusters/2)), group_dict_path, group_idx_path, num_points=num_points)
            #dynamic_groups(df_list,)
            #tmp = pd.concat([data_ll, merged], axis =0, ignore_index=True)

    cluster_content = craft_prompt(
        merged.loc[merged['cluster'] == selected_cluster])

    with lcol:
        st.markdown('<h5>Error Groups</h5>', unsafe_allow_html=True)
        with st.expander("How to read this table:"):
            st.markdown(
                "* *Error groups* refers to the subset of evaluation dataset the model performs poorly on.")
            st.markdown(
                "* The table displays model error groups on the evaluation dataset, sorted by loss.")
            st.markdown(
                "* Each row is an input example that includes the label, model pred, loss, and error group.")
        with st.spinner(text='loading error groups...'):
            #dataframe=read_file_to_df('./assets/data/'+dataset+ '_'+ model+'_error-slices.parquet')
            #uncomment the next next line to run dynamically and not from file
            dataframe = merged[['content', 'label', 'pred', 'loss', 'cluster']].sort_values(
                by=['loss'], ascending=False)
            #table_html = dataframe.to_html(columns=['content', 'label', 'pred', 'loss', 'cluster'], max_rows=50)
            #table_html = table_html.replace("<th>", '<th align="left">')  # left-align the headers
            st.write(dataframe.style.format(
                {'loss': '{:.2f}'}), width=1000, height=300)

    with rcol:
        with st.spinner(text='loading...'):
            st.markdown('<h5>Word Distribution in Error Groups</h5>',
                        unsafe_allow_html=True)
            #uncomment the next two lines to run dynamically and not from file
            # if model == 'albert-base-v2-yelp-polarity':
            #     tokenizer = AutoTokenizer.from_pretrained('textattack/'+model)
            # else:
            #     tokenizer = AutoTokenizer.from_pretrained(model)
            # commontokens = frequent_tokens(data_df, tokenizer, loss_quantile=loss_quantile)
            if dataset == 'imdb':
                commontokens = read_file_to_df('./assets/data/imdb_distilbert_commontokens.parquet')
            else:
                commontokens = read_file_to_df(
                    './assets/data/'+dataset + '_' + model+'_commontokens.parquet')
            with st.expander("How to read this table:"):
                st.markdown(
                    "* The table displays the most frequent tokens in error groups, relative to their frequencies in the val set.")

            st.write(commontokens)

    with st.spinner(text='loading visualization...'):
        viz_panel(merged)

    st.sidebar.download_button(
        data=cluster_content,
        label="Build prompt from data",
        file_name='prompt'
    )
