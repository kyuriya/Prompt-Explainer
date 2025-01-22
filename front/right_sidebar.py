# import streamlit as st

# def render_right_sidebar():
#     """오른쪽 사이드바 구현"""
#     st.header("Alternative Suggestions")
#     st.write("This section is reserved for future features!")
import streamlit as st
import pandas as pd
from umap import UMAP
from sentence_transformers import SentenceTransformer
from streamlit_plotly_events import plotly_events
from plotly.graph_objs import Scatter, Figure
# from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.colors import qualitative
# from gensim.models import Word2Vec


# 텍스트 임베딩 함수 (캐싱 적용)
@st.cache_data
def get_embeddings(words):
    # model = SentenceTransformer('jhgan/ko-sroberta-multitask') #한국어 임베딩 모델
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2') #영어 임베딩 모델
    return model.encode(words, show_progress_bar=True)

# 텍스트 임베딩 함수 (Word2Vec 사용, 캐싱 적용)
# @st.cache_data
# def get_embeddings(words):
#     # Word2Vec 모델 초기화 및 학습
#     model = Word2Vec(sentences=[words], vector_size=50, window=3, min_count=1, sg=1, epochs=50)
#     embeddings = [model.wv[word] for word in words]
#     return embeddings

# UMAP 차원 축소 함수 (캐싱 적용)
@st.cache_data
def apply_umap(embeddings):
    umap_model = UMAP(n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42)
    return umap_model.fit_transform(embeddings)


def render_right_sidebar():
    # 카테고리별 단어 리스트
    categories = {
        "DFS": [
            "Depth-First Search", "Tree Traversal", "Node Tracking", "Recursive Call", "Stack Data Structure",
            "Graph Traversal", "Backtracking", "Child Node", "Root Node", "Stack-Based Search",
            "Node Visit", "Graph Search", "Recursive Call", "Tree Structure", "Parent Node",
            "Stack Utilization", "Deep Search", "Traversal Path", "Recursive Search", "Stack Trace"
        ],
        "BFS": [
            "Breadth-First Search", "Queue Data Structure", "Level Search", "Shortest Path Search", "Graph Exploration", 
            "Connection Search", "Visited Records", "Adjacent Node Search", "Queue Implementation", "Search Order", 
            "Queue-Based Search", "Graph Traversal", "Level-Wise Visit", "Adjacency List", "Graph Queue", 
            "Queue Search", "Breadth-First Search Implementation", "Queue Node", "Search Node", "Connection Structure"
        ],
        "Sort":[
            "Sorting Algorithm", "Bubble Sort", "Selection Sort", "Insertion Sort", "Merge Sort", 
            "Quick Sort", "Heap Sort", "Radix Sort", "Array Sorting", "List Sorting", 
            "Sort Comparison", "Sort Structure", "Sort Order", "Data Sorting", "Sorting Speed", 
            "Sorting Method", "Sorting Conditions", "Sorting Efficiency", "Sort Implementation", "Sorting Time Complexity"
            
        ],
        "Greedy":[
             "Greedy Selection", "Optimal Solution Construction", "Optimization Algorithm", "Greedy Strategy", "Greedy Search", 
            "Greedy Solution Method", "Maximum Value Selection", "Cost Minimization", "Greedy Analysis", "Greedy Pattern", 
            "Greedy Structure", "Greedy Step", "Greedy Operation", "Greedy Decision", "Greedy Optimization", 
            "Greedy Implementation", "Greedy Approach", "Greedy Problem", "Greedy Efficiency", "Greedy Construction Method"
        ],
        "DP":[
             "Dynamic Programming", "Subproblems", "Optimal Substructure", "Memoization", "Cache Utilization", 
            "Recurrence Relation", "Optimal Structure", "DP Algorithm", "DP Implementation", "Top-Down", 
            "Bottom-Up", "Eliminating Redundant Calculations", "Optimal Solution Structure", "DP Pattern", "Partial Optimization", 
            "DP Efficiency", "DP Application", "DP Time Complexity", "DP Structure", "Optimization Partitioning"
        ],
        "Shortest Distance":[
            "Shortest Distance", "Dijkstra", "Bellman-Ford", "Floyd-Warshall", "Path Search", 
            "Graph Weights", "Minimum Path", "Shortest Path Search", "Distance Calculation", "Path Efficiency", 
            "Shortest Path Implementation", "Graph Structure", "Minimum Cost", "Path Operation", "Shortest Path Analysis", 
            "Optimal Path Search", "Graph Path", "Shortest Path Pattern", "Shortest Path Time", "Shortest Path Design"
        ]
    }

    # 모든 단어와 카테고리 데이터프레임 생성
    words = [word for category in categories.values() for word in category]
    category_labels = [cat for cat, words_list in categories.items() for _ in words_list]
    df = pd.DataFrame({'Word': words, 'Category': category_labels})

    # 임베딩 및 차원 축소
    embeddings = get_embeddings(df['Word'].tolist())
    embedding = apply_umap(embeddings)

    # 결과를 데이터프레임에 저장
    df['UMAP_1'] = embedding[:, 0]
    df['UMAP_2'] = embedding[:, 1]

    # Streamlit UI
    st.subheader("🪄 Word suggestions by algorithm")
    selected_categories = st.multiselect(
        "Choose the algorithm you're interested in!",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )

    # 선택한 카테고리 필터링
    filtered_df = df[df['Category'].isin(selected_categories)]

    # 기본 색상 팔레트를 이용한 카테고리별 색상 매핑
    # category_colors = {cat: color for cat, color in zip(categories, DEFAULT_PLOTLY_COLORS)}
    # 카테고리별로 색상을 매핑 (Set1 팔레트 사용)
    category_colors = {cat: color for cat, color in zip(categories, qualitative.D3)}

    # Plotly 시각화를 위한 데이터 생성
    fig = Figure()
    # category_colors = {
    #     "DFS": "aquamarine", "BFS": "tomato", "Sort": "lightgreen", 
    #     "Greedy": "plum", "DP": "lightsalmon", "Shortest Path": "chocolate"
    # }

    for category in filtered_df['Category'].unique():
        category_df = filtered_df[filtered_df['Category'] == category]
        fig.add_trace(Scatter(
            x=category_df['UMAP_1'],
            y=category_df['UMAP_2'],
            mode='markers',
            # marker=dict(size=8, color=category_colors[category]),
            marker=dict(color=category_colors[category], size=7),
            # opacity=0.5,
            name=category,
            text=category_df['Word'],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    # 그래프 레이아웃 설정
    fig.update_layout(
        title=None,  # 제목 제거
        xaxis_title=None,
        yaxis_title=None,
        height=450,
        width=480,
        clickmode='event+select',
        legend=dict(
            orientation="h",  # 수평으로 정렬
            yanchor="bottom",
            y=-0.3,  # 그래프 아래로 이동
            xanchor="center",
            x=0.5
        )
    )
    # 그래프 축 숨기기
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # Streamlit CSS 스타일로 왼쪽 정렬
    st.markdown(
        """
        <style>
        .plot-container {
            display: flex;
            justify-content: flex-start; /* 왼쪽 정렬 */
            margin-left: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # 그래프 표시
    # st.plotly_chart(fig, use_container_width=True)
    clicked_points = plotly_events(fig, click_event=True, hover_event=False)

    