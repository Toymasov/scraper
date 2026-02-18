import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import os
import datetime as dt_mod
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Kun.uz Content Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #1e293b;
        color: white;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
FILES = {
    "Kun.uz": "kun_uz_news.csv",
    "Daryo.uz": "daryo_news.csv",
    "Vaqt.uz": "vaqt_news.csv",
    "Gazeta.uz": "gazeta_news.csv"
}
MERGED_FILE = "merged_news.csv"

def parse_standard_date(date_str):
    if pd.isna(date_str) or date_str == 'N/A':
        return None
    try:
        date_str = str(date_str).strip()
        # Match DD.MM.YYYY
        match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', date_str)
        if match:
            return pd.to_datetime(match.group(), format='%d.%m.%Y')
        
        # Gazeta.uz format: '2026-yil, 25-yanvar, 21:08'
        match_uz = re.search(r'(\d{4})-yil,\s+(\d{1,2})-([a-z‘]+)', date_str, re.I)
        if match_uz:
            year = int(match_uz.group(1))
            day = int(match_uz.group(2))
            month_str = match_uz.group(3).lower()
            months = {
                'yanvar': 1, 'fevral': 2, 'mart': 3, 'aprel': 4, 'may': 5, 'iyun': 6,
                'iyul': 7, 'avgust': 8, 'sentabr': 9, 'oktabr': 10, 'noyabr': 11, 'dekabr': 12
            }
            month = months.get(month_str, 1)
            return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")

        # Gazeta.uz older format: '27.01, 15:51'
        if ',' in date_str and len(date_str.split(',')[0].split('.')) == 2:
            current_year = datetime.now().year
            return pd.to_datetime(f"{date_str.split(',')[0]}.{current_year}", format='%d.%m.%Y')

        if "Bugun" in date_str:
            return pd.to_datetime(datetime.now().date())
        if "Kecha" in date_str:
            return pd.to_datetime((datetime.now() - timedelta(days=1)).date())
        
        # Check for just DD.MM.YYYY anywhere in string
        match_any = re.search(r'\d{2}\.\d{2}\.\d{4}', date_str)
        if match_any:
                return pd.to_datetime(match_any.group(), format='%d.%m.%Y')

        return None
    except:
        return None

def update_database():
    """Reads all source CSVs, merges them, removes duplicates, and saves to MERGED_FILE."""
    dfs = []
    for source, file_path in FILES.items():
        if os.path.exists(file_path):
            try:
                # Read individual file
                df = pd.read_csv(file_path, encoding='utf-8-sig', on_bad_lines='skip')
                
                # Normalize columns if needed
                if 'Source' not in df.columns:
                    df['Source'] = source
                else:
                    df['Source'] = source # Enforce correct source name
                
                # Basic column validation
                required_cols = ['URL', 'Title', 'Content', 'Date', 'Category']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None # Fill missing cols with None
                
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                pass
    
    if dfs:
        # Concatenate all
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates based on URL
        if 'URL' in full_df.columns:
            full_df = full_df.drop_duplicates(subset=['URL'], keep='last')
        
        # Save to merged file
        full_df.to_csv(MERGED_FILE, index=False, encoding='utf-8-sig')
        return True, len(full_df)
    
    return False, 0

@st.cache_data
def load_data():
    # If merged file doesn't exist, create it
    if not os.path.exists(MERGED_FILE):
        update_database()
    
    if os.path.exists(MERGED_FILE):
        try:
            df = pd.read_csv(MERGED_FILE, encoding='utf-8-sig', on_bad_lines='skip')
            
            # --- Preprocessing ---
            # Date parsing
            if 'Date' in df.columns:
                df['parsed_date'] = df['Date'].apply(parse_standard_date)
                df['parsed_date'] = pd.to_datetime(df['parsed_date'], errors='coerce')
                df['parsed_date'] = df['parsed_date'].ffill().bfill().fillna(pd.Timestamp.now())
            else:
                df['parsed_date'] = pd.Timestamp.now()
            
            # Fill string NaNs
            str_cols = ['Content', 'Title', 'Category', 'URL']
            for col in str_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
                else:
                    df[col] = ''

            return df
        except Exception as e:
            st.error(f"Ma'lumotlar bazasini o'qishda xatolik: {e}")
            return None
    else:
        return None

# Sidebar
st.sidebar.header("⚙️ Sozlamalar")

# Update button
if st.sidebar.button("🔄 Bazani yangilash"):
    with st.sidebar.status("Ma'lumotlar yangilanmoqda...", expanded=True) as status:
        success, count = update_database()
        if success:
            status.update(label=f"Yangilandi! Jami: {count}", state="complete", expanded=False)
            st.cache_data.clear()
            st.rerun()
        else:
            status.update(label="Yangilashda xatolik", state="error", expanded=False)

source_option = st.sidebar.selectbox("Ma'lumotlar manbasi", ["Barchasi", "Kun.uz", "Daryo.uz", "Vaqt.uz", "Gazeta.uz"])

# Load all data
df_all = load_data()
df = None

if df_all is not None:
    # Filter by source
    if source_option != "Barchasi":
        df = df_all[df_all['Source'] == source_option].copy()
    else:
        df = df_all.copy()

if df is not None:
    st.title(f"📊 {source_option} Content Analyser")
    st.markdown("---")

    # Filters
    st.sidebar.header("🔍 Filtrlar")
    categories = ['Barchasi'] + sorted(df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Kategoriyani tanlang", categories)

    min_date = df['parsed_date'].min()
    max_date = df['parsed_date'].max()
    
    # Handle NaT/NaN safely
    if pd.isna(min_date) or pd.isna(max_date):
        min_date = pd.Timestamp.now()
        max_date = pd.Timestamp.now()

    try:
        date_range = st.sidebar.date_input(
            "Sana oralig'i",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
    except Exception as e:
        st.error(f"Sana filtrini yaratishda xatolik: {e}")
        date_range = (min_date.date(), max_date.date())

    # Filter data
    filtered_df = df.copy()
    if selected_category != 'Barchasi':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['parsed_date'].dt.date >= start_date) & 
            (filtered_df['parsed_date'].dt.date <= end_date)
        ]

    # Main Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jami yangiliklar", f"{len(filtered_df)}")
    with col2:
        st.metric("Kategoriyalar", f"{filtered_df['Category'].nunique()}")
    with col3:
        days_diff = (filtered_df['parsed_date'].max() - filtered_df['parsed_date'].min()).days
        avg_per_day = round(len(filtered_df) / max(1, days_diff), 1)
        st.metric("Kunlik o'rtacha", f"{avg_per_day}")
    with col4:
        recent_date = filtered_df['parsed_date'].max().strftime('%d.%m.%Y')
        st.metric("Oxirgi yangilanish", recent_date)

    st.markdown("### 📈 Statistika")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Sana bo'yicha", "📂 Kategoriya bo'yicha", "🔍 So'z qidiruvi", "☁️ Vizualizatsiya"])

    with tab1:
        st.subheader("Vaqt o'tishi bilan yangiliklar soni")
        daily_counts = filtered_df.groupby([filtered_df['parsed_date'].dt.date, 'Source']).size().reset_index(name='count')
        fig_time = px.line(daily_counts, x='parsed_date', y='count', color='Source',
                          labels={'parsed_date': 'Sana', 'count': 'Yangiliklar soni'},
                          line_shape='spline', render_mode='svg')
        fig_time.update_traces(line_width=3)
        st.plotly_chart(fig_time, width="stretch")

    with tab2:
        st.subheader("Kategoriyalar bo'yicha taqsimot")
        cat_counts = filtered_df.groupby(['Category', 'Source']).size().reset_index(name='count')
        
        fig_cat = px.bar(cat_counts, x='Category', y='count', color='Source',
                        labels={'Category': 'Kategoriya', 'count': 'Soni'},
                        barmode='group')
        st.plotly_chart(fig_cat, width="stretch")

        # Source share for "Barchasi"
        if source_option == "Barchasi":
            source_counts = filtered_df['Source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'count']
            fig_source = px.pie(source_counts, names='Source', values='count', hole=0.4, title="Manbalar ulushi")
            st.plotly_chart(fig_source, width="stretch")

    with tab3:
        st.subheader("Kontent ichidan so'z qidirish")
        search_query = st.text_input("Qidirish uchun so'zni kiriting (masalan: iqtisodiyot, sport, prezident)", "")
        
        if search_query:
            # Case insensitive search in Title and Content
            mask = filtered_df.apply(lambda row: 
                                   search_query.lower() in row['Title'].lower() or 
                                   search_query.lower() in row['Content'].lower(), axis=1)
            search_results = filtered_df[mask]
            
            found_count = len(search_results)
            total_count = len(filtered_df)
            percentage = (found_count / total_count * 100) if total_count > 0 else 0
            
            c1, c2 = st.columns(2)
            c1.metric("Topilgan kontentlar", f"{found_count}")
            c2.metric("Ulushi", f"{percentage:.2f}%")
            
            if found_count > 0:
                st.markdown(f"#### '{search_query}' so'zi qatnashgan yangiliklar:")
                for idx, row in search_results.head(15).iterrows():
                    source_label = f"[{row['Source']}] " if source_option == "Barchasi" else ""
                    with st.expander(f"{source_label}{row['Title']} ({row['Date']})"):
                        st.write(f"**URL:** {row['URL']}")
                        st.write(f"**Kategoriya:** {row['Category']}")
                        st.write(row['Content'][:700] + "...")
                
                if found_count > 15:
                    st.info(f"Yana {found_count - 15} ta natija mavjud.")
            else:
                st.warning("Ushbu so'z hech qaysi maqolada topilmadi.")

    with tab4:
        st.subheader("☁️ Word Cloud (So'zlar buluti)")
        st.info("Bu bo'limda matnlardagi eng ko'p uchraydigan so'zlarni ko'rishingiz mumkin.")

        # Updated Stopwords definition
        stopwords_uz = set([
            'va', 'bu', 'bilan', 'uchun', 'ham', 'deb', 'o‘z', 'bir', 'bor', 'edi', 'esa', 'kabi', 
            'shu', 'u', 'ushbu', 'lekin', 'ammo', 'chunki', 'bo‘lib', 'bo‘lgan', 'yili', 'so‘ng', 
            'yil', 'faqat', 'qildi', 'qilish', 'mumkin', 'kerak', 'qilingan', 'bo‘ladi', 'etildi', 
            'etadi', 'ekan', 'avval', 'keyin', 'ular', 'biz', 'siz', 'men', 'u', 'bu', 'o‘sha',
            'haqida', 'tomonidan', 'bo‘yicha', 'hamda', 'uni', 'uning', 'unga', 'unda', 'undan',
            'eng', 'juda', 'yana', 'endi', 'mana', 'barcha', 'boshqa', 'har', 'hamma', 'edi', 'edi',
            'yoki', 'jumladan', 'bo‘lsa', 'ko‘ra', 'ta', 'ya\'ni', 'yani', 'balki', 'holatda', 'sababli',
            'tufayli', 'biri', 'ko‘p', 'yilgi', 'yangi', 'kun', 'oy', 'bo‘yi', 'o‘zi', 
            'o‘zining', 'o‘ziga', 'o‘zini', 'o‘zidan', 'haq', 'hali', 'endi', 'o', 'g', 'n', 's', 'm', 'd', 'k', 'z',
            'esa', 'emas', 'aholda', 'aynan', 'demak', 'gohi', 'gohida', 'kerakli', 'keraksiz', 'aniq', 'agar', 'shuningdek','biroq','buni','haqda','orqali','degan', 'nisbatan','qilib','dedi'
        ])
        
        # Keyword filter for Word Cloud
        wc_query = st.text_input("Mavzu bo'yicha filtrlash (ixtiyoriy, bir nechta so'zni vergul bilan ajrating):", 
                                help="Agar so'z kiritilsa, faqat shu so'zlar qatnashgan maqolalardan Word Cloud yasaladi.")
        
        col_wc1, col_wc2 = st.columns([1, 3])
        
        with col_wc1:
            generate_wc = st.button("Word Cloud yaratish")
            max_words = st.slider("So'zlar soni", 50, 200, 100)
            
        if generate_wc:
            with st.spinner("Word Cloud yaratilmoqda..."):
                # Use filtered_df to respect current filters
                target_df = filtered_df.copy()

                # Apply specific keyword filter if provided
                if wc_query:
                    keywords = [k.strip().lower() for k in wc_query.split(',') if k.strip()]
                    if keywords:
                        # Filter rows where Title OR Content contains ANY of the keywords
                        mask = target_df.apply(lambda row: any(k in row['Title'].lower() or k in row['Content'].lower() for k in keywords), axis=1)
                        target_df = target_df[mask]
                        st.success(f"Filtrlashdan so'ng {len(target_df)} ta maqola topildi.")
                    else:
                        st.warning("Kiritilgan so'zlar bo'sh, barcha ma'lumotlar ishlatilmoqda.")
                
                text_data = target_df['Content'].dropna().astype(str).tolist()
                all_text = " ".join(text_data)
                
                if not all_text.strip():
                    st.warning("Tahlil qilish uchun yetarli matn mavjud emas.")
                else:
                    # Basic cleaning
                    msg = st.empty()
                    msg.text("Matnlar tozalanmoqda...")
                    
                    # Convert to lower case
                    all_text = all_text.lower()

                    # 1. Remove URLs
                    all_text = re.sub(r'http\S+|www\.\S+', '', all_text)

                    # 2. Keep letters, numbers, underscores and apostrophes used in Uzbek (', ‘, ’)
                    # We want to remove OTHER punctuation. 
                    # Strategy: Replace everything that is NOT a word char or space or apostrophe with space.
                    all_text = re.sub(r"[^\w\s'‘’`]", ' ', all_text)

                    # 3. Collapse multiple spaces
                    all_text = re.sub(r'\s+', ' ', all_text).strip()
                    
                    msg.text("Rasm chizilmoqda...")
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white', 
                        stopwords=stopwords_uz,
                        max_words=max_words,
                        colormap='viridis',
                        # IMPORTANT: Regex for tokenizing words to include apostrophes inside words
                        regexp=r"\w+['‘’`]?\w*"
                    ).generate(all_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)

                    # Download button logic
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="📷 Rasmni yuklab olish (PNG)",
                        data=buf,
                        file_name="wordcloud.png",
                        mime="image/png",
                    )
                    
                    msg.empty()
        
        st.markdown("---")
        st.subheader("🌳 Tree Map (Maxsus so'zlar tahlili)")
        st.info("O'zingiz qiziqtirgan so'zlarni kiriting va ularning matnda qancha uchrashini ko'ring.")
        
        default_keywords = "iqtisodiyot, siyosat, jamiyat, sport, texnologiya, ta'lim, tibbiyot"
        user_keywords = st.text_area("Kalit so'zlar (vergul bilan ajrating):", default_keywords)
        
        if st.button("Tree Map yaratish"):
            keywords_list = [k.strip().lower() for k in user_keywords.split(',') if k.strip()]
            
            if keywords_list:
                with st.spinner("Tahlil qilinmoqda..."):
                    # Process text once (using filtered df)
                    text_data = filtered_df['Content'].dropna().astype(str).tolist()
                    all_text_lower = " ".join(text_data).lower()
                    
                    keyword_counts = []
                    for keyword in keywords_list:
                        # Use regex for whole word match to be more accurate
                        try:
                            match_count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', all_text_lower))
                        except:
                            match_count = all_text_lower.count(keyword) # Fallback
                        
                        if match_count > 0:
                            keyword_counts.append({'So\'z': keyword, 'Soni': match_count})
                    
                    if keyword_counts:
                        df_tree = pd.DataFrame(keyword_counts)
                        fig_tree = px.treemap(
                            df_tree, 
                            path=['So\'z'], 
                            values='Soni',
                            color='Soni',
                            color_continuous_scale='RdBu',
                            title="Kalit so'zlar uchrashish chastotasi"
                        )
                        st.plotly_chart(fig_tree, width="stretch")
                    else:
                        st.warning("Kiritilgan so'zlardan hech biri matnlar tarkibida topilmadi (Tanlangan filtrlar bo'yicha).")
            else:
                st.warning("Iltimos, kamida bitta so'z kiriting.")
        
        st.markdown("---")
        st.subheader("🕸️ Network Graph (Tarmoq grafigi)")
        st.info("So'zlar yoki mavzular o'rtasidagi bog'liqlikni ko'rsatadi.")

        network_keywords_input = st.text_area("So'zlarni kiriting (vergul bilan ajrating):", "iqtisodiyot, bank, dollar, kredit, investitsiya, soliq")
        
        if st.button("Network Graf yaratish"):
            network_keywords = [k.strip().lower() for k in network_keywords_input.split(',') if k.strip()]
            
            if len(network_keywords) < 2:
                st.warning("Bog'liqlikni ko'rish uchun kamida 2 ta so'z kiriting.")
            else:
                with st.spinner("Graf yaratilmoqda..."):
                    co_occurrence = {k: {k2: 0 for k2 in network_keywords} for k in network_keywords}
                    node_counts = {k: 0 for k in network_keywords}

                    text_data = filtered_df['Content'].dropna().astype(str).tolist()
                    
                    for text in text_data:
                        text_lower = text.lower()
                        found_keywords = []
                        for k in network_keywords:
                            if k in text_lower:
                                found_keywords.append(k)
                                node_counts[k] += 1
                        
                        for i in range(len(found_keywords)):
                            for j in range(i + 1, len(found_keywords)):
                                w1, w2 = found_keywords[i], found_keywords[j]
                                co_occurrence[w1][w2] += 1
                                co_occurrence[w2][w1] += 1
                    
                    G = nx.Graph()
                    for k in network_keywords:
                        if node_counts[k] > 0:
                            G.add_node(k, size=np.log1p(node_counts[k])*15 + 10, count=node_counts[k])
                    
                    for k1 in network_keywords:
                        for k2 in network_keywords:
                            if k1 < k2 and co_occurrence[k1][k2] > 0:
                                G.add_edge(k1, k2, weight=co_occurrence[k1][k2])

                    if G.number_of_nodes() == 0:
                        st.warning("Kiritilgan so'zlar bo'yicha hech qanday ma'lumot topilmadi.")
                    else:
                        pos = nx.spring_layout(G, seed=42)
                        
                        fig_net = go.Figure()

                        # Normalize weights for width calculation
                        weights = [G[u][v]['weight'] for u, v in G.edges()]
                        max_weight = max(weights) if weights else 1
                        
                        # Add edges as individual traces to control width and hover text
                        for edge in G.edges(data=True):
                            u, v = edge[0], edge[1]
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            weight = edge[2]['weight']
                            
                            # Jaccard Similarity Calculation
                            # J(A,B) = Intersection(A,B) / Union(A,B)
                            count_a = G.nodes[u]['count']
                            count_b = G.nodes[v]['count']
                            union = count_a + count_b - weight
                            jaccard = weight / union if union > 0 else 0
                            
                            # Scaling width: more weight -> thicker line (min 2, max 10)
                            width = 2 + (weight / max_weight) * 8
                            
                            # Create interpolated points along the line for continuous hover effect
                            dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
                            # Generate enough points so hover snaps effectively everywhere
                            # Using a fixed high number like 20 is usually enough for visual "continuous" feel on web
                            num_points = 20 
                            
                            xs = np.linspace(x0, x1, num_points)
                            ys = np.linspace(y0, y1, num_points)

                            fig_net.add_trace(go.Scatter(
                                x=xs,
                                y=ys,
                                line=dict(width=width, color='#659965'),
                                hoverinfo='text',
                                text=[f"{u} - {v}<br>Birga kelish: {weight} marta<br>Jaccard indeksi: {jaccard:.2f}"] * num_points,
                                mode='lines',
                                showlegend=False,
                                opacity=0.8
                            ))

                        node_x = []
                        node_y = []
                        node_text = []
                        node_size = []
                        node_color = []

                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            count = G.nodes[node]['count']
                            node_text.append(f"{node}: {count} marta")
                            node_size.append(G.nodes[node]['size'])
                            node_color.append(count)

                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            hoverinfo='text',
                            text=node_text,
                            textposition="top center",
                            marker=dict(
                                showscale=True,
                                colorscale='YlGnBu',
                                size=node_size,
                                color=node_color,
                                colorbar=dict(
                                    thickness=15,
                                    title=dict(
                                        text='Uchrashish soni',
                                        side='right'
                                    ),
                                    xanchor='left'
                                ),
                                line_width=2))
                        
                        fig_net.add_trace(node_trace)
                        
                        fig_net.update_layout(
                            title=dict(
                                text='So\'zlar bog\'liqligi grafigi',
                                font=dict(size=16)
                            ),
                            showlegend=False,
                            hovermode='closest',
                            dragmode=False,
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                        
                        st.plotly_chart(fig_net, width="stretch", config={'toImageButtonOptions': {'format': 'png', 'filename': 'network_graph', 'scale': 2}})

        st.markdown("---")
        st.subheader("🌳 Word Tree (So'z daraxti)")
        st.info("Biror so'zni kiriting va undan keyin kelgan so'zlarni shoxlanishini ko'ring. (Sunburst diagramma shaklida)")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            root_word = st.text_input("Asosiy so'zni kiriting (bitta so'z):", "O'zbekiston")
        with c2:
            tree_depth = st.slider("Daraxt chuqurligi", 2, 4, 3)
        with c3:
            tree_type = st.selectbox("Ko'rinish turi", ["Sunburst (Doira)", "Tree (Daraxt)"])
        
        if st.button("Word Tree yaratish"):
            root = root_word.strip().lower()
            if not root:
                st.warning("Iltimos, so'z kiriting.")
            else:
                with st.spinner("Daraxt shakllantirilmoqda..."):
                    sequences = []
                    text_list = filtered_df['Content'].dropna().astype(str).tolist()
                    
                    found = False
                    for text in text_list:
                        # Improved tokenizer to keep Uzbek apostrophes
                        # Matches words that may contain an apostrophe in the middle/end
                        raw_words = re.findall(r"\w+['‘’`]?\w*", text.lower())
                        
                        # Filter out stopwords and numbers
                        words = [
                            w for w in raw_words 
                            if w not in stopwords_uz             # Remove stopwords
                            and not w.isdigit()                  # Remove pure numbers
                            and not re.match(r'^\d+$', w)        # Double check digits
                            and len(w) > 1                       # Remove single chars like 'o', 'g' noise
                        ]

                        try:
                            indices = [i for i, x in enumerate(words) if x == root]
                            for idx in indices:
                                if idx + tree_depth < len(words):
                                    seq = words[idx : idx + tree_depth + 1]
                                    sequences.append(seq)
                                    found = True
                        except:
                            continue
                    
                    if not found:
                        st.warning(f"'{root_word}' so'zi matnda topilmadi yoki uning davomi yo'q.")
                    else:
                        if tree_type == "Sunburst (Doira)":
                            df_paths = pd.DataFrame(sequences, columns=[f'step_{i}' for i in range(tree_depth + 1)])
                            
                            fig_tree_sunburst = px.sunburst(
                                df_paths,
                                path=[f'step_{i}' for i in range(tree_depth + 1)],
                                title=f"'{root_word}' so'zi uchun Word Tree ({len(sequences)} ta holat)",
                                color_discrete_sequence=px.colors.qualitative.Prism,
                                width=800,
                                height=800
                            )
                            
                            st.plotly_chart(fig_tree_sunburst, width="stretch", config={'toImageButtonOptions': {'format': 'png', 'filename': 'word_tree', 'scale': 2}})
                        else:
                            # Graphviz Tree Visualization
                            import graphviz
                            import math
                            
                            dot = graphviz.Digraph(comment='Word Tree')
                            dot.attr(rankdir='LR') 
                            dot.attr('node', shape='ellipse', style='filled', color='lightblue', fontname='Arial')
                            
                            class TrieNode:
                                def __init__(self, name):
                                    self.name = name
                                    self.children = {}
                                    self.count = 0

                            root_node = TrieNode(root)
                            
                            for seq in sequences:
                                current = root_node
                                current.count += 1
                                for word in seq[1:]:
                                    if word not in current.children:
                                        current.children[word] = TrieNode(word)
                                    current = current.children[word]
                                    current.count += 1
                            
                            def add_nodes_edges(node, parent_id=None, graph=None):
                                node_id = str(id(node))
                                label = f"{node.name}\n({node.count})"
                                
                                fontsize = str(max(10, min(24, 10 + int(math.log(node.count+1)*3))))
                                fillcolor = "#e6f3ff" if node.count < 5 else ("#b3d9ff" if node.count < 20 else "#80bfff")
                                
                                graph.node(node_id, label, fontsize=fontsize, fillcolor=fillcolor)
                                
                                if parent_id:
                                    penwidth = str(max(1, min(4, 1 + int(math.log(node.count+1)))))
                                    graph.edge(parent_id, node_id, penwidth=penwidth, color="#666666")
                                
                                sorted_children = sorted(node.children.values(), key=lambda x: x.count, reverse=True)
                                for child in sorted_children[:15]: 
                                    add_nodes_edges(child, node_id, graph)

                            add_nodes_edges(root_node, None, dot)
                            
                            st.graphviz_chart(dot)

    # Data Table
    st.markdown("---")
    st.subheader("📋 Ma'lumotlar jadvali")
    cols_to_show = ['Source', 'Date', 'Category', 'Title', 'URL'] if source_option == "Barchasi" else ['Date', 'Category', 'Title', 'URL']
    st.dataframe(filtered_df[cols_to_show], width="stretch")

else:
    st.error(f"Ma'lumotlar topilmadi. Iltimos, manba uchun CSV fayli mavjudligini tekshiring.")
    if source_option == "Daryo.uz":
        st.info("Eslatma: Daryo.uz scraperi fon rejimida ishlayotgan bo'lishi mumkin. Fayl yaratilishini kuting.")
