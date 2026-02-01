import streamlit as st
import sys

from meld_based import search, get_dataset_info, parse_srt_content, search_dynamic, build_contextual_clips
import tempfile
import os

st.set_page_config(
    page_title="🎬 dAIrectors - Semantic Footage Search",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .header-container {
        text-align: center;
        padding: 2rem 0;
        color: white;
    }
    
    .header-container h1 {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .search-container {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .result-card {
        background: white;
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .result-rank {
        background: #667eea;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .result-movie {
        font-weight: bold;
        font-size: 1.1rem;
        color: #333;
    }
    
    .result-time {
        color: #666;
        font-size: 0.9rem;
        font-family: monospace;
    }
    
    .confidence-badge {
        background: #eef2ff;
        color: #4f46e5;
        padding: 0.4rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.85rem;
        border: 1px solid #c7d2fe;
        display: inline-block;
    }
    
    .result-text {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-style: italic;
        color: #333;
        border-left: 4px solid #667eea;
        margin-top: 1rem;
        margin-left: 3.5rem; /* Indent to align with content, not rank circle */
    }
    
    .sidebar-info {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-info h4 {
        color: #aebbf2;
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
        font-weight: bold;
    }

    .sidebar-info p {
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .no-results {
        text-align: center;
        padding: 3rem;
        background: #fff3cd;
        border-radius: 0.5rem;
        color: #856404;
    }
    
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.7);
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

MOVIE_MAP = {
    "Adolescence S01E03.srt": "https://www.netflix.com/watch/81763206?trackId=284616272&tctx=0%2C0%2C5189220c-1a36-4bb4-bad9-68d0241a026d%2C5189220c-1a36-4bb4-bad9-68d0241a026d%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vYWRvbGUvMC8wIiwibG9jYWxTZWN0aW9uSWQiOiIyIn0%3D%2C%2C%2C%2C%2C81756069%2CVideo%3A81763206%2CdetailsPageEpisodePlayButton",
    "Article 15.srt": "https://www.netflix.com/watch/81154455?trackId=284616272&tctx=0%2C0%2Ce039913c-2632-4b96-8bb1-c661b384ca3d%2Ce039913c-2632-4b96-8bb1-c661b384ca3d%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vYXJ0aWNsZSAxNS8wLzAiLCJsb2NhbFNlY3Rpb25JZCI6IjIifQ%3D%3D%2C%2C%2C%2CtitlesResults%2C81154455%2CVideo%3A81154455%2CminiDpPlayButton",
    "Breaking Bad S05E14.srt": "https://www.netflix.com/watch/70236426?trackId=284616272&tctx=0%2C0%2Ccadf9c5a-e25b-424b-9251-8707d8e50d5e%2Ccadf9c5a-e25b-424b-9251-8707d8e50d5e%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vYnJlYWtpbmcgYmFkLzAvMCIsImxvY2FsU2VjdGlvbklkIjoiMiJ9%2C%2C%2C%2CtitlesResults%2C70143836%2CVideo%3A70236426%2CdetailsPageEpisodePlayButton",
    "Gangubai.srt": "https://www.netflix.com/watch/81280352?trackId=14277281&tctx=-97%2C-97%2C%2C%2C%2C%2C%2C%2C%2CVideo%3A81280352%2CdetailsPagePlayButton",
    "Hi.Nanna.srt": "https://www.netflix.com/watch/81682028?trackId=284616272&tctx=0%2C0%2C30bb384c-bdec-4ae3-abe4-97c6850a5e36%2C30bb384c-bdec-4ae3-abe4-97c6850a5e36%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vaGkgbmFuLzAvMCIsImxvY2FsU2VjdGlvbklkIjoiMiJ9%2C%2C%2C%2CtitlesResults%2C81682028%2CVideo%3A81682028%2CminiDpPlayButton",
    "Im.Thinking.of.Ending.Things.srt": "https://www.netflix.com/watch/80211559?trackId=284616272&tctx=0%2C0%2Ccee44524-3a71-47a4-9d83-9eeb54c11088%2Ccee44524-3a71-47a4-9d83-9eeb54c11088%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vaW0gdGhpbmtpbi8wLzAiLCJsb2NhbFNlY3Rpb25JZCI6IjIifQ%3D%3D%2C%2C%2C%2CtitlesResults%2C80211559%2CVideo%3A80211559%2CminiDpPlayButton",
    "Jurassic.World.srt": "https://www.netflix.com/watch/80029196?trackId=284616272&tctx=0%2C1%2C6dd8babc-0151-466e-b8f5-c0ca56dd1bc6%2C6dd8babc-0151-466e-b8f5-c0ca56dd1bc6%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vanVyYXNzaWMgd29ybGQgMjAxNS8wLzAiLCJsb2NhbFNlY3Rpb25JZCI6IjIifQ%3D%3D%2C%2C%2C%2CtitlesResults%2C80029196%2CVideo%3A80029196%2CminiDpPlayButton",
    "Kumbalangi Nights.srt": "https://www.primevideo.com/detail/0HBV5G7X1PJ16OYMD66SJO0AVC/ref=atv_sr_fle_c_sr782405_pvsearchresults_1_1?autoplay=1",
    "Marriage.Story.srt": "https://www.netflix.com/watch/80223779?trackId=284616272&tctx=0%2C0%2C0617bcc2-b307-476f-8ebe-08e094af1bdc%2C0617bcc2-b307-476f-8ebe-08e094af1bdc%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vbWFycmlhZ2UvMC8wIiwibG9jYWxTZWN0aW9uSWQiOiIyIn0%3D%2C%2C%2C%2CtitlesResults%2C80223779%2CVideo%3A80223779%2CminiDpPlayButton",
    "Rocky Aur Rani Ki Prem Kahaani.srt": "https://www.primevideo.com/detail/0I6U0N56BVTVGY24EM2FARBNIC/ref=atv_sr_fle_c_sr9fee6d_pvsearchresults_1_1?autoplay=1",  # Example ID provided
    "Shutter.Island.srt": "https://www.netflix.com/watch/70095139?trackId=284616272&tctx=0%2C0%2Cd4aa62b6-e545-4f6b-99a2-cfe023c78577%2Cd4aa62b6-e545-4f6b-99a2-cfe023c78577%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vc2h1dHQvMC8wIiwibG9jYWxTZWN0aW9uSWQiOiIyIn0%3D%2C%2C%2C%2CtitlesResults%2C70095139%2CVideo%3A70095139%2CminiDpPlayButton",
    "Shyam.Singha.Roy.srt": "https://www.netflix.com/watch/81486768?trackId=284616272&tctx=0%2C0%2C59a96b77-8e57-4036-a046-a19d9ce3a8de%2C59a96b77-8e57-4036-a046-a19d9ce3a8de%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vc2h5YS8wLzAiLCJsb2NhbFNlY3Rpb25JZCI6IjIifQ%3D%3D%2C%2C%2C%2CtitlesResults%2C81486768%2CVideo%3A81486768%2CminiDpPlayButton",
    "The.Girl.on.the.Train.srt": "https://www.netflix.com/watch/81144153?trackId=284616272&tctx=0%2C1%2Cac53b11f-233c-4b9e-8746-6e957ca7fb09%2Cac53b11f-233c-4b9e-8746-6e957ca7fb09%7C%3DeyJwYWdlSWQiOiI5NjdlYWY1Yy1mMGZkLTQ5NDctYjdiNy1iMWJjODAzMzk5MWIvMS8vdGhlIGdpcmwvMC8wIiwibG9jYWxTZWN0aW9uSWQiOiIyIn0%3D%2C%2C%2C%2CtitlesResults%2C81144153%2CVideo%3A81144153%2CminiDpPlayButton"
}   

def main():
    st.markdown("""
        <div class="header-container">
            <h1>🎬 dAIrectors</h1>
            <p>Semantic Footage Search Engine</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Search Settings")
        
        top_k_results = st.slider(
            "Results to return",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Number of results to display"
        )
        
        min_confidence = st.slider(
            "Minimum confidence (%)",
            min_value=10,
            max_value=80,
            value=30,
            step=5,
            help="Filter results by minimum confidence score"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Dataset Information")
        
        try:
            info = get_dataset_info()
            
            st.markdown(f"""
                <div class="sidebar-info">
                    <h4>📁 Total Clips</h4>
                    <p>{info['total_clips']:,}</p>
                    <h4>🎬 Movies</h4>
                    <p>{info['unique_movies']}</p>
                    <h4>🧠 Embedding Model</h4>
                    <p>{info['model_name']}</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading dataset info: {e}")
    
    tab1, tab2 = st.tabs(["🔍 Database Search", "📤 Upload & Search"])
    
    with tab1:
        st.markdown("### Search Existing Movie Database")
        with st.form(key='search_form'):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                query = st.text_input(
                    "🔎 Enter your search query",
                    placeholder="e.g., 'hesitant reply', 'dominant reply', 'grieving'",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button("🔍 Search", use_container_width=True, type="primary")
        
        if submit_button and query:
            if len(query.strip()) < 3:
                st.warning("⚠️ Please enter a query with at least 3 characters.")
            else:
                with st.spinner("🔍 Searching through subtitles..."):
                    results = search(query, top_k=top_k_results, min_confidence=min_confidence)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                if results:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Top Match</h3>
                                <div class="value">{results[0]['confidence']}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        avg_confidence = sum(r['confidence'] for r in results) / len(results)
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Avg Confidence</h3>
                                <div class="value">{avg_confidence:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Results Found</h3>
                                <div class="value">{len(results)}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    
                    st.markdown("### 🎞️ Results (Ranked by Confidence)")
                    
                    for idx, result in enumerate(results, 1):
                        movie_filename = result['movie']
                        base_url = MOVIE_MAP.get(movie_filename)
                        
                        link_html = ""
                        if base_url and "########" not in base_url:
                            separator = "&" if "?" in base_url else "?"
                            timestamp_url = f"{base_url}{separator}t={result['start_sec']}"
                            
                            btn_color = "#4a5568" # Neutral Slate Grey
                            btn_text = "Watch Clip"
                            
                            link_html = f"""<a href="{timestamp_url}" target="_blank" style="display: inline-block; background-color: {btn_color}; color: white; padding: 0.4rem 0.8rem; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 0.85rem; transition: opacity 0.2s;">🍿 {btn_text}</a>"""
                        st.markdown(f"""<div class="result-card" style="display: flex; flex-direction: column;"><div class="result-header" style="display: flex; justify-content: space-between; align-items: flex-start;"><div style="display: flex; gap: 1rem; align-items: flex-start;"><div class="result-rank">#{idx}</div><div><div class="result-movie">🎬 {result['movie']}</div><div class="result-time">⏱️ {result['start_time']} → {result['end_time']}</div></div></div><div style="display: flex; gap: 10px; align-items: center; justify-content: flex-end;">{link_html}<div class="confidence-badge">{result['confidence']}% Match</div></div></div><div class="result-text">💬 {result['text']}</div></div>""", unsafe_allow_html=True)
                
                else:
                    st.markdown(f"""
                        <div class="no-results">
                            <h3>❌ No Results Found</h3>
                            <p>Try lowering the minimum confidence threshold or using different keywords.</p>
                            <p>Current threshold: <strong>{min_confidence}%</strong></p>
                        </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📤 Upload Video & Transcript")
        
        st.info("ℹ️ Upload a video file and its corresponding .srt transcript to search within specific timestamps.")
        
        col_up1, col_up2 = st.columns(2)
        
        with col_up1:
            uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi', 'mkv'])
            
        with col_up2:
            uploaded_srt = st.file_uploader("Upload Transcript (.srt)", type=['srt', 'txt'])
            
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None
        if 'video_start_time' not in st.session_state:
            st.session_state.video_start_time = 0

        with st.form(key='upload_search_form'):
            upload_query = st.text_input("🔎 Video Search Query", placeholder="What are you looking for in this video?")
            submit_button = st.form_submit_button("🚀 Analyze & Search", type="primary")

        if submit_button:
            if uploaded_srt and upload_query:
                with st.spinner("Processing..."):
                    try:
                        stringio = uploaded_srt.getvalue().decode("utf-8")
                        clips_raw = parse_srt_content(stringio, movie_name=uploaded_srt.name)
                        clips = build_contextual_clips(clips_raw, window=1)
                        
                        st.info(f"Parsed {len(clips)} clips from transcript.")
                        
                        if len(clips) > 0:
                            results = search_dynamic(upload_query, clips, top_k=top_k_results, min_confidence=min_confidence)
                            st.session_state.search_results = results
                            if results:
                                st.session_state.video_start_time = results[0]['start_sec']
                            
                        else:
                            st.error("Could not parse any clips from the SRT file. Please check formatting.")
                            st.session_state.search_results = None
                            
                    except Exception as e:
                        st.error(f"Error parsing SRT or searching: {e}")
                        st.session_state.search_results = None
            else:
                st.warning("⚠️ Please upload a transcript and enter a query.")
        
        if st.session_state.search_results:
            results = st.session_state.search_results
            
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            st.markdown(f"### 🎬 Video Player")
            st.video(uploaded_video, start_time=st.session_state.video_start_time, autoplay=True)
            
            st.markdown("---")
            st.markdown(f"### Found Matches ({len(results)})")
            
            for idx, result in enumerate(results, 1):
                btn_key = f"play_{idx}_{result['start_sec']}"
                
                with st.container():
                    col_res_1, col_res_2 = st.columns([4, 1])
                    
                    with col_res_1:
                        st.markdown(f"""
                            <div class="result-card" style="margin-bottom: 1.5rem;">
                                <div class="result-header">
                                    <div style="display: flex; gap: 1rem; align-items: center;">
                                        <div class="result-rank">#{idx}</div>
                                        <div>
                                            <div class="result-movie">Timestamp: {result['start_time']}</div>
                                        </div>
                                    </div>
                                    <div class="confidence-badge">{result['confidence']}% Match</div>
                                </div>
                                <div class="result-text">
                                    💬 {result['text']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col_res_2:
                        st.markdown(f"**Start: {result['start_time']}**")
                        if st.button(f"▶️ Play {result['start_sec']}s", key=btn_key, type="primary"):
                            st.session_state.video_start_time = result['start_sec']
                            st.rerun()
                
                st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        elif st.session_state.search_results is not None and len(st.session_state.search_results) == 0:
             st.warning("No matches found in this transcript for your query.")
    
    st.markdown("""
        <div class="footer">
            <p>🎞️ dAIrectors | Semantic Footage Search Engine</p>
            <p>Powered by Sentence Transformers & FAISS Vector Search</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()