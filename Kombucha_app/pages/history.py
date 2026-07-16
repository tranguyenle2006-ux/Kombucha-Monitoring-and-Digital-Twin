import streamlit as st
import pandas as pd
import plotly.express as px


def show():
    st.title("📜 Fermentation History Log")
    
    if 'history_rows' not in st.session_state or not st.session_state.history_rows:
        st.info("No history recorded yet. Use 'Log Current Reading' on F1/F2 pages.")
        return
    
    df = pd.DataFrame(st.session_state.history_rows)
    
    # Filters
    st.sidebar.header("Filters")
    process_filter = st.sidebar.multiselect("Process", options=df['process'].unique(),
                                            default=df['process'].unique())
    batch_filter = st.sidebar.multiselect("Batch", options=df['batch'].unique(),
                                          default=df['batch'].unique()) if 'batch' in df.columns else None
    source_filter = st.sidebar.multiselect("Source", options=df['source'].unique(),
                                           default=df['source'].unique()) if 'source' in df.columns else None
    
    filtered = df[df['process'].isin(process_filter)]
    if batch_filter is not None:
        filtered = filtered[filtered['batch'].isin(batch_filter)]
    if source_filter is not None:
        filtered = filtered[filtered['source'].isin(source_filter)]
    
    st.dataframe(filtered, use_container_width=True)
    
    # Aggregated plots
    st.subheader("Overview Plots")
    if not filtered.empty:
        if 'pH' in filtered.columns:
            f1_data = filtered[filtered['process'] == 'F1']
            if not f1_data.empty:
                fig = px.line(f1_data, x='day', y='pH', color='batch', title='F1 pH History')
                st.plotly_chart(fig, use_container_width=True)
        if 'pressure' in filtered.columns:
            f2_data = filtered[filtered['process'] == 'F2']
            if not f2_data.empty:
                fig = px.line(f2_data, x='day', y='pressure', color='batch', title='F2 Pressure History')
                st.plotly_chart(fig, use_container_width=True)
    
    # Export
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data as CSV", csv, "history.csv", "text/csv")
    
    if st.button("Clear All History"):
        st.session_state.history_rows = []
        st.rerun()