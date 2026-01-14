import streamlit as st
from databricks import sql
import pandas as pd

# 1. Page Config
st.set_page_config(page_title="Game Night Selector", layout="centered", page_icon="🎲")

# 2. Connection Function
# We use st.cache_data so we don't hit Databricks on every button click
@st.cache_data(ttl=3600)
def get_data():
    conn = sql.connect(
        server_hostname=st.secrets["DATABRICKS_HOST"],
        http_path=st.secrets["DATABRICKS_HTTP_PATH"],
        access_token=st.secrets["DATABRICKS_TOKEN"]
    )
    
    # Custom query for your BGG Gold Schema
    query = """
    WITH LastPlayed AS (
        SELECT GameID, MAX(PlayDate) as LastPlayDate
        FROM bgg.gold.plays_detail
        GROUP BY GameID
    ),
    WinCounts AS (
        SELECT GameID, PlayerName, COUNT(*) as Wins
        FROM bgg.gold.plays_detail
        WHERE PlayerWin = true
        GROUP BY GameID, PlayerName
    ),
    TopWinner AS (
        SELECT GameID, PlayerName, Wins,
               ROW_NUMBER() OVER(PARTITION BY GameID ORDER BY Wins DESC) as rn
        FROM WinCounts
    )
    SELECT 
        c.GameID,
        c.GameName,
        c.MinPlayers,
        c.MaxPlayers,
        c.AvgPlayTime,
        c.GameThumbnailURL,
        c.Description,
        lp.LastPlayDate,
        tw.PlayerName as Champion,
        tw.Wins as ChampionWins
    FROM bgg.gold.Collection c
    LEFT JOIN LastPlayed lp ON c.GameID = lp.GameID
    LEFT JOIN TopWinner tw ON c.GameID = tw.GameID AND tw.rn = 1
    WHERE c.Owned = true
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 3. Load Data
try:
    df = get_data()
except Exception as e:
    st.error(f"Error connecting to Databricks: {e}")
    st.stop()

# 4. Sidebar Filters
st.sidebar.header("🎯 Filter Options")

# Player Count
players = st.sidebar.slider("Number of Players", 1, 10, 4)

# Time
max_time = st.sidebar.slider("Max Time (minutes)", 15, 240, 90)

# 5. Filtering Logic
# We filter where the game supports the player count AND fits within the time
valid_games = df[
    (df['MinPlayers'] <= players) & 
    (df['MaxPlayers'] >= players) & 
    (df['AvgPlayTime'] <= max_time)
]

# 6. Main Interface
st.title("🎲 What are we playing?")

tab1, tab2 = st.tabs(["The Picker", "The Collection"])

with tab1:
    st.markdown(f"### Found **{len(valid_games)}** games")
    
    if st.button("🎲 Pick a Random Game", type="primary", use_container_width=True):
        if not valid_games.empty:
            game = valid_games.sample(1).iloc[0]
            
            st.divider()
            
            # Layout: Image on Left, Stats on Right
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if game['GameThumbnailURL']:
                    st.image(game['GameThumbnailURL'])
                else:
                    st.write("No Image")
            
            with col2:
                st.subheader(game['GameName'])
                st.caption(f"⏱️ {game['AvgPlayTime']} mins | 👥 {game['MinPlayers']}-{game['MaxPlayers']} players")
                
                # Show Champion
                if pd.notnull(game['Champion']):
                    st.success(f"🏆 **Champion:** {game['Champion']} ({game['ChampionWins']} wins)")
                else:
                    st.info("No recorded wins yet.")
                    
                # Show Last Played
                last_played = game['LastPlayDate'] if pd.notnull(game['LastPlayDate']) else "Never"
                st.write(f"**Last Played:** {last_played}")
                
            # Expandable description
            with st.expander("Read Description"):
                st.write(game['Description'])
                
        else:
            st.warning("No games match those filters! Try increasing time or changing player count.")

with tab2:
    # Clean table for browsing
    display_df = valid_games[['GameName', 'AvgPlayTime', 'MinPlayers', 'MaxPlayers', 'LastPlayDate', 'Champion']]
    st.dataframe(display_df, use_container_width=True, hide_index=True)