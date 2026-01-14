import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time
import html
import plotly.express as px
from collections import Counter, defaultdict
from datetime import datetime
import os

st.set_page_config(page_title="Seth's BG Tool", layout="wide", page_icon="🎲")

# --- CONFIGURATION & TOKEN HANDLING ---
if "BGG_API_TOKEN" in st.secrets:
    BGG_API_TOKEN = st.secrets["BGG_API_TOKEN"]
else:
    if "manual_token" not in st.session_state: st.session_state.manual_token = ""
    if not st.session_state.manual_token:
        st.sidebar.header("⚠️ Setup Required")
        user_token = st.sidebar.text_input("Enter BGG API Token:", type="password", help="Token not found in secrets.")
        if user_token:
            st.session_state.manual_token = user_token
            st.rerun()
        else:
            st.info("👋 Enter BGG API Token in sidebar to continue.")
            st.stop()
    BGG_API_TOKEN = st.session_state.manual_token

# --- SESSION STATE INITIALIZATION ---
if 'active_tab' not in st.session_state: st.session_state['active_tab'] = "🎲 Pick a Game"

# --- HELPER FUNCTIONS ---
def get_auth_session():
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {BGG_API_TOKEN}", "User-Agent": "StreamlitGamePicker/10.0", "Accept": "application/xml"})
    return session

def clean_description(desc_text):
    if not desc_text: return "No description available."
    return html.unescape(desc_text)

def get_best_player_count(poll_tag):
    if not poll_tag: return "N/A"
    best_count = "N/A"
    max_best_votes = -1
    for results in poll_tag.findall('results'):
        num_players = results.get('numplayers')
        try: best_votes = int(results.find("result[@value='Best']").get('numvotes'))
        except: best_votes = 0
        if best_votes > max_best_votes and best_votes > 0:
            max_best_votes = best_votes
            best_count = num_players
    return best_count

# --- DATA LOADING ENGINE (CSV vs API) ---

@st.cache_data(ttl=3600)
def fetch_from_api(username):
    """Fetches fresh data from BGG API."""
    session = get_auth_session()
    subtypes = ['boardgame', 'boardgameexpansion']
    combined_items = []
    
    for stype in subtypes:
        with st.spinner(f"Fetching {stype}s from BGG..."):
            url = f"https://boardgamegeek.com/xmlapi2/collection?username={username}&subtype={stype}"
            attempts = 0
            while attempts < 5:
                try:
                    r = session.get(url)
                    if r.status_code == 200: 
                        try:
                            root = ET.fromstring(r.content)
                            combined_items.extend(root.findall('item'))
                        except: pass
                        break
                    elif r.status_code == 202: time.sleep(4); attempts += 1
                    elif r.status_code == 429: time.sleep(5); attempts += 1
                    elif r.status_code == 401: st.error("Auth failed."); break
                    else: time.sleep(2); attempts += 1
                except: attempts += 1; time.sleep(2)
        time.sleep(1)

    if not combined_items: return pd.DataFrame()

    collection_map = {}
    ownership_map = {}
    for item in combined_items:
        g_id = item.get('objectid')
        try: plays = int(item.find('numplays').text)
        except: plays = 0
        status = item.find('status')
        is_owned = status.get('own') == "1" if status is not None else False
        collection_map[g_id] = plays
        ownership_map[g_id] = is_owned

    game_ids = list(collection_map.keys())
    if not game_ids: return pd.DataFrame()

    batch_size = 20
    all_games = []
    progress_bar = st.progress(0, text=f"Analyzing {len(game_ids)} items...")
    
    for i in range(0, len(game_ids), batch_size):
        batch = game_ids[i:i + batch_size]
        ids_str = ",".join(batch)
        details_url = f"https://boardgamegeek.com/xmlapi2/thing?id={ids_str}&stats=1"
        try:
            r = session.get(details_url)
            if r.status_code == 200:
                det_root = ET.fromstring(r.content)
                for item in det_root.findall('item'):
                    try:
                        g_id = item.get('id')
                        g_type = item.get('type')
                        name = item.find("name[@type='primary']").get('value')
                        image = item.find('image').text if item.find('image') is not None else None
                        if not image: image = item.find('thumbnail').text if item.find('thumbnail') is not None else None
                        desc = clean_description(item.find('description').text)
                        min_p = int(item.find("minplayers").get('value'))
                        max_p = int(item.find("maxplayers").get('value'))
                        time_p = int(item.find("playingtime").get('value'))
                        min_age = int(item.find("minage").get('value')) if item.find("minage") is not None else 0
                        stats = item.find('statistics').find('ratings')
                        rating = float(stats.find('average').get('value'))
                        weight = float(stats.find('averageweight').get('value'))
                        mechanics = [l.get('value') for l in item.findall("link[@type='boardgamemechanic']")]
                        categories = [l.get('value') for l in item.findall("link[@type='boardgamecategory']")]
                        family_mechanisms = [link.get('value').replace("Mechanism:", "").strip() for link in item.findall("link[@type='boardgamefamily']") if link.get('value').startswith("Mechanism:")]
                        poll = item.find("poll[@name='suggested_numplayers']")
                        best_at = get_best_player_count(poll)
                        
                        all_games.append({
                            "Name": name, "Image": image, "Description": desc, "Type": g_type,
                            "MinPlayers": min_p, "MaxPlayers": max_p, "MinAge": min_age,
                            "BestPlayers": best_at, "Time": time_p, "Rating": rating, "Weight": weight,
                            "NumPlays": collection_map.get(g_id, 0), "IsOwned": ownership_map.get(g_id, False),
                            "Mechanics": mechanics, "FamilyMechanisms": family_mechanisms, "Categories": categories, "ID": g_id
                        })
                    except: continue
        except: pass
        progress_bar.progress(min((i + batch_size) / len(game_ids), 1.0))
        time.sleep(0.5)
    progress_bar.empty()
    
    df = pd.DataFrame(all_games)
    if not df.empty and 'Type' not in df.columns: df['Type'] = 'boardgame'
    return df

def load_data(username):
    """Smart loader: tries CSV first, then API."""
    csv_file = "bgg_collection.csv"
    
    # 1. If explicit reload requested, skip CSV and go to API
    if st.session_state.get('force_reload', False):
        df = fetch_from_api(username)
        st.session_state['force_reload'] = False # Reset flag
        return df, "api"

    # 2. Try loading local CSV
    if os.path.exists(csv_file):
        try:
            # We need to eval the list columns because CSV saves them as strings
            df = pd.read_csv(csv_file, converters={
                'Mechanics': eval, 
                'Categories': eval, 
                'FamilyMechanisms': eval
            })
            return df, "csv"
        except:
            pass # CSV might be corrupt, fall through to API

    # 3. Fallback to API
    return fetch_from_api(username), "api"

# --- HISTORY & STATS FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_full_play_history(username):
    session = get_auth_session()
    all_plays = []
    page = 1
    status_text = st.empty()
    while True:
        status_text.text(f"Fetching play history page {page}...")
        url = f"https://boardgamegeek.com/xmlapi2/plays?username={username}&page={page}"
        try:
            r = session.get(url); 
            if r.status_code != 200: break
            root = ET.fromstring(r.content)
            plays = root.findall('play')
            if not plays: break
            for play in plays:
                try:
                    date_str = play.get('date')
                    game_name = play.find('item').get('name')
                    game_id = play.find('item').get('objectid')
                    player_list = play.find('players')
                    if not player_list: continue
                    participants = []
                    for p in player_list.findall('player'):
                        try: score = float(p.get('score'))
                        except: score = -99999 
                        participants.append({'name': p.get('name'), 'score': score, 'win': p.get('win') == '1'})
                    participants.sort(key=lambda x: x['score'], reverse=True)
                    for i, p in enumerate(participants):
                        if i > 0 and p['score'] == participants[i-1]['score'] and p['score'] != -99999: p['rank'] = participants[i-1]['rank']
                        elif p['score'] == -99999: p['rank'] = None 
                        else: p['rank'] = i + 1
                    for p in participants:
                        all_plays.append({
                            'Date': pd.to_datetime(date_str), 'Year': pd.to_datetime(date_str).year,
                            'Game': game_name, 'GameID': game_id,
                            'Player': p['name'], 'Win': p['win'], 'Rank': p['rank'],
                            'Score': p['score'] if p['score'] != -99999 else None,
                            'Opponents': [x['name'] for x in participants if x['name'] != p['name']]
                        })
                except: continue
            page += 1; time.sleep(0.5)
        except: break
    status_text.empty()
    return pd.DataFrame(all_plays)

def fetch_game_stats(username, game_id):
    session = get_auth_session()
    url = f"https://boardgamegeek.com/xmlapi2/plays?username={username}&id={game_id}&page=1"
    try:
        r = session.get(url)
        if r.status_code != 200: return None, 0
        root = ET.fromstring(r.content)
        total_plays = int(root.get('total'))
        player_stats = {} 
        for play in root.findall('play'):
            players = play.find('players')
            if not players: continue
            for p in players.findall('player'):
                name = p.get('name')
                win = 1 if p.get('win') == '1' else 0
                if name not in player_stats: player_stats[name] = {'wins': 0, 'plays': 0}
                player_stats[name]['plays'] += 1
                player_stats[name]['wins'] += win
        data = []
        for name, stats in player_stats.items():
            if name.lower() != 'anonymous player':
                win_pct = (stats['wins'] / stats['plays']) * 100 if stats['plays'] > 0 else 0
                data.append({"Player": name, "Wins": stats['wins'], "Plays": stats['plays'], "Win Rate": win_pct})
        return pd.DataFrame(data).sort_values(by="Wins", ascending=False) if data else None, total_plays
    except: return None, 0

def render_game_card(game, username):
    st.divider()
    cA, cB = st.columns([1, 2])
    with cA:
        if game['Image']: st.image(game['Image'], use_container_width=True)
        if not game['IsOwned']: st.caption("⚠️ Not currently owned")
        if 'Type' in game and game['Type'] == 'boardgameexpansion': st.caption("🧩 Expansion")
    with cB:
        st.subheader(game['Name'])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rating", f"{game['Rating']:.1f}")
        m2.metric("Weight", f"{game['Weight']:.2f}")
        m3.metric("Time", f"{game['Time']}m")
        m4.metric("Plays", f"{game['NumPlays']}")
        st.caption(f"Category: {', '.join(game['Categories'][:3])}")
        st.caption(f"Mechanics: {', '.join((game['Mechanics'][:3] + game['FamilyMechanisms'][:2]))}")
        st.write(f"**Players:** {game['MinPlayers']} - {game['MaxPlayers']} (Best: {game['BestPlayers']})")
        if game['NumPlays'] > 0:
            with st.expander("📊 Play History & Win Rates"):
                with st.spinner("Fetching history..."):
                    df_stats, total_plays = fetch_game_stats(username, game['ID'])
                if df_stats is not None and not df_stats.empty:
                     sc1, sc2 = st.columns([2, 1])
                     with sc1:
                         fig = px.pie(df_stats, values='Wins', names='Player', title=f"Win Share (Total Plays: {total_plays})", hole=0.4)
                         fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                         st.plotly_chart(fig, use_container_width=True)
                     with sc2:
                         st.dataframe(df_stats[['Player', 'Win Rate', 'Wins']], hide_index=True, use_container_width=True, column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%")})
        else: st.info("No play history found.")
        with st.expander("📖 Description"): st.markdown(game['Description'], unsafe_allow_html=True)
        st.markdown(f"[View on BGG](https://boardgamegeek.com/boardgame/{game['ID']})")

# --- APP START ---
st.sidebar.title("Seth's BG Tool")
username = st.sidebar.text_input("BGG Username", value="sparker0285")

# --- DATA RELOAD LOGIC ---
if st.sidebar.button("Reload Collection from BGG"):
    st.session_state['force_reload'] = True
    st.cache_data.clear()
    st.rerun()

pick_qty = st.sidebar.number_input("Pick Quantity", 1, 5, 1)

if username:
    full_df, source = load_data(username)
    
    # Show status/download options if loaded from API
    if source == "api" and not full_df.empty:
        st.sidebar.success("Loaded fresh data from BGG!")
        csv = full_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="💾 Download CSV to Persist",
            data=csv,
            file_name="bgg_collection.csv",
            mime="text/csv",
            help="Download this file, overwrite the one in your project folder, and push to GitHub. This makes the app load instantly next time!"
        )
    elif source == "csv":
        st.sidebar.info("⚡ Loaded instantly from CSV")
else: st.stop()

if full_df.empty:
    st.warning(f"No games found for user: {username}.")
    st.stop()

# --- SEGREGATION ---
if 'Type' in full_df.columns:
    owned_df = full_df[(full_df['IsOwned'] == True) & (full_df['Type'] == 'boardgame')].copy()
else:
    owned_df = full_df[full_df['IsOwned'] == True].copy()

# --- FILTERS ---
st.sidebar.header("Criteria")
def get_sorted_options(dataframe, column_name):
    if dataframe.empty: return []
    all_lists = dataframe[column_name].tolist()
    flat_list = [item for sublist in all_lists for item in sublist]
    counts = Counter(flat_list)
    return [m[0] for m in counts.most_common()]

sorted_fam_mechs = get_sorted_options(owned_df, 'FamilyMechanisms')
selected_fam_mechs = st.sidebar.multiselect("Game Type", sorted_fam_mechs)
sorted_mechs = get_sorted_options(owned_df, 'Mechanics')
selected_mechanics = st.sidebar.multiselect("Game Mechanics", sorted_mechs)
sorted_cats = get_sorted_options(owned_df, 'Categories')
selected_cats = st.sidebar.multiselect("Game Categories", sorted_cats)
player_count = st.sidebar.slider("Number of Players", 1, 10, 4)
strict_best = st.sidebar.checkbox("Only 'Best At' this count", value=False, help="Only show games voted Best at this count.")

# UPDATED: Added (pile of shame) text
play_status = st.sidebar.radio("History", ["All", "Played", "Unplayed (pile of shame)"])

c1, c2 = st.sidebar.columns(2)
max_age_req = c1.slider("Max Age", 4, 18, 18)
max_time = c2.slider("Max Time", 15, 240, 90)
weight_range = st.sidebar.slider("Complexity", 1.0, 5.0, (1.0, 5.0))

mask = (owned_df['Time'] <= max_time) & (owned_df['Weight'].between(weight_range[0], weight_range[1])) & (owned_df['MinAge'] <= max_age_req) 
if play_status == "Played": mask = mask & (owned_df['NumPlays'] > 0)
elif play_status == "Unplayed (pile of shame)": mask = mask & (owned_df['NumPlays'] == 0) # Logic Updated to match text
if strict_best: mask = mask & (owned_df['BestPlayers'].astype(str) == str(player_count))
else: mask = mask & (owned_df['MinPlayers'] <= player_count) & (owned_df['MaxPlayers'] >= player_count)
if selected_mechanics: mask = mask & owned_df['Mechanics'].apply(lambda x: bool(set(selected_mechanics) & set(x)))
if selected_cats: mask = mask & owned_df['Categories'].apply(lambda x: bool(set(selected_cats) & set(x)))
if selected_fam_mechs: mask = mask & owned_df['FamilyMechanisms'].apply(lambda x: bool(set(selected_fam_mechs) & set(x)))
valid_owned_games = owned_df[mask]

# --- MAIN NAVIGATION ---
nav_options = ["🎲 Pick a Game", "👤 Player Stats", "🔍 Search for a Game", "📜 List View"]
selection = st.radio("Navigation", nav_options, horizontal=True, label_visibility="collapsed", index=nav_options.index(st.session_state['active_tab']), key="nav_radio")
if selection != st.session_state['active_tab']: st.session_state['active_tab'] = selection

if st.session_state['active_tab'] == "🎲 Pick a Game":
    st.markdown(f"### **{len(valid_owned_games)} out of {len(owned_df)}** games match criteria")
    if st.button("🎲 Pick Game(s)", type="primary", use_container_width=True):
        if not valid_owned_games.empty:
            picked = valid_owned_games.sample(min(pick_qty, len(valid_owned_games)))
            for _, game in picked.iterrows(): render_game_card(game, username)
        else: st.warning("No games found. Adjust filters.")

elif st.session_state['active_tab'] == "👤 Player Stats":
    st.markdown("### 👤 Deep Player Analysis")
    if 'history_df' not in st.session_state:
        st.info("This tab downloads your *entire* play history. It may take a minute.")
        if st.button("Load Full Play History"):
            with st.spinner("Downloading full history from BGG..."):
                plays_df = fetch_full_play_history(username)
                if not plays_df.empty:
                    st.session_state['history_df'] = plays_df
                    st.rerun()
    if 'history_df' in st.session_state:
        plays_df = st.session_state['history_df']
        all_players = sorted(list(set(plays_df['Player'].unique())))
        default_ix = all_players.index(username) if username in all_players else 0
        selected_player = st.selectbox("Select Player to Analyze", all_players, index=default_ix)
        p_df = plays_df[plays_df['Player'] == selected_player].copy()
        meta_lookup = full_df.set_index('ID')[['Mechanics', 'Categories', 'FamilyMechanisms']].to_dict('index')
        stats_mech = defaultdict(lambda: {'plays': 0, 'wins': 0})
        stats_cat = defaultdict(lambda: {'plays': 0, 'wins': 0})
        stats_fam = defaultdict(lambda: {'plays': 0, 'wins': 0})
        for _, row in p_df.iterrows():
            gid = row['GameID']
            won = row['Win']
            if gid in meta_lookup:
                def update_stats(source_list, target_dict):
                    for item in source_list:
                        target_dict[item]['plays'] += 1
                        if won: target_dict[item]['wins'] += 1
                update_stats(meta_lookup[gid]['Mechanics'], stats_mech)
                update_stats(meta_lookup[gid]['Categories'], stats_cat)
                update_stats(meta_lookup[gid]['FamilyMechanisms'], stats_fam)
        def create_stats_df(stats_dict, limit=5, by_win_rate=False):
            data = []
            for k, v in stats_dict.items():
                if by_win_rate and v['plays'] < 3: continue 
                wr = (v['wins'] / v['plays']) * 100 if v['plays'] > 0 else 0
                data.append({'Name': k, 'Plays': v['plays'], 'Win Rate': wr})
            df = pd.DataFrame(data)
            if df.empty: return df
            sort_col = 'Win Rate' if by_win_rate else 'Plays'
            return df.sort_values(by=sort_col, ascending=False).head(limit)
        k1, k2, k3, k4 = st.columns(4)
        total_wins = len(p_df[p_df['Win'] == True])
        total_plays = len(p_df)
        win_rate = (total_wins / total_plays * 100) if total_plays > 0 else 0
        avg_rank = p_df['Rank'].mean()
        k1.metric("Total Plays", total_plays)
        k2.metric("Wins", total_wins)
        k3.metric("Win Rate", f"{win_rate:.1f}%")
        k4.metric("Avg Rank", f"{avg_rank:.2f}" if pd.notna(avg_rank) else "N/A")
        st.divider()
        st.subheader(f"🧠 {selected_player}'s Favorites (Most Played)")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown("**Game Types**"); st.dataframe(create_stats_df(stats_fam, by_win_rate=False), hide_index=True, use_container_width=True, column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%")})
        with c2: st.markdown("**Mechanics**"); st.dataframe(create_stats_df(stats_mech, by_win_rate=False), hide_index=True, use_container_width=True, column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%")})
        with c3: st.markdown("**Categories**"); st.dataframe(create_stats_df(stats_cat, by_win_rate=False), hide_index=True, use_container_width=True, column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%")})
        st.divider()
        st.subheader(f"🥇 {selected_player}'s Best (Highest Win Rate - min 3 plays)")
        b1, b2, b3 = st.columns(3)
        with b1: st.markdown("**Game Types**"); st.dataframe(create_stats_df(stats_fam, by_win_rate=True), hide_index=True, use_container_width=True, column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%")})
        with b2: st.markdown("**Mechanics**"); st.dataframe(create_stats_df(stats_mech, by_win_rate=True), hide_index=True, use_container_width=True, column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%")})
        with b3: st.markdown("**Categories**"); st.dataframe(create_stats_df(stats_cat, by_win_rate=True), hide_index=True, use_container_width=True, column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%")})
        st.divider()
        c1, c2 = st.columns(2)
        with c1: year_counts = p_df.groupby('Year').size().reset_index(name='Plays'); fig_year = px.bar(year_counts, x='Year', y='Plays', title="Activity by Year"); st.plotly_chart(fig_year, use_container_width=True)
        with c2: year_wins = p_df.groupby('Year')['Win'].mean().reset_index(name='WinRate'); year_wins['WinRate'] = year_wins['WinRate'] * 100; fig_wr = px.line(year_wins, x='Year', y='WinRate', title="Win Rate % over Time", markers=True); st.plotly_chart(fig_wr, use_container_width=True)
        coop_ids = set(full_df[full_df['Mechanics'].apply(lambda x: "Cooperative Game" in x)]['ID'].astype(str))
        game_stats = p_df.groupby(['Game', 'GameID']).agg(Plays=('Game', 'count'), Wins=('Win', 'sum'), AvgRank=('Rank', 'mean')).reset_index()
        game_stats['WinRate'] = (game_stats['Wins'] / game_stats['Plays']) * 100
        st.subheader(f"🏆 Top Games (Performance)")
        perf_df = game_stats[(game_stats['Plays'] >= 3) & (~game_stats['GameID'].isin(coop_ids))].sort_values(by=['WinRate', 'Plays'], ascending=False).head(10)
        st.caption("Excludes Cooperative Games (min 3 plays)")
        st.dataframe(perf_df[['Game', 'WinRate', 'Plays', 'AvgRank']], use_container_width=True, hide_index=True, column_config={"WinRate": st.column_config.NumberColumn(format="%.1f%%"), "AvgRank": st.column_config.NumberColumn(format="%.2f")})
        st.subheader(f"🎲 Top Games (Play Count)")
        play_count_df = game_stats.sort_values(by='Plays', ascending=False).head(10)
        st.dataframe(play_count_df[['Game', 'Plays', 'AvgRank', 'WinRate']], use_container_width=True, hide_index=True, column_config={"WinRate": st.column_config.NumberColumn(format="%.1f%%"), "AvgRank": st.column_config.NumberColumn(format="%.2f")})
        st.subheader("⚔️ Frequent Opponents")
        opponent_stats = {} 
        for index, row in p_df.iterrows():
            did_i_win = row['Win']
            opps = row['Opponents']
            for opp in opps:
                if opp not in opponent_stats: opponent_stats[opp] = {'games': 0, 'wins': 0}
                opponent_stats[opp]['games'] += 1
                if did_i_win: opponent_stats[opp]['wins'] += 1
        opp_data = []
        for opp_name, stats in opponent_stats.items():
            win_rate_vs = (stats['wins'] / stats['games']) * 100
            opp_data.append({"Opponent": opp_name, "Games Together": stats['games'], "My Win Rate": win_rate_vs})
        if opp_data:
            opp_df = pd.DataFrame(opp_data).sort_values(by='Games Together', ascending=False).head(10)
            st.dataframe(opp_df, use_container_width=True, hide_index=True, column_config={"My Win Rate": st.column_config.NumberColumn(format="%.1f%%", help="Percentage of games YOU won when playing against this opponent.")})
        else: st.write("No opponent data found (Solo plays?)")

elif st.session_state['active_tab'] == "🔍 Search for a Game":
    if 'Type' in full_df.columns:
        total_bg = len(full_df[full_df['Type'] == 'boardgame'])
        total_exp = len(full_df[full_df['Type'] == 'boardgameexpansion'])
    else: total_bg = len(full_df); total_exp = 0
    st.markdown(f"### Find a Game (Collection: {total_bg} Games + {total_exp} Expansions)")
    search_options = full_df.sort_values("Name")['Name'].tolist()
    selected_game_name = st.selectbox("Select a game:", options=search_options, index=None, placeholder="Type to search...")
    if selected_game_name:
        game_row = full_df[full_df['Name'] == selected_game_name].iloc[0]
        render_game_card(game_row, username)

elif st.session_state['active_tab'] == "📜 List View":
    v_df = valid_owned_games.copy()
    v_df['Mechanics'] = v_df['Mechanics'].apply(lambda x: ", ".join(x))
    v_df['Categories'] = v_df['Categories'].apply(lambda x: ", ".join(x))
    v_df['FamilyMechanisms'] = v_df['FamilyMechanisms'].apply(lambda x: ", ".join(x))
    st.dataframe(v_df[['Name', 'NumPlays', 'BestPlayers', 'Weight', 'Rating', 'Time', 'Categories']], use_container_width=True, hide_index=True, column_config={"Rating": st.column_config.NumberColumn(format="%.1f")})