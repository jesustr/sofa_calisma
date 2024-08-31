from typing import Optional
def corner(
    home_team: str,
    away_team: str,
    period: int,
    max_home_sira: Optional[int] = None,
    min_home_sira: Optional[int] = None,
    max_away_sira: Optional[int] = None,
    min_away_sira: Optional[int] = None
) -> None:
    """
    Girilen parametrelere bagli olarak tum korner detaylarini grafik olarak ekrana getirir.

    Args:
        home_team,away_team  (str): Contains seklinde calisir aranilan takimin isminin bir kisminin yazilmasi yeterli. Ilk harf buyuk olmali.
        max_home_sira, min_home_sira, max_away_sira, min_away_sira (int): Verilen degerler arasindaki siraya sahip rakipleri arasindaki maclarin grafiklerini getirir. Deger verilmezse bu grafikler gozukmez.
        period (int): 1-4 arasi deger alir. Seasonun ceyreklerini temsil eder. Verilen degerdeki donemin grafiklerini gosterir.Deger verilmez ise o kisim gelmez.
         
    Returns:
        Aralarindaki korner
        Ev son 20 korner
        Dep son 20 korner
        Ev son 10 korner peroid a gore
        Dep son 10 orner period a gore
        Ev son 10 korner siralamaya gore
        Dep son 10 korner siralamaya gore
    """

    import pandas as pd 
    import numpy as np 
    from IPython.display import HTML
    from IPython.display import display
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd.set_option('display.max_colwidth', None) 
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    import matplotlib.gridspec as gridspec
    df_all = pd.read_csv('/Users/icy/sofascore/sofa_calisma/csvler/last_df.csv', parse_dates=['date'])
    df = df_all[['date','org_detay','ulke','home_team','away_team','home_goal','away_goal','corner_kick_total','corner_kicks_home','corner_kicks_away','corner_kick_total_firsthalf','corner_kicks_home_firsthalf',
    'corner_kicks_away_firsthalf','corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf','season_period','place_away','place_home','match_round']]
    df = df[~(df.ulke == 'Europe')]
    fig, ax = plt.subplots(2,3,figsize = (21,9), layout='tight') 
    #------ax[0,0]------  ARALARINDAKI KORNER ------------
    df_aralarinda_corner_all = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total','corner_kicks_home','corner_kicks_away']][:10]
    team_home = df_aralarinda_corner_all['home_team'].values[0] #-> takım isimleri home-dep
    team_away = df_aralarinda_corner_all['away_team'].values[0]

    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(corner_dates))
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax[0,0].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax[0,0].bar_label(rects, padding=3)
        multiplier += 1
    ax[0,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[0,0].set_title(f'{team_home} - {team_away} Corner Kick' )
    ax[0,0].set_ylabel('Count')
    ax[0,0].set_ylim(0, df_aralarinda_corner['corner_kick_total'].max() + 3) 
    ax[0,0].set_xticks(x_length + width )
    ax[0,0].set_xticklabels(corner_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[0,1]------  ARALARINDAKI KORNER FIRST HALF ------------
    df_aralarinda_corner_all = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_firsthalf','corner_kicks_home_firsthalf','corner_kicks_away_firsthalf']][:10]
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax[0,1].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax[0,1].bar_label(rects, padding=3)
        multiplier += 1
    ax[0,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[0,1].set_title(f'{team_home} - {team_away} First Half Corner Kick' )
    ax[0,1].set_ylabel('Count')
    ax[0,1].set_ylim(0, df_aralarinda_corner['corner_kick_total_firsthalf'].max() + 3) 
    ax[0,1].set_xticks(x_length + width )
    ax[0,1].set_xticklabels(corner_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[0,2]------  ARALARINDAKI KORNER SECOND HALF ------------
    df_aralarinda_corner_all = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf']][:10]
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax[0,2].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax[0,2].bar_label(rects, padding=3)
        multiplier += 1
    ax[0,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[0,2].set_title(f'{team_home} - {team_away} Second Half Corner Kick' )
    ax[0,2].set_ylabel('Count')
    ax[0,2].set_ylim(0, df_aralarinda_corner['corner_kick_total_secondhalf'].max() + 3) 
    ax[0,2].set_xticks(x_length + width )
    ax[0,2].set_xticklabels(corner_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[1,0]------  ARALARINDAKI KORNER ------------
    df_aralarinda_corner_all = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total','corner_kicks_home','corner_kicks_away']][:10]
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax[1,0].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax[1,0].bar_label(rects, padding=3)
        multiplier += 1
    ax[1,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[1,0].set_title(f'{team_away} - {team_home} Corner Kick' )
    ax[1,0].set_ylabel('Count')
    ax[1,0].set_ylim(0, df_aralarinda_corner['corner_kick_total'].max() + 3) 
    ax[1,0].set_xticks(x_length + width )
    ax[1,0].set_xticklabels(corner_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[1,1]------  ARALARINDAKI KORNER FIRST HALF ------------
    df_aralarinda_corner_all = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_firsthalf','corner_kicks_home_firsthalf','corner_kicks_away_firsthalf']][:10]
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax[1,1].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax[1,1].bar_label(rects, padding=3)
        multiplier += 1
    ax[1,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[1,1].set_title(f'{team_away} - {team_home} First Half Corner Kick' )
    ax[1,1].set_ylabel('Count')
    ax[1,1].set_ylim(0, df_aralarinda_corner['corner_kick_total_firsthalf'].max() + 3) 
    ax[1,1].set_xticks(x_length + width )
    ax[1,1].set_xticklabels(corner_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[1,2]------  ARALARINDAKI KORNER SECOND HALF ------------
    df_aralarinda_corner_all = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf']][:10]
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax[1,2].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax[1,2].bar_label(rects, padding=3)
        multiplier += 1
    ax[1,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[1,2].set_title(f'{team_away} - {team_home} Second Half Corner Kick' )
    ax[1,2].set_ylabel('Count')
    ax[1,2].set_ylim(0, df_aralarinda_corner['corner_kick_total_secondhalf'].max() + 3) 
    ax[1,2].set_xticks(x_length + width )
    ax[1,2].set_xticklabels(corner_dates, rotation= 45, ha= 'right', fontsize= 7)

    fig.suptitle("Corner Kicks Overview", fontsize=16, y=1.02)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(18,24))
    gs = gridspec.GridSpec(6,1)
    #------ax1------  EV SON 20 KORNER ------------
    ax1 = plt.subplot(gs[0, : ])
    df_aralarinda_corner_all = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total','corner_kicks_home','corner_kicks_away']][:20]
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    away_teams = list(df_aralarinda_corner_all['away_team'][:20])
    place_away = list(df_aralarinda_corner_all['place_away'][:20])
    match_round = list(df_aralarinda_corner_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, away_teams, place_away, match_round)]
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax1.bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax1.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax1.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax1.set_title(f'{team_home} Son 20 Home Maci Korner Kick (w= week, p= place)')
    ax1.set_ylabel('Count')
    ax1.set_ylim(0, df_aralarinda_corner['corner_kick_total'].max() + 3) 
    ax1.set_xticks(x_length + width + x_shift )
    ax1.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # centik gizleme 
    ax1.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax2------  EV SON 20 KORNER FIRST HALF ------------
    ax2 = plt.subplot(gs[1, : ])
    df_aralarinda_corner_all = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_firsthalf','corner_kicks_home_firsthalf','corner_kicks_away_firsthalf']][:20]
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    away_teams = list(df_aralarinda_corner_all['away_team'][:20])
    place_away = list(df_aralarinda_corner_all['place_away'][:20])
    match_round = list(df_aralarinda_corner_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, away_teams, place_away, match_round)]
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax2.bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax2.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax2.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax2.set_title(f'{team_home} Son 20 Ev Maci Korner Kick First Half (w= week, p= place)')
    ax2.set_ylabel('Count')
    ax2.set_ylim(0, df_aralarinda_corner['corner_kick_total_firsthalf'].max() + 3) 
    ax2.set_xticks(x_length + width + x_shift )
    ax2.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # Centik gizleme 
    ax2.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax3------  EV SON 20 KORNER SECOND HALF ------------
    ax3 = plt.subplot(gs[2, : ])
    df_aralarinda_corner_all = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf']][:20]
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    away_teams = list(df_aralarinda_corner_all['away_team'][:20])
    place_away = list(df_aralarinda_corner_all['place_away'][:20])
    match_round = list(df_aralarinda_corner_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, away_teams, place_away, match_round)]
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax3.bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax3.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax3.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax3.set_title(f'{team_home} Son 20 Home Maci Korner Kick Second Half (w= week, p= place)')
    ax3.set_ylabel('Count')
    ax3.set_ylim(0, df_aralarinda_corner['corner_kick_total_secondhalf'].max() + 3) 
    ax3.set_xticks(x_length + width + x_shift )
    ax3.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # Centik gizleme 
    ax3.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax5------  DEP SON 20 KORNER ------------
    ax4 = plt.subplot(gs[3, : ])
    df_aralarinda_corner_all = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total','corner_kicks_home','corner_kicks_away']][:20]
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    home_teams = list(df_aralarinda_corner_all['home_team'][:20])
    place_home = list(df_aralarinda_corner_all['place_home'][:20])
    match_round = list(df_aralarinda_corner_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, home_teams, place_home, match_round)]
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax4.bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax4.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax4.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax4.set_title(f'{team_away} Son 20 Dep Maci Korner Kick (w= week, p= place)')
    ax4.set_ylabel('Count')
    ax4.set_ylim(0, df_aralarinda_corner['corner_kick_total'].max() + 3) 
    ax4.set_xticks(x_length + width + x_shift )
    ax4.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # centik gizleme 
    ax4.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax5------  DEP SON 20 KORNER FIRST HALF ------------
    ax5 = plt.subplot(gs[4, : ])
    df_aralarinda_corner_all = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_firsthalf','corner_kicks_home_firsthalf','corner_kicks_away_firsthalf']][:20]
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    home_teams = list(df_aralarinda_corner_all['home_team'][:20])
    place_home = list(df_aralarinda_corner_all['place_home'][:20])
    match_round = list(df_aralarinda_corner_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, home_teams, place_home, match_round)]
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax5.bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax5.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax5.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax5.set_title(f'{team_away} Son 20 Dep Maci Korner Kick First Half (w= week, p= place)')
    ax5.set_ylabel('Count')
    ax5.set_ylim(0, df_aralarinda_corner['corner_kick_total_firsthalf'].max() + 3) 
    ax5.set_xticks(x_length + width + x_shift )
    ax5.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # Centik gizleme 
    ax5.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax6------  DEP SON 20 KORNER SECOND HALF ------------
    ax6 = plt.subplot(gs[5, : ])
    df_aralarinda_corner_all = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf']][:20]
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    home_teams = list(df_aralarinda_corner_all['home_team'][:20])
    place_home = list(df_aralarinda_corner_all['place_home'][:20])
    match_round = list(df_aralarinda_corner_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, home_teams, place_home, match_round)]
    x_length = np.arange(len(corner_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    corner_dates = list(df_aralarinda_corner_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_corner.columns : 
        offset = width * multiplier
        rects = ax6.bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
        ax6.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax6.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax6.set_title(f'{team_away} Son 20 Dep Maci Korner Kick Second Half (w= week, p= place)')
    ax6.set_ylabel('Count')
    ax6.set_ylim(0, df_aralarinda_corner['corner_kick_total_secondhalf'].max() + 3) 
    ax6.set_xticks(x_length + width + x_shift )
    ax6.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # Centik gizleme 
    ax6.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tight_layout()
    plt.show()
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if period: 
        fig, ax = plt.subplots(2,3,figsize = (21,9), layout='tight') 
        #------------  EV SON 10 KORNER SEASON_PERIOD A GORE ------------

        df_aralarinda_corner_all = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total','corner_kicks_home','corner_kicks_away']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_corner_all['away_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(corner_dates, away_teams)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[0,0].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[0,0].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,0].set_title(f'{team_home} Son 10 Home Maci Korner Kick Period = {period}')
        ax[0,0].set_ylabel('Count')
        ax[0,0].set_ylim(0, df_aralarinda_corner['corner_kick_total'].max() + 3) 
        ax[0,0].set_xticks(x_length + width + x_shift )
        ax[0,0].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,0].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  EV SON 10 KORNER FIRST HALF SEASON_PERIOD A GORE ------------

        df_aralarinda_corner_all = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_firsthalf','corner_kicks_home_firsthalf','corner_kicks_away_firsthalf']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_corner_all['away_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(corner_dates, away_teams)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[0,1].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[0,1].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,1].set_title(f'{team_home} Son 10 Home Maci Korner Kick First Half Period = {period}')
        ax[0,1].set_ylabel('Count')
        ax[0,1].set_ylim(0, df_aralarinda_corner['corner_kick_total_firsthalf'].max() + 3) 
        ax[0,1].set_xticks(x_length + width + x_shift )
        ax[0,1].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,1].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  EV SON 10 KORNER SECOND HALF SEASON_PERIOD A GORE ------------

        df_aralarinda_corner_all = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_corner_all['away_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(corner_dates, away_teams)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[0,2].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[0,2].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,2].set_title(f'{team_home} Son 10 Home Maci Korner Kick Period = {period}')
        ax[0,2].set_ylabel('Count')
        ax[0,2].set_ylim(0, df_aralarinda_corner['corner_kick_total_secondhalf'].max() + 3) 
        ax[0,2].set_xticks(x_length + width + x_shift )
        ax[0,2].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,2].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  DEP SON 10 KORNER SEASON_PERIOD A GORE ------------

        df_aralarinda_corner_all = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total','corner_kicks_home','corner_kicks_away']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_corner_all['home_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(corner_dates, home_teams)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[1,0].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[1,0].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,0].set_title(f'{team_away} Son 10 Dep Maci Korner Kick Period = {period}')
        ax[1,0].set_ylabel('Count')
        ax[1,0].set_ylim(0, df_aralarinda_corner['corner_kick_total'].max() + 3) 
        ax[1,0].set_xticks(x_length + width + x_shift )
        ax[1,0].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,0].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  DEF SON 10 KORNER FIRST HALF SEASON_PERIOD A GORE ------------

        df_aralarinda_corner_all = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_firsthalf','corner_kicks_home_firsthalf','corner_kicks_away_firsthalf']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_corner_all['home_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(corner_dates, home_teams)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[1,1].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[1,1].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,1].set_title(f'{team_away} Son 10 Dep Maci Korner Kick First Half Period = {period}')
        ax[1,1].set_ylabel('Count')
        ax[1,1].set_ylim(0, df_aralarinda_corner['corner_kick_total_firsthalf'].max() + 3) 
        ax[1,1].set_xticks(x_length + width + x_shift )
        ax[1,1].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,1].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  EV SON 10 KORNER SECOND HALF SEASON_PERIOD A GORE ------------

        df_aralarinda_corner_all = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_corner_all['home_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(corner_dates, home_teams)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[1,2].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[1,2].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,2].set_title(f'{team_away} Son 10 Dep Maci Korner Kick Period = {period}')
        ax[1,2].set_ylabel('Count')
        ax[1,2].set_ylim(0, df_aralarinda_corner['corner_kick_total_secondhalf'].max() + 3) 
        ax[1,2].set_xticks(x_length + width + x_shift )
        ax[1,2].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,2].tick_params(axis='x', which='both', bottom=False, top=False)
        plt.tight_layout()
        plt.show()
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #------------  SIRALAMAYA GORE SON 10 HOME KORNER ------------
    #min_sira, max_sira burda kullaniliyor 
    if max_away_sira and min_away_sira : 
        fig, ax = plt.subplots(2,3,figsize = (21,9), layout='tight')

        df_aralarinda_corner_all = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total','corner_kicks_home','corner_kicks_away']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_corner_all['away_team'][:10])
        place_away = list(df_aralarinda_corner_all['place_away'][:10])
        match_round = list(df_aralarinda_corner_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, away_teams, place_away, match_round)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[0,0].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[0,0].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,0].set_title(f'{team_home} Son 10 Home Maci, Rakip Sirasi : {min_away_sira}-{max_away_sira} Arasinda (w:week, p:place)')
        ax[0,0].set_ylabel('Count')
        ax[0,0].set_ylim(0, df_aralarinda_corner['corner_kick_total'].max() + 3) 
        ax[0,0].set_xticks(x_length + width + x_shift )
        ax[0,0].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,0].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 HOME KORNER FIRST HALF ------------

        df_aralarinda_corner_all = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_firsthalf','corner_kicks_home_firsthalf','corner_kicks_away_firsthalf']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_corner_all['away_team'][:10])
        place_away = list(df_aralarinda_corner_all['place_away'][:10])
        match_round = list(df_aralarinda_corner_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, away_teams, place_away, match_round)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[0,1].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[0,1].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,1].set_title(f'{team_home} Son 10 Home Maci First Half, Rakip Sirasi : {min_away_sira}-{max_away_sira} Arasinda ')
        ax[0,1].set_ylabel('Count')
        ax[0,1].set_ylim(0, df_aralarinda_corner['corner_kick_total_firsthalf'].max() + 3) 
        ax[0,1].set_xticks(x_length + width + x_shift )
        ax[0,1].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,1].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 HOME KORNER SECOND HALF ------------
    
        df_aralarinda_corner_all = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_corner_all['away_team'][:10])
        place_away = list(df_aralarinda_corner_all['place_away'][:10])
        match_round = list(df_aralarinda_corner_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, away_teams, place_away, match_round)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[0,2].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[0,2].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,2].set_title(f'{team_home} Son 10 Home Maci Second Half, Rakip Sirasi : {min_away_sira}-{max_away_sira} Arasinda')
        ax[0,2].set_ylabel('Count')
        ax[0,2].set_ylim(0, df_aralarinda_corner['corner_kick_total_secondhalf'].max() + 3) 
        ax[0,2].set_xticks(x_length + width + x_shift )
        ax[0,2].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,2].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 DEP KORNER ------------
    if max_home_sira and min_home_sira :
        df_aralarinda_corner_all = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total','corner_kicks_home','corner_kicks_away']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_corner_all['home_team'][:10])
        place_home = list(df_aralarinda_corner_all['place_home'][:10])
        match_round = list(df_aralarinda_corner_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, home_teams, place_home, match_round)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[1,0].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[1,0].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,0].set_title(f'{team_away} Son 10 Dep Maci, Rakip Sirasi : {min_home_sira}-{max_home_sira} Arasinda (w:week, p:place)')
        ax[1,0].set_ylabel('Count')
        ax[1,0].set_ylim(0, df_aralarinda_corner['corner_kick_total'].max() + 3) 
        ax[1,0].set_xticks(x_length + width + x_shift )
        ax[1,0].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,0].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 DEP KORNER FIRST HALF ------------

        df_aralarinda_corner_all = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_firsthalf','corner_kicks_home_firsthalf','corner_kicks_away_firsthalf']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_corner_all['home_team'][:10])
        place_home = list(df_aralarinda_corner_all['place_home'][:10])
        match_round = list(df_aralarinda_corner_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, home_teams, place_home, match_round)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[1,1].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[1,1].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,1].set_title(f'{team_away} Son 10 Dep Maci First Half, Rakip Sirasi : {min_home_sira}-{max_home_sira} Arasinda ')
        ax[1,1].set_ylabel('Count')
        ax[1,1].set_ylim(0, df_aralarinda_corner['corner_kick_total_firsthalf'].max() + 3) 
        ax[1,1].set_xticks(x_length + width + x_shift )
        ax[1,1].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,1].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 DEP KORNER SECOND HALF ------------
    
        df_aralarinda_corner_all = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_corner = df_aralarinda_corner[['corner_kick_total_secondhalf','corner_kicks_home_secondhalf','corner_kicks_away_secondhalf']][:10]
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_corner_all['home_team'][:10])
        place_home = list(df_aralarinda_corner_all['place_home'][:10])
        match_round = list(df_aralarinda_corner_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(corner_dates, home_teams, place_home, match_round)]
        x_length = np.arange(len(corner_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        corner_dates = list(df_aralarinda_corner_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_corner.columns : 
            offset = width * multiplier
            rects = ax[1,2].bar(x_length + offset, df_aralarinda_corner[column], width, label = column)
            ax[1,2].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,2].set_title(f'{team_away} Son 10 Dep Maci Second Half, Rakip Sirasi : {min_home_sira}-{max_home_sira} Arasinda')
        ax[1,2].set_ylabel('Count')
        ax[1,2].set_ylim(0, df_aralarinda_corner['corner_kick_total_secondhalf'].max() + 3) 
        ax[1,2].set_xticks(x_length + width + x_shift )
        ax[1,2].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,2].tick_params(axis='x', which='both', bottom=False, top=False)

        
    plt.tight_layout()
    plt.show()




def shoot_on_target(
    home_team: str,
    away_team: str,
    period: int,
    max_home_sira: Optional[int] = None,
    min_home_sira: Optional[int] = None,
    max_away_sira: Optional[int] = None,
    min_away_sira: Optional[int] = None
) -> None:
    """
    Girilen parametrelere bagli olarak tum Shoot On Target detaylarini grafik olarak ekrana getirir.

    Args:
        home_team,away_team  (str): Contains seklinde calisir aranilan takimin isminin bir kisminin yazilmasi yeterli. Ilk harf buyuk olmali.
        max_home_sira, min_home_sira, max_away_sira, min_away_sira (int): Verilen degerler arasindaki siraya sahip rakipleri arasindaki maclarin grafiklerini getirir. Deger verilmezse bu grafikler gozukmez.
        period (int): 1-4 arasi deger alir. Seasonun ceyreklerini temsil eder. Verilen degerdeki donemin grafiklerini gosterir.Deger verilmez ise o kisim gelmez.
         
    Returns:
        Aralarindaki shoot_on_target
        Ev son 20 shoot_on_target
        Dep son 20 shoot_on_target
        Ev son 10 Shoot On Target peroid a gore
        Dep son 10 orner period a gore
        Ev son 10 Shoot On Target siralamaya gore
        Dep son 10 Shoot On Target siralamaya gore
    """

    import pandas as pd 
    import numpy as np 
    from IPython.display import HTML
    from IPython.display import display
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd.set_option('display.max_colwidth', None) 
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    import matplotlib.gridspec as gridspec
    df_all = pd.read_csv('/Users/icy/sofascore/sofa_calisma/csvler/last_df.csv', parse_dates=['date'])
    df = df_all[['date','org_detay','ulke','home_team','away_team','home_goal','away_goal','shots_on_target_total','shots_on_target_home','shots_on_target_away','shots_on_target_total_firsthalf','shots_on_target_home_firsthalf',
    'shots_on_target_away_firsthalf','shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf','season_period','place_away','place_home','match_round']]
    df = df[~(df.ulke == 'Europe')]
    fig, ax = plt.subplots(2,3,figsize = (21,9), layout='tight') 
    #------ax[0,0]------  ARALARINDAKI Shoot On Target ------------
    df_aralarinda_shoot_all = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total','shots_on_target_home','shots_on_target_away']][:10]
    team_home = df_aralarinda_shoot_all['home_team'].values[0] #-> takım isimleri home-dep
    team_away = df_aralarinda_shoot_all['away_team'].values[0]

    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(shoot_dates))
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax[0,0].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax[0,0].bar_label(rects, padding=3)
        multiplier += 1
    ax[0,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[0,0].set_title(f'{team_home} - {team_away} Shoot On Target' )
    ax[0,0].set_ylabel('Count')
    ax[0,0].set_ylim(0, df_aralarinda_shoot['shots_on_target_total'].max() + 3) 
    ax[0,0].set_xticks(x_length + width )
    ax[0,0].set_xticklabels(shoot_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[0,1]------  ARALARINDAKI Shoot On Target FIRST HALF ------------
    df_aralarinda_shoot_all = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_firsthalf','shots_on_target_home_firsthalf','shots_on_target_away_firsthalf']][:10]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax[0,1].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax[0,1].bar_label(rects, padding=3)
        multiplier += 1
    ax[0,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[0,1].set_title(f'{team_home} - {team_away} First Half Shoot On Target' )
    ax[0,1].set_ylabel('Count')
    ax[0,1].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_firsthalf'].max() + 3) 
    ax[0,1].set_xticks(x_length + width )
    ax[0,1].set_xticklabels(shoot_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[0,2]------  ARALARINDAKI Shoot On Target SECOND HALF ------------
    df_aralarinda_shoot_all = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[(df.home_team.str.contains(home_team)) & (df.away_team.str.contains(away_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf']][:10]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax[0,2].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax[0,2].bar_label(rects, padding=3)
        multiplier += 1
    ax[0,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[0,2].set_title(f'{team_home} - {team_away} Second Half Shoot On Target' )
    ax[0,2].set_ylabel('Count')
    ax[0,2].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_secondhalf'].max() + 3) 
    ax[0,2].set_xticks(x_length + width )
    ax[0,2].set_xticklabels(shoot_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[1,0]------  ARALARINDAKI Shoot On Target ------------
    df_aralarinda_shoot_all = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total','shots_on_target_home','shots_on_target_away']][:10]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax[1,0].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax[1,0].bar_label(rects, padding=3)
        multiplier += 1
    ax[1,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[1,0].set_title(f'{team_away} - {team_home} Shoot On Target' )
    ax[1,0].set_ylabel('Count')
    ax[1,0].set_ylim(0, df_aralarinda_shoot['shots_on_target_total'].max() + 3) 
    ax[1,0].set_xticks(x_length + width )
    ax[1,0].set_xticklabels(shoot_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[1,1]------  ARALARINDAKI Shoot On Target FIRST HALF ------------
    df_aralarinda_shoot_all = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_firsthalf','shots_on_target_home_firsthalf','shots_on_target_away_firsthalf']][:10]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax[1,1].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax[1,1].bar_label(rects, padding=3)
        multiplier += 1
    ax[1,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[1,1].set_title(f'{team_away} - {team_home} First Half Shoot On Target' )
    ax[1,1].set_ylabel('Count')
    ax[1,1].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_firsthalf'].max() + 3) 
    ax[1,1].set_xticks(x_length + width )
    ax[1,1].set_xticklabels(shoot_dates, rotation= 45, ha= 'right', fontsize= 7)
    #------ax[1,2]------  ARALARINDAKI Shoot On Target SECOND HALF ------------
    df_aralarinda_shoot_all = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[(df.home_team.str.contains(away_team)) & (df.away_team.str.contains(home_team))].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf']][:10]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax[1,2].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax[1,2].bar_label(rects, padding=3)
        multiplier += 1
    ax[1,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax[1,2].set_title(f'{team_away} - {team_home} Second Half Shoot On Target' )
    ax[1,2].set_ylabel('Count')
    ax[1,2].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_secondhalf'].max() + 3) 
    ax[1,2].set_xticks(x_length + width )
    ax[1,2].set_xticklabels(shoot_dates, rotation= 45, ha= 'right', fontsize= 7)

    fig.suptitle("Shoot On Target Overview", fontsize=16, y=1.02)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(18,24))
    gs = gridspec.GridSpec(6,1)
    #------ax1------  EV SON 20 Shoot On Target ------------
    ax1 = plt.subplot(gs[0, : ])
    df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total','shots_on_target_home','shots_on_target_away']][:20]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    away_teams = list(df_aralarinda_shoot_all['away_team'][:20])
    place_away = list(df_aralarinda_shoot_all['place_away'][:20])
    match_round = list(df_aralarinda_shoot_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, away_teams, place_away, match_round)]
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax1.bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax1.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax1.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax1.set_title(f'{team_home} Son 20 Home Maci Shoot On Target (w= week, p= place)')
    ax1.set_ylabel('Count')
    ax1.set_ylim(0, df_aralarinda_shoot['shots_on_target_total'].max() + 3) 
    ax1.set_xticks(x_length + width + x_shift )
    ax1.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # centik gizleme 
    ax1.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax2------  EV SON 20 Shoot On Target FIRST HALF ------------
    ax2 = plt.subplot(gs[1, : ])
    df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_firsthalf','shots_on_target_home_firsthalf','shots_on_target_away_firsthalf']][:20]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    away_teams = list(df_aralarinda_shoot_all['away_team'][:20])
    place_away = list(df_aralarinda_shoot_all['place_away'][:20])
    match_round = list(df_aralarinda_shoot_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, away_teams, place_away, match_round)]
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax2.bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax2.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax2.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax2.set_title(f'{team_home} Son 20 Ev Maci Shoot On Target First Half (w= week, p= place)')
    ax2.set_ylabel('Count')
    ax2.set_ylim(0, df_aralarinda_shoot['shots_on_target_total_firsthalf'].max() + 3) 
    ax2.set_xticks(x_length + width + x_shift )
    ax2.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # Centik gizleme 
    ax2.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax3------  EV SON 20 Shoot On Target SECOND HALF ------------
    ax3 = plt.subplot(gs[2, : ])
    df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[df.home_team.str.contains(home_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf']][:20]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    away_teams = list(df_aralarinda_shoot_all['away_team'][:20])
    place_away = list(df_aralarinda_shoot_all['place_away'][:20])
    match_round = list(df_aralarinda_shoot_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, away_teams, place_away, match_round)]
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax3.bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax3.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax3.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax3.set_title(f'{team_home} Son 20 Home Maci Shoot On Target Second Half (w= week, p= place)')
    ax3.set_ylabel('Count')
    ax3.set_ylim(0, df_aralarinda_shoot['shots_on_target_total_secondhalf'].max() + 3) 
    ax3.set_xticks(x_length + width + x_shift )
    ax3.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # Centik gizleme 
    ax3.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax5------  DEP SON 20 Shoot On Target ------------
    ax4 = plt.subplot(gs[3, : ])
    df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total','shots_on_target_home','shots_on_target_away']][:20]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    home_teams = list(df_aralarinda_shoot_all['home_team'][:20])
    place_home = list(df_aralarinda_shoot_all['place_home'][:20])
    match_round = list(df_aralarinda_shoot_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, home_teams, place_home, match_round)]
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax4.bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax4.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax4.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax4.set_title(f'{team_away} Son 20 Dep Maci Shoot On Target (w= week, p= place)')
    ax4.set_ylabel('Count')
    ax4.set_ylim(0, df_aralarinda_shoot['shots_on_target_total'].max() + 3) 
    ax4.set_xticks(x_length + width + x_shift )
    ax4.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # centik gizleme 
    ax4.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax5------  DEP SON 20 Shoot On Target FIRST HALF ------------
    ax5 = plt.subplot(gs[4, : ])
    df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_firsthalf','shots_on_target_home_firsthalf','shots_on_target_away_firsthalf']][:20]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    home_teams = list(df_aralarinda_shoot_all['home_team'][:20])
    place_home = list(df_aralarinda_shoot_all['place_home'][:20])
    match_round = list(df_aralarinda_shoot_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, home_teams, place_home, match_round)]
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax5.bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax5.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax5.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax5.set_title(f'{team_away} Son 20 Dep Maci Shoot On Target First Half (w= week, p= place)')
    ax5.set_ylabel('Count')
    ax5.set_ylim(0, df_aralarinda_shoot['shots_on_target_total_firsthalf'].max() + 3) 
    ax5.set_xticks(x_length + width + x_shift )
    ax5.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # Centik gizleme 
    ax5.tick_params(axis='x', which='both', bottom=False, top=False)

    #------ax6------  DEP SON 20 Shoot On Target SECOND HALF ------------
    ax6 = plt.subplot(gs[5, : ])
    df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df[df.away_team.str.contains(away_team)].sort_values(by= 'date', ascending= False)
    df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf']][:20]
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    home_teams = list(df_aralarinda_shoot_all['home_team'][:20])
    place_home = list(df_aralarinda_shoot_all['place_home'][:20])
    match_round = list(df_aralarinda_shoot_all['match_round'][:20])
    xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, home_teams, place_home, match_round)]
    x_length = np.arange(len(shoot_dates))
    width = 0.25  # the width of the bars
    multiplier = 0
    shoot_dates = list(df_aralarinda_shoot_all['date'][:20].astype(str))  # Tarihleri string olarak alıyoruz
    for column in df_aralarinda_shoot.columns : 
        offset = width * multiplier
        rects = ax6.bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
        ax6.bar_label(rects, padding=3)
        multiplier += 1
        x_shift = 0.4
    ax6.legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
    ax6.set_title(f'{team_away} Son 20 Dep Maci Shoot On Target Second Half (w= week, p= place)')
    ax6.set_ylabel('Count')
    ax6.set_ylim(0, df_aralarinda_shoot['shots_on_target_total_secondhalf'].max() + 3) 
    ax6.set_xticks(x_length + width + x_shift )
    ax6.set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
    # Centik gizleme 
    ax6.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tight_layout()
    plt.show()
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if period: 
        fig, ax = plt.subplots(2,3,figsize = (21,9), layout='tight') 
        #------------  EV SON 10 Shoot On Target SEASON_PERIOD A GORE ------------

        df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total','shots_on_target_home','shots_on_target_away']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_shoot_all['away_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(shoot_dates, away_teams)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[0,0].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[0,0].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,0].set_title(f'{team_home} Son 10 Home Maci Shoot On Target Period = {period}')
        ax[0,0].set_ylabel('Count')
        ax[0,0].set_ylim(0, df_aralarinda_shoot['shots_on_target_total'].max() + 3) 
        ax[0,0].set_xticks(x_length + width + x_shift )
        ax[0,0].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,0].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  EV SON 10 Shoot On Target FIRST HALF SEASON_PERIOD A GORE ------------

        df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_firsthalf','shots_on_target_home_firsthalf','shots_on_target_away_firsthalf']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_shoot_all['away_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(shoot_dates, away_teams)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[0,1].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[0,1].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,1].set_title(f'{team_home} Son 10 Home Maci Shoot On Target First Half Period = {period}')
        ax[0,1].set_ylabel('Count')
        ax[0,1].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_firsthalf'].max() + 3) 
        ax[0,1].set_xticks(x_length + width + x_shift )
        ax[0,1].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,1].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  EV SON 10 Shoot On Target SECOND HALF SEASON_PERIOD A GORE ------------

        df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.home_team.str.contains(home_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_shoot_all['away_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(shoot_dates, away_teams)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[0,2].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[0,2].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,2].set_title(f'{team_home} Son 10 Home Maci Shoot On Target Period = {period}')
        ax[0,2].set_ylabel('Count')
        ax[0,2].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_secondhalf'].max() + 3) 
        ax[0,2].set_xticks(x_length + width + x_shift )
        ax[0,2].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,2].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  DEP SON 10 Shoot On Target SEASON_PERIOD A GORE ------------

        df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total','shots_on_target_home','shots_on_target_away']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_shoot_all['home_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(shoot_dates, home_teams)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[1,0].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[1,0].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,0].set_title(f'{team_away} Son 10 Dep Maci Shoot On Target Period = {period}')
        ax[1,0].set_ylabel('Count')
        ax[1,0].set_ylim(0, df_aralarinda_shoot['shots_on_target_total'].max() + 3) 
        ax[1,0].set_xticks(x_length + width + x_shift )
        ax[1,0].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,0].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  DEF SON 10 Shoot On Target FIRST HALF SEASON_PERIOD A GORE ------------

        df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_firsthalf','shots_on_target_home_firsthalf','shots_on_target_away_firsthalf']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_shoot_all['home_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(shoot_dates, home_teams)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[1,1].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[1,1].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,1].set_title(f'{team_away} Son 10 Dep Maci Shoot On Target First Half Period = {period}')
        ax[1,1].set_ylabel('Count')
        ax[1,1].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_firsthalf'].max() + 3) 
        ax[1,1].set_xticks(x_length + width + x_shift )
        ax[1,1].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,1].tick_params(axis='x', which='both', bottom=False, top=False)

        #------------  EV SON 10 Shoot On Target SECOND HALF SEASON_PERIOD A GORE ------------

        df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.away_team.str.contains(away_team) & (df.season_period == period)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_shoot_all['home_team'][:10])
        xticks_labels = [f"{date}\n{team}" for date, team in zip(shoot_dates, home_teams)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[1,2].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[1,2].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,2].set_title(f'{team_away} Son 10 Dep Maci Shoot On Target Period = {period}')
        ax[1,2].set_ylabel('Count')
        ax[1,2].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_secondhalf'].max() + 3) 
        ax[1,2].set_xticks(x_length + width + x_shift )
        ax[1,2].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,2].tick_params(axis='x', which='both', bottom=False, top=False)
        plt.tight_layout()
        plt.show()
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #------------  SIRALAMAYA GORE SON 10 HOME Shoot On Target ------------
    #min_sira, max_sira burda kullaniliyor 
    if max_away_sira and min_away_sira : 
        fig, ax = plt.subplots(2,3,figsize = (21,9), layout='tight')

        df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total','shots_on_target_home','shots_on_target_away']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_shoot_all['away_team'][:10])
        place_away = list(df_aralarinda_shoot_all['place_away'][:10])
        match_round = list(df_aralarinda_shoot_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, away_teams, place_away, match_round)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[0,0].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[0,0].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,0].set_title(f'{team_home} Son 10 Home Maci, Rakip Sirasi : {min_away_sira}-{max_away_sira} Arasinda (w:week, p:place)')
        ax[0,0].set_ylabel('Count')
        ax[0,0].set_ylim(0, df_aralarinda_shoot['shots_on_target_total'].max() + 3) 
        ax[0,0].set_xticks(x_length + width + x_shift )
        ax[0,0].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,0].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 HOME Shoot On Target FIRST HALF ------------

        df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_firsthalf','shots_on_target_home_firsthalf','shots_on_target_away_firsthalf']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_shoot_all['away_team'][:10])
        place_away = list(df_aralarinda_shoot_all['place_away'][:10])
        match_round = list(df_aralarinda_shoot_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, away_teams, place_away, match_round)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[0,1].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[0,1].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,1].set_title(f'{team_home} Son 10 Home Maci First Half, Rakip Sirasi : {min_away_sira}-{max_away_sira} Arasinda ')
        ax[0,1].set_ylabel('Count')
        ax[0,1].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_firsthalf'].max() + 3) 
        ax[0,1].set_xticks(x_length + width + x_shift )
        ax[0,1].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,1].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 HOME Shoot On Target SECOND HALF ------------
    
        df_aralarinda_shoot_all = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.home_team.str.contains(home_team) & (df.place_away >= min_away_sira) & (df.place_away <= max_away_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        away_teams = list(df_aralarinda_shoot_all['away_team'][:10])
        place_away = list(df_aralarinda_shoot_all['place_away'][:10])
        match_round = list(df_aralarinda_shoot_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, away_teams, place_away, match_round)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[0,2].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[0,2].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[0,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[0,2].set_title(f'{team_home} Son 10 Home Maci Second Half, Rakip Sirasi : {min_away_sira}-{max_away_sira} Arasinda')
        ax[0,2].set_ylabel('Count')
        ax[0,2].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_secondhalf'].max() + 3) 
        ax[0,2].set_xticks(x_length + width + x_shift )
        ax[0,2].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[0,2].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 DEP Shoot On Target ------------
    if max_home_sira and min_home_sira :
        df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total','shots_on_target_home','shots_on_target_away']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_shoot_all['home_team'][:10])
        place_home = list(df_aralarinda_shoot_all['place_home'][:10])
        match_round = list(df_aralarinda_shoot_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, home_teams, place_home, match_round)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[1,0].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[1,0].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,0].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,0].set_title(f'{team_away} Son 10 Dep Maci, Rakip Sirasi : {min_home_sira}-{max_home_sira} Arasinda (w:week, p:place)')
        ax[1,0].set_ylabel('Count')
        ax[1,0].set_ylim(0, df_aralarinda_shoot['shots_on_target_total'].max() + 3) 
        ax[1,0].set_xticks(x_length + width + x_shift )
        ax[1,0].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,0].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 DEP Shoot On Target FIRST HALF ------------

        df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_firsthalf','shots_on_target_home_firsthalf','shots_on_target_away_firsthalf']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_shoot_all['home_team'][:10])
        place_home = list(df_aralarinda_shoot_all['place_home'][:10])
        match_round = list(df_aralarinda_shoot_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, home_teams, place_home, match_round)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[1,1].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[1,1].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,1].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,1].set_title(f'{team_away} Son 10 Dep Maci First Half, Rakip Sirasi : {min_home_sira}-{max_home_sira} Arasinda ')
        ax[1,1].set_ylabel('Count')
        ax[1,1].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_firsthalf'].max() + 3) 
        ax[1,1].set_xticks(x_length + width + x_shift )
        ax[1,1].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,1].tick_params(axis='x', which='both', bottom=False, top=False)

    #------------  SIRALAMAYA GORE SON 10 DEP Shoot On Target SECOND HALF ------------
    
        df_aralarinda_shoot_all = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df[df.away_team.str.contains(away_team) & (df.place_home >= min_home_sira) & (df.place_home <= max_home_sira)].sort_values(by= 'date', ascending= False)
        df_aralarinda_shoot = df_aralarinda_shoot[['shots_on_target_total_secondhalf','shots_on_target_home_secondhalf','shots_on_target_away_secondhalf']][:10]
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        home_teams = list(df_aralarinda_shoot_all['home_team'][:10])
        place_home = list(df_aralarinda_shoot_all['place_home'][:10])
        match_round = list(df_aralarinda_shoot_all['match_round'][:10])
        xticks_labels = [f"{date}\n{team}\nw:{round} - p:{place}" for date, team, place, round in zip(shoot_dates, home_teams, place_home, match_round)]
        x_length = np.arange(len(shoot_dates))
        width = 0.25  # the width of the bars
        multiplier = 0
        shoot_dates = list(df_aralarinda_shoot_all['date'][:10].astype(str))  # Tarihleri string olarak alıyoruz
        for column in df_aralarinda_shoot.columns : 
            offset = width * multiplier
            rects = ax[1,2].bar(x_length + offset, df_aralarinda_shoot[column], width, label = column)
            ax[1,2].bar_label(rects, padding=3)
            multiplier += 1
            x_shift = 0.4
        ax[1,2].legend(loc='upper right', ncols= 3, fontsize= 7,  bbox_to_anchor=(1, 1))
        ax[1,2].set_title(f'{team_away} Son 10 Dep Maci Second Half, Rakip Sirasi : {min_home_sira}-{max_home_sira} Arasinda')
        ax[1,2].set_ylabel('Count')
        ax[1,2].set_ylim(0, df_aralarinda_shoot['shots_on_target_total_secondhalf'].max() + 3) 
        ax[1,2].set_xticks(x_length + width + x_shift )
        ax[1,2].set_xticklabels(xticks_labels, rotation= 70, ha= 'right', fontsize= 10)
        # centik gizleme 
        ax[1,2].tick_params(axis='x', which='both', bottom=False, top=False)

        
    plt.tight_layout()
    plt.show()




