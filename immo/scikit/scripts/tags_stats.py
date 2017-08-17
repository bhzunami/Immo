import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

s1 = pd.Series([53227, 48693, 47116, 43824, 43334, 42726, 39891,
                38687, 38448, 38275, 37269, 34009, 33855, 31332,
                29775, 29324, 29319, 25466, 25126, 24492, 24465,
                23346, 21971, 20756, 17457, 17254, 16543, 15903,
                15379, 15107, 14879, 14790, 14488, 14387, 14339,
                13454, 13340, 13186, 12822, 11021, 10890, 10577,
                10254, 10006, 9939, 9901, 9720, 9663, 9247, 8982])

s2 = pd.Series(['balkon','zimmer', 'küche', 'badewanne', 'wohnung',
                'aussicht', 'terrasse', 'garage', 'sitzplatz', 'gross',
                'wc','keller','lage','garten','lift','parkplatz','dusche','kinderfreundlich','schlafzimmer','cheminée','ruhig','haus','eingang','tv','waschküche','wohnzimmer','raum','badezimmer','primarschule','kindergarten','rollstuhlgängig','neubau','erdgeschoss','modern','heizung','verkehr','zentral','esszimmer','einkaufen','waschmaschine','liegenschaft','oberstufe','quartier','villa','abstellplatz','autobahnanschluss','sommer','minergie','lavabo','anschluss'])

tags = pd.DataFrame({'anzahl': s1, 'tag': s2})

import pdb
b = sns.barplot(x=tags.tag, y=tags.anzahl)
plt.xlabel('Tags')
plt.ylabel('Anzahl')
plt.title('Anzahl der Tags in Beschreibung')
plt.xticks(rotation='90')
plt.tight_layout()
#for text in b.get_xticklabels():
#    text.set_text(text.get_text().replace("_", " ").title())
plt.savefig("images/analysis/tags.png", dpi=250)
plt.clf()
plt.close()