# Dict for different languages
# homegate / newhome

FIELDS = {
    'Verkaufspreis': 'price_brutto',  # homegate, immoscout24
    'Preis': 'price_brutto',  # newhome
    'Kaufpreis': 'price_brutto',  # immoscout24
    'Preis Garage': 'additional_costs',  # newhome

    'Etage': 'floor',     # homegate, urbanhome
    'Stockwerk': 'floor', # newhome, immoscout24

    'Anzahl Etagen': 'num_floors',  # homegate
    'Etagen im Haus': 'num_floors',  # newhome, urbanhome
    'Anzahl Etagen des Objektes': 'num_floors', # immoscout24

    'Verfügbar': 'available',  # homegate, urbanhome
    'Bezug': 'available',  # newhome

    'Objekttyp': 'objecttype',  # homegate
    'Objektart': 'objecttype', # newhome

    'Zimmer': 'num_rooms',  # homegate, newhome, urbanhome
    'Anzahl Zimmer': 'num_rooms',

    'Wohnfläche': 'living_area',  # homegate, newhome, urbanhome

    'Baujahr': 'build_year',  # homegate, newhome, immoscout24

    'Nutzfläche': 'effective_area', # homegate, immoscout24

    'Kubatur': 'cubature',  # homegate, newhome, immoscout24

    'Raumhöhe': 'room_height',  # homegate

    'Grundstückfläche': 'plot_area',  # homegate, newhome, immoscout24
    'Grundstück': 'plot_area',  # urbanhome

    'Zustand': 'condition',  # newhome

    'Letzte Renovation': 'last_renovation_year',  # homegate
    'Renoviert im Jahr': 'last_renovation_year',  # newhome
    'Letztes Renovationsjahr': 'last_renovation_year', # immoscout24

    'Immocode' : 'object_id',  # newhome
    'ImmoScout24-Code': 'object_id', # immoscout24
    'Inserate-Nr': 'object_id', # urbanhome

    'Objektnummer': 'reference_no', # newhome
    'Referenz': 'reference_no', # immoscout24
    'Objekt-Referenz': 'reference_no', # urbanhome
}
