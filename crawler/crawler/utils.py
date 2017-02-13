# Dict for different languages
# homegate / newhome

FIELDS = {
    'Verkaufspreis': 'price_brutto',  # homegate
    'Preis': 'price_brutto',  # newhome
    'Preis Garage': 'additional_costs',  # newhome

    'Etage': 'floor',     # homegate
    'Stockwerk': 'floor', # newhome, immoscout24

    'Anzahl Etagen': 'num_floors',  # homegate
    'Etagen im Haus': 'num_floors',  # newhome
    'Anzahl Etagen des Objektes': 'num_floors', # immoscout24

    'Verfügbar': 'available',  # homegate
    'Bezug': 'available',  # newhome

    'Objekttyp': 'objecttype',  # homegate
    'Objektart': 'objecttype', # newhome

    'Zimmer': 'num_rooms',  # homegate / newhome
    'Anzahl Zimmer': 'num_rooms',

    'Wohnfläche': 'living_area',  # homegate / newhome

    'Baujahr': 'build_year',  # homegate, newhome, immoscout24

    'Nutzfläche': 'effective_area', # homegate, immoscout24

    'Kubatur': 'cubature',  # homegate, newhome, immoscout24

    'Raumhöhe': 'room_height',  # homegate

    'Grundstückfläche': 'plot_area',  # homegate, newhome, immoscout24

    'Zustand': 'condition',  # newhome

    'Letzte Renovation': 'last_renovation_year',  # homegate
    'Renoviert im Jahr': 'last_renovation_year',  # newhome
    'Letztes Renovationsjahr': 'last_renovation_year', # immoscout24

    'Immocode' : 'object_id',  # newhome
    'ImmoScout24-Code': 'object_id', # immoscout24

    'Objektnummer': 'reference_no', # newhome
    'Referenz': 'reference_no', # immoscout24
}
