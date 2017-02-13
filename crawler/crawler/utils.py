# Dict for different languages
# homegate / newhome

FIELDS = {
    'Verkaufspreis': 'price_brutto',  # homegate
    'Preis': 'price_brutto',  # newhome
    'Preis Garage': 'additional_costs',  # newhome

    'Etage': 'floor',     # homegate
    'Stockwerk': 'floor', # newhome

    'Anzahl Etagen': 'num_floors',  # homegate
    'Etagen im Haus': 'num_floors',  # newhome

    'Verfügbar': 'available',  # homegate
    'Bezug': 'available',  # newhome

    'Objekttyp': 'objecttype',  # homegate
    'Objektart': 'objecttype', # newhome

    'Zimmer': 'num_rooms',  # homegate / newhome
    'Anzahl Zimmer': 'num_rooms',

    'Wohnfläche': 'living_area',  # homegate / newhome

    'Baujahr': 'build_year',  # homegate / newhome
    
    'Nutzfläche': 'effective_area', # homegate

    'Kubatur': 'cubature',  # homegate / newhome

    'Raumhöhe': 'room_height',  # homegate
   
    'Grundstückfläche': 'plot_area',  # homegate / newhome
    
    'Zustand': 'condition',  # newhome

    'Letzte Renovation': 'last_renovation_year',  # homegate
    'Renoviert im Jahr': 'last_renovation_year',  # newhome

    'Immocode' : 'object_id',  # newhome

    'Objektnummer': 'reference_no'  # newhome
 
}
