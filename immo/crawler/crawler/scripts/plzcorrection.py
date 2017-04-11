import os


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, defer
from sqlalchemy.sql.expression import bindparam
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import select, update
from sqlalchemy.types import NullType, ARRAY, Integer
from models import Advertisement, Municipality
import json


engine = create_engine(os.environ.get('DATABASE_URL'), echo=True)
Session = sessionmaker(bind=engine)

# start transaction
session = Session()

try:
    # get all entries which have an unparsed municipality
    # info: no matching where no plz exists (join vs left join in query)
    rs = session.execute("""
    select min(m.id) MID, municipality_unparsed, count(municipality_unparsed), min(substring(a.municipality_unparsed from E'^.*? (.*)$')) ALT_NAME, string_agg(a.id::text, ',') AIDs

    from advertisements a

    join municipalities m on cast(m.zip as text) = split_part(a.municipality_unparsed, ' ', 1)

    where municipalities_id is null

    group by municipality_unparsed
    order by 3 desc
    """)

    # group by found municipality and add all found alternative_names
    municipalities = {}

    ads = []

    for row in rs:
        if row.mid not in municipalities:
            municipalities[row.mid] = []

        municipalities[row.mid].append(row.alt_name)

        ads.append({'aids': row.aids.split(','), 'mid': row.mid})

    print(municipalities)

    prepared = []

    for mid in municipalities:
        prepared.append({'mid': mid, '_alternate_names': json.dumps(municipalities[mid])})

    stmt2 = update(Municipality).where(Municipality.id == bindparam('mid')).values(alternate_names=Municipality.alternate_names + bindparam('_alternate_names'))

    print(str(stmt2.compile(dialect=postgresql.dialect())))

    session.execute(stmt2, prepared)

    # set municipalities_id for the ads
    session.execute("""
    update advertisements a
    set municipalities_id = m.id
    from municipalities m
    where cast(m.zip as text) = split_part(a.municipality_unparsed, ' ', 1)
    and a.municipalities_id is null
    """)

    session.commit()
except:
    session.rollback()
    raise
