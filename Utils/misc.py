import datetime
import uuid


def get_date_string():
    return datetime.date.today().strftime('%Y%m%d')


def get_uuid_name():
    return str(uuid.uuid4()).replace('-', '')
