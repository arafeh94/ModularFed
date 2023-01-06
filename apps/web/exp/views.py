import logging

from debugpy.common.messaging import Request
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render

# Create your views here.
from src import manifest
from src.apis import checkpoints_utils
from src.apis.fed_sqlite import FedDB
from src.apis.utils import smooth


def index(request: HttpRequest):
    db = FedDB(manifest.DB_PATH)
    tables = db.tables()
    requested_tables = list(request.GET)
    accuracies = {}
    for table in requested_tables:
        accuracies[table] = db.acc(table)
    return render(request, 'index.html', {'tables': tables, 'accuracies': accuracies})
